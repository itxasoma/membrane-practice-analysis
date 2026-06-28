#!/usr/bin/env python3
"""
corr.py — Dipolar rotational autocorrelation C_rot(t) for shell-selected waters.

Shell-conditioned ensemble correlation:
    For each lag dt, average over all possible time origins t0.
    For each (t0, t0+dt), correlate only waters present in BOTH frames (by resid).

Assumes XYZ atom lines have the format:
    resid atomname x y z

Example:
    2 OH2 32.1134 16.4525 6.8640

Usage:
    Set TEMP_TAG to match the simulation temperature (e.g. "290.5").
    The output CSVs will be named 2.corr_<shell>_<TEMP_TAG>.csv.
"""

import os
import re
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR  = os.path.join(BASE_DIR, '../figures')
ANA_DIR  = os.path.join(BASE_DIR, '../1_Analysis/Vctt-fast-307.5/')

os.makedirs(FIG_DIR, exist_ok=True)

# Timestep between consecutive frames in the trajectory XYZ files (ps).
# With FRAME_STEP=10, timestep=2fs, DCDfreq=10 --> each XYZ frame = 10×10×2fs = 0.20 ps
DT_PS = 0.20

STRIDE     = 1
MAX_LAG_PS = 10.0
TEMP_TAG   = "307.5"

SHELLS = [
    ('trajectory_d0_3.xyz',   r'$0$-$3\,\AA$',   '#F0A500'),
    ('trajectory_d3_5.xyz',   r'$3$-$5\,\AA$',   '#E87D72'),
    ('trajectory_d5_10.xyz',  r'$5$-$10\,\AA$',  '#56A0D3'),
    ('trajectory_d10_15.xyz', r'$10$-$15\,\AA$', '#845B97'),
]

SHELL_TAGS = {
    r'$0$-$3\,\AA$':   '0_3A',
    r'$3$-$5\,\AA$':   '3_5A',
    r'$5$-$10\,\AA$':  '5_10A',
    r'$10$-$15\,\AA$': '10_15A',
}



def _parse_frame_header(header):
    header = header.strip()
    m = re.search(r'Frame\s+(\d+)', header)
    if m:
        return int(m.group(1))
    m = re.search(r'Timeframe\s*=\s*(\d+)', header)
    if m:
        return int(m.group(1))
    return None


def _read_nonempty_line(f):
    while True:
        line = f.readline()
        if not line:
            return None
        if line.strip():
            return line


def iter_xyz_frames(filename, stride=1):
    """
    Yield (frame_idx, frame_dict) for every stride-th frame.

    frame_dict = {resid (int): mu (np.ndarray shape (3,))}
    where mu = r_O - 0.5*(r_H1 + r_H2)  (raw, not normalised)
    """
    with open(filename, 'r') as f:
        raw_frame_idx = 0

        while True:
            line = _read_nonempty_line(f)
            if line is None:
                break

            try:
                n_atoms = int(line.strip())
            except ValueError:
                raise ValueError(f'Invalid atom-count line in {filename}: {line!r}')

            header = f.readline()
            if not header:
                break

            if raw_frame_idx % stride != 0:
                atoms_read = 0
                while atoms_read < n_atoms:
                    line = f.readline()
                    if not line:
                        break
                    if not line.strip():
                        continue
                    atoms_read += 1
                raw_frame_idx += 1
                continue

            atoms = []
            while len(atoms) < n_atoms:
                line = f.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                resid = int(parts[0])
                atom  = parts[1]
                x, y, z = map(float, parts[2:5])
                atoms.append((resid, atom, x, y, z))

            if len(atoms) % 3 != 0:
                atoms = atoms[:(len(atoms) // 3) * 3]

            frame = {}
            for i in range(0, len(atoms), 3):
                r0, a0, x0, y0, z0 = atoms[i]
                r1, a1, x1, y1, z1 = atoms[i + 1]
                r2, a2, x2, y2, z2 = atoms[i + 2]

                if not (r0 == r1 == r2):
                    continue
                if a0 != 'OH2':
                    continue

                O  = np.array([x0, y0, z0], dtype=np.float64)
                H1 = np.array([x1, y1, z1], dtype=np.float64)
                H2 = np.array([x2, y2, z2], dtype=np.float64)

                mu = O - 0.5 * (H1 + H2)
                frame[r0] = mu

            frame_idx = _parse_frame_header(header)
            if frame_idx is None:
                frame_idx = raw_frame_idx

            yield frame_idx, frame
            raw_frame_idx += 1


def load_dipole_frames(filename, stride=1):
    frames  = []
    indices = []
    for frame_idx, frame in iter_xyz_frames(filename, stride=stride):
        indices.append(frame_idx)
        frames.append(frame)
    return indices, frames


def compute_crot_shell_conditioned(frames, max_lag_frames):
    """
    Compute C_rot(t) averaging over all t0 origins.

    For each (t0, t0+dt), only waters present in BOTH frames (by resid) contribute.
    Normalisation: denom = sum(mu0·mu0), so C(0) = 1 exactly.

    Returns
    -------
    t_ps       : (n_lags,) float64
    C_rot      : (n_lags,) float64   — NaN where no pairs were found
    pair_count : (n_lags,) int64     — diagnostic only, not saved
    t0_count   : (n_lags,) int64     — diagnostic only, not saved
    """
    n_frames = len(frames)
    n_lags   = min(max_lag_frames + 1, n_frames)

    numer      = np.zeros(n_lags, dtype=np.float64)
    denom      = np.zeros(n_lags, dtype=np.float64)
    pair_count = np.zeros(n_lags, dtype=np.int64)
    t0_count   = np.zeros(n_lags, dtype=np.int64)

    for dt in range(n_lags):
        for t0 in range(n_frames - dt):
            f0 = frames[t0]
            ft = frames[t0 + dt]

            common = set(f0.keys()) & set(ft.keys())
            if not common:
                continue

            keys = list(common)
            mu0  = np.array([f0[k] for k in keys], dtype=np.float64)
            mut  = np.array([ft[k] for k in keys], dtype=np.float64)

            numer[dt]      += np.sum(mu0 * mut)
            denom[dt]      += np.sum(mu0 * mu0)
            pair_count[dt] += len(keys)
            t0_count[dt]   += 1

    C_rot        = np.full(n_lags, np.nan, dtype=np.float64)
    valid        = denom > 0
    C_rot[valid] = numer[valid] / denom[valid]
    if valid[0]:
        C_rot[0] = 1.0

    t_ps = np.arange(n_lags, dtype=np.float64) * DT_PS * STRIDE
    return t_ps, C_rot, pair_count, t0_count


# output 

def save_csv(label, t_ps, C_rot):
    """Save only t_ps and C_rot — pair/t0 counts are diagnostic, not needed downstream."""
    tag  = SHELL_TAGS[label]
    out  = os.path.join(FIG_DIR, f'2.corr_{tag}_{TEMP_TAG}.csv')
    mask = np.isfinite(C_rot)
    data = np.column_stack([t_ps[mask], C_rot[mask]])
    header = f't_ps C_rot  label="{label}" temp="{TEMP_TAG}"'
    np.savetxt(out, data, header=header, comments='# ')
    print(f'  Saved {os.path.relpath(out)}')


# main 

if __name__ == '__main__':
    dt_eff         = DT_PS * STRIDE
    max_lag_frames = int(round(MAX_LAG_PS / dt_eff))

    print(f'Settings: STRIDE={STRIDE}, dt_eff={dt_eff:.4f} ps, '
          f'max_lag={max_lag_frames} frames = {MAX_LAG_PS} ps')
    print(f'TEMP_TAG = {TEMP_TAG}')
    print('Averaging over all t0, shell-conditioned pairs (resid intersection).\n')

    for fname, label, color in SHELLS:
        path = os.path.join(ANA_DIR, fname)
        if not os.path.exists(path):
            print(f'[skip] {fname} not found, run trajectory_to_xyz.py first\n')
            continue

        print(f'── {label}  ({fname}) ──')
        frame_ids, frames = load_dipole_frames(path, stride=STRIDE)

        if not frames:
            print('  No frames found — check the XYZ file.\n')
            continue

        n_w = np.array([len(fr) for fr in frames], dtype=int)
        print(f'  Frames loaded : {len(frames)}')
        print(f'  Waters/frame  : min={n_w.min()}  mean={n_w.mean():.1f}  max={n_w.max()}')

        t_ps, C_rot, pair_count, t0_count = compute_crot_shell_conditioned(
            frames, max_lag_frames
        )

        idx2  = min(len(t_ps) - 1, int(round(2.0  / dt_eff)))
        idx10 = min(len(t_ps) - 1, int(round(10.0 / dt_eff)))
        print(f'  C(0 ps)  = {C_rot[0]:.4f}')
        print(f'  C(2 ps)  = {C_rot[idx2]:.4f}')
        print(f'  C(10 ps) = {C_rot[idx10]:.4f}')
        print(f'  pairs@0  = {pair_count[0]}  |  pairs@end = {pair_count[np.isfinite(C_rot)][-1]}')

        save_csv(label, t_ps, C_rot)
        print()

    print('Done. Run corr-plots.py to generate figures.')