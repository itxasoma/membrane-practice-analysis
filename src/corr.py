#!/usr/bin/env python3
"""
corr.py — Dipolar rotational autocorrelation C_rot(t) for shell-selected waters.

Shell-conditioned ensemble correlation:
    For each lag dt, average over all possible time origins t0.
    For each (t0, t0+dt), correlate only waters that are present in BOTH frames.

This version assumes XYZ atom lines have the format:
    resid atomname x y z

Example:
    2 OH2 32.1134 16.4525 6.8640
"""

import os
import re
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR  = os.path.join(BASE_DIR, '../figures')
ANA_DIR  = os.path.join(BASE_DIR, '../1_Analysis/Vctt')

os.makedirs(FIG_DIR, exist_ok=True)

DT_PS = 0.02
STRIDE = 1
MAX_LAG_PS = 10.0

SHELLS = [
    ('trajectory_d0_3.xyz',   r'$0$-$3\,\AA$',   '#E87D72'),
    ('trajectory_d3_5.xyz',   r'$3$-$5\,\AA$',   '#F0A500'),
    ('trajectory_d5_10.xyz',  r'$5$-$10\,\AA$',  '#845B97'),
    ('trajectory_d10_15.xyz', r'$10$-$15\,\AA$', '#56A0D3'),
]


def _shell_tag(label):
    tag = label
    tag = tag.replace('$', '')
    tag = tag.replace('\\,', '')
    tag = tag.replace('\\AA', 'A')
    tag = tag.replace('–', '-')
    tag = tag.replace(' ', '')
    tag = tag.replace('-', '_')
    return tag


def parse_frame_header(header):
    header = header.strip()
    m = re.search(r'Frame\s+(\d+)', header)
    if m:
        return int(m.group(1))
    m = re.search(r'Timeframe\s*=\s*(\d+)', header)
    if m:
        return int(m.group(1))
    return None


def read_nonempty_line(f):
    while True:
        line = f.readline()
        if not line:
            return None
        if line.strip():
            return line


def iter_xyz_frames(filename, stride=1):
    """
    Yield frames as:
        frame_idx, frame_dict

    where frame_dict = {resid: mu}
    and mu is the raw dipole vector:
        mu = r_O - 0.5 * (r_H1 + r_H2)
    """
    with open(filename, 'r') as f:
        raw_frame_idx = 0

        while True:
            line = read_nonempty_line(f)
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
                atom = parts[1]
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

            frame_idx = parse_frame_header(header)
            if frame_idx is None:
                frame_idx = raw_frame_idx

            yield frame_idx, frame
            raw_frame_idx += 1


def load_dipole_frames(filename, stride=1):
    frames = []
    indices = []
    for frame_idx, frame in iter_xyz_frames(filename, stride=stride):
        indices.append(frame_idx)
        frames.append(frame)
    return indices, frames


def compute_crot_shell_conditioned(frames, max_lag_frames):
    n_frames = len(frames)
    n_lags = min(max_lag_frames + 1, n_frames)

    numer = np.zeros(n_lags, dtype=np.float64)
    denom = np.zeros(n_lags, dtype=np.float64)
    pair_count = np.zeros(n_lags, dtype=np.int64)
    t0_count = np.zeros(n_lags, dtype=np.int64)

    for dt in range(n_lags):
        for t0 in range(n_frames - dt):
            f0 = frames[t0]
            ft = frames[t0 + dt]

            common = set(f0.keys()) & set(ft.keys())
            if not common:
                continue

            keys = list(common)
            mu0 = np.array([f0[k] for k in keys], dtype=np.float64)
            mut = np.array([ft[k] for k in keys], dtype=np.float64)

            numer[dt] += np.sum(mu0 * mut)
            denom[dt] += np.sum(mu0 * mu0)
            pair_count[dt] += len(keys)
            t0_count[dt] += 1

    C_rot = np.full(n_lags, np.nan, dtype=np.float64)
    valid = denom > 0
    C_rot[valid] = numer[valid] / denom[valid]

    if valid[0]:
        C_rot[0] = 1.0

    t_ps = np.arange(n_lags, dtype=np.float64) * DT_PS * STRIDE
    return t_ps, C_rot, pair_count, t0_count


def save_csv(label, color, t_ps, C_rot, pair_count, t0_count, suffix=''):
    tag = _shell_tag(label)
    out = os.path.join(FIG_DIR, f'2.corr{suffix}_{tag}.csv')

    mask = np.isfinite(C_rot)
    data = np.column_stack([
        t_ps[mask],
        C_rot[mask],
        pair_count[mask],
        t0_count[mask],
    ])

    header = (
        f't_ps C_rot pair_count t0_count '
        f'label="{label}" color="{color}"'
    )
    np.savetxt(out, data, header=header, comments='# ')
    print(f'Saved {os.path.relpath(out)}')


if __name__ == '__main__':
    dt_eff = DT_PS * STRIDE
    max_lag_frames = int(round(MAX_LAG_PS / dt_eff))

    print(f'STRIDE={STRIDE}, dt_eff={dt_eff:.4f} ps, MAX_LAG_PS={MAX_LAG_PS}')
    print('Averaging over all possible t0, shell-conditioned pairs only.\n')

    for fname, label, color in SHELLS:
        path = os.path.join(ANA_DIR, fname)
        if not os.path.exists(path):
            print(f'[skip] {fname} not found')
            continue

        print(f'── {label} ({fname}) ──')
        frame_ids, frames = load_dipole_frames(path, stride=STRIDE)

        if not frames:
            print('  No frames found\n')
            continue

        n_w = np.array([len(fr) for fr in frames], dtype=int)
        print(f'  Frames: {len(frames)}')
        print(f'  Waters/frame: min={n_w.min()} mean={n_w.mean():.1f} max={n_w.max()}')

        t_ps, C_rot, pair_count, t0_count = compute_crot_shell_conditioned(
            frames, max_lag_frames
        )

        idx2 = min(len(t_ps) - 1, int(round(2.0 / dt_eff)))
        print(f'  C(0 ps)  = {C_rot[0]:.4f}')
        print(f'  C(2 ps)  = {C_rot[idx2]:.4f}')
        print(f'  pairs@0  = {pair_count[0]}')
        print(f'  pairs@end= {pair_count[np.isfinite(C_rot)][-1]}')

        save_csv(label, color, t_ps, C_rot, pair_count, t0_count)
        print()

    print('Done. Run corr-plots.py afterwards.')