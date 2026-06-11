#!/usr/bin/env python3
"""
corr.py — Dipolar rotational autocorrelation function C_rot(t) for water
          molecules in the hydration shell of a DMPC membrane.

Definition (Informe_practica, eq. 1):

    C_rot(t) = < mu_hat(t0+t) · mu_hat(t0) >

where mu_hat is a unit vector (denominator = 1 always), and the average is:
  1. Over all water molecules present in BOTH frames t0 and t0+dt
  2. Over all t0 starting times using CIRCULAR (periodic) time averaging,
     so every lag dt has the same number of t0 origins → equal error bars.

Molecule tracking:
  - Waters are identified by their index in the XYZ file (stable across frames
    because trajectory_to_xyz.py writes them in consistent MDAnalysis order).
  - For each pair (t0, t0+dt mod N), only molecules present in both frames
    contribute. The intersection is taken by index.

Two estimates per shell:
  1. Full average  — circular average over all t0, all common molecules
  2. Last-frame    — only the last frame as t0 (single-origin, noisier;
                     useful to check that the final config is representative)

Input:  1_Analysis/trajectory_d*.xyz  (produced by trajectory_to_xyz.py)
Output: figures/2.corr_<tag>.csv           — per-shell full-average data
        figures/2.corr_lastframe_<tag>.csv — per-shell last-frame data

Run corr-plots.py afterwards to generate all PDF figures from the CSVs.
"""

import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR  = os.path.join(BASE_DIR, '../figures')
ANA_DIR  = os.path.join(BASE_DIR, '../1_Analysis')

os.makedirs(FIG_DIR, exist_ok=True)

# Timestep between consecutive frames (ps).
# DCDfreq=10 and timestep=2fs → one frame every 10×2 fs = 20 fs = 0.02 ps.
DT_PS = 0.02

# Read every STRIDE-th frame.
# STRIDE=1: full resolution (0.02 ps). Recommended — the fast librational
# decay happens in the first ~0.5 ps and needs fine sampling.
STRIDE = 1

# Compute C_rot only up to MAX_LAG_PS picoseconds.
MAX_LAG_PS = 10.0

# Shell definitions: (filename, label, color)
SHELLS = [
    ('trajectory_d0_3.xyz',   r'$0$-$3\,\AA$',   '#E87D72'),   # red
    ('trajectory_d3_5.xyz',   r'$3$-$5\,\AA$',   '#F0A500'),   # orange
    ('trajectory_d5_10.xyz',  r'$5$-$10\,\AA$',  '#845B97'),   # violet
    ('trajectory_d10_15.xyz', r'$10$-$15\,\AA$', '#56A0D3'),   # blue
]


# ── XYZ reader (strided, float32) ─────────────────────────────────────────────

def read_xyz(filename, stride=1):
    """
    Parse trajectory.xyz, keeping only every `stride`-th frame.
    Skipped frames are fast-forwarded without parsing.
    Uses float32 to halve memory usage.

    Returns
    -------
    frames : list of (n_atoms, 4) arrays — columns: atom_index, x, y, z
             Each row corresponds to one atom; atom_index is its position
             within the frame (0-based), used as a stable molecule identifier.
    n_waters_per_frame : list of int — number of water molecules per frame
    """
    frames = []
    frame_index = 0
    with open(filename) as f:
        while True:
            line = f.readline()
            if not line:
                break
            try:
                n_atoms = int(line.strip())
            except ValueError:
                break
            f.readline()  # skip frame header
            if frame_index % stride == 0:
                atoms = []
                for _ in range(n_atoms):
                    parts = f.readline().split()
                    atoms.append(np.array(parts[1:4], dtype=np.float32))
                frames.append(np.stack(atoms))   # (n_atoms, 3)
            else:
                for _ in range(n_atoms):
                    f.readline()
            frame_index += 1
    return frames


# ── Build dipoles (vectorized) ────────────────────────────────────────────────

def build_dipoles(frames):
    """
    Vectorized dipole builder. Assumes strict OH2/H1/H2 ordering in triplets,
    as produced by trajectory_to_xyz.py.

    mu = r_O - 0.5*(r_H1 + r_H2)
    mu_hat = mu / |mu|

    Each frame may have a different number of water molecules (n_w varies
    because the shell is distance-based). We keep the per-frame arrays
    separate so molecule tracking can use index-based intersection.

    Returns
    -------
    dipoles : list of (n_w, 3) float32 arrays, one per frame.
              Molecule i in frame t corresponds to molecule i in any other
              frame (stable MDAnalysis ordering). Common molecules between
              two frames are those with indices 0..min(n_w_t0, n_w_t1)-1.
    """
    dipoles = []
    for coords in frames:
        n = (len(coords) // 3) * 3
        c = coords[:n].reshape(-1, 3, 3).astype(np.float32)  # (n_w, atom, xyz)
        # c[:, 0] = O,  c[:, 1] = H1,  c[:, 2] = H2
        mu = c[:, 0] - 0.5 * (c[:, 1] + c[:, 2])
        norms = np.linalg.norm(mu, axis=1, keepdims=True)
        dipoles.append(mu / np.where(norms > 1e-10, norms, 1.0))
    return dipoles


# ── C_rot: circular time average ──────────────────────────────────────────────

def compute_crot(dipoles, max_lag_frames):
    """
    Compute C_rot(t) with CIRCULAR (periodic) time averaging.

    For every lag dt, all N frames are used as t0:
        t1 = (t0 + dt) mod N

    Only molecules present in BOTH frames contribute (index intersection):
        n_common = min(n_w[t0], n_w[t1])
        contribution = mean over molecules 0..n_common-1 of mu(t1)·mu(t0)

    This gives exactly N origins for every lag dt → equal error bars for all lags.

    Parameters
    ----------
    dipoles        : list of (n_w_i, 3) float32 arrays
    max_lag_frames : int, maximum lag in frames

    Returns
    -------
    t_ps  : (max_lag_frames+1,) array — lag times in ps
    C_rot : (max_lag_frames+1,) array — C_rot(0) = 1
    """
    N = len(dipoles)
    C_rot = np.zeros(max_lag_frames + 1)

    for dt in range(max_lag_frames + 1):
        acc = 0.0
        for t0 in range(N):
            t1 = (t0 + dt) % N
            n_common = min(dipoles[t0].shape[0], dipoles[t1].shape[0])
            d0 = dipoles[t0][:n_common].astype(np.float64)
            d1 = dipoles[t1][:n_common].astype(np.float64)
            acc += np.sum(d0 * d1, axis=1).mean()
        C_rot[dt] = acc / N   # equal N contributions for every dt

    C_rot /= C_rot[0]   # enforce C_rot(0) = 1

    dt_eff = DT_PS * STRIDE
    t_ps   = np.arange(max_lag_frames + 1) * dt_eff
    return t_ps, C_rot


# ── Last-frame C_rot ──────────────────────────────────────────────────────────

def compute_crot_lastframe(dipoles, max_lag_frames):
    """
    Compute C_rot(t) using ONLY the last frame as t_0, going backwards.

    C_rot(lag) = mean over common molecules of  mu(T-1) · mu(T-1-lag)

    Single-origin estimate: fast but noisier. If the final configuration is
    representative of equilibrium, this should agree with compute_crot.
    """
    N      = len(dipoles)
    t0_idx = N - 1
    C_rot  = np.zeros(max_lag_frames + 1)

    for lag in range(min(max_lag_frames + 1, N)):
        t_lag    = t0_idx - lag
        n_common = min(dipoles[t0_idx].shape[0], dipoles[t_lag].shape[0])
        d0   = dipoles[t0_idx][:n_common].astype(np.float64)
        d_lag = dipoles[t_lag][:n_common].astype(np.float64)
        C_rot[lag] = np.sum(d0 * d_lag, axis=1).mean()

    C_rot /= C_rot[0]

    dt_eff = DT_PS * STRIDE
    t_ps   = np.arange(max_lag_frames + 1) * dt_eff
    return t_ps, C_rot


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _shell_tag(label):
    return (label.replace('$', '').replace('\\', '')
                 .replace(' ', '').replace(',', '')
                 .replace('–', '_').replace('-', '_')
                 .replace('AA', 'A'))


def save_csv(label, color, t_ps, C_rot, suffix=''):
    tag = _shell_tag(label)
    out = os.path.join(FIG_DIR, f'2.corr{suffix}_{tag}.csv')
    np.savetxt(out,
               np.column_stack([t_ps, C_rot]),
               header=f't_ps C_rot  label="{label}"  color="{color}"',
               comments='# ')
    print(f'Saved {os.path.relpath(out)}')


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    dt_eff         = DT_PS * STRIDE
    max_lag_frames = int(MAX_LAG_PS / dt_eff)
    print(f'Settings: STRIDE={STRIDE}, dt_eff={dt_eff:.4f} ps, '
          f'max_lag={max_lag_frames} frames = {MAX_LAG_PS} ps')
    print('Time averaging: CIRCULAR (periodic) — equal N origins per lag\n')

    for fname, label, color in SHELLS:
        path = os.path.join(ANA_DIR, fname)
        if not os.path.exists(path):
            print(f'[skip] {fname} not found\n')
            continue

        print(f'── {label}  ({fname}) ──')
        print(f'  Reading (stride={STRIDE})...')
        frames = read_xyz(path, stride=STRIDE)
        print(f'  {len(frames)} frames loaded')

        print('  Building dipole vectors...')
        dipoles = build_dipoles(frames)
        n_w = [d.shape[0] for d in dipoles]
        print(f'  Waters/frame: min={min(n_w)}  mean={np.mean(n_w):.0f}  max={max(n_w)}')

        print(f'  Computing C_rot (circular avg, up to {MAX_LAG_PS} ps)...')
        t_ps, C_rot = compute_crot(dipoles, max_lag_frames)
        print(f'  C_rot(0)={C_rot[0]:.4f}  '
              f'C_rot(t=2ps)={C_rot[int(2.0/dt_eff)]:.4f}  '
              f'C_rot(t={MAX_LAG_PS}ps)={C_rot[-1]:.4f}')
        save_csv(label, color, t_ps, C_rot, suffix='')

        print('  Computing C_rot (last-frame estimate)...')
        t_ps_last, C_last = compute_crot_lastframe(dipoles, max_lag_frames)
        save_csv(label, color, t_ps_last, C_last, suffix='_lastframe')
        print()

    print('Done. Run corr-plots.py to generate all PDF figures.')
