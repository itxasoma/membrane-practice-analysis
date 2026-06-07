#!/usr/bin/env python3
"""
stats.py — Extra statistical analysis of water mobility in the DMPC hydration shell.

Complements corr.py (dipolar rotational autocorrelation) with translational
mobility diagnostics that are not strictly required by the assignment but are
very useful for interpretation:

  Plot 3a  MSD(t)         — Mean squared displacement, full shell average.
                            Slope in the diffusive regime → diffusion coefficient D.
  Plot 3b  VACF(t)        — Velocity autocorrelation function via central finite
                            differences. Zero-crossing gives the velocity
                            decorrelation timescale.
  Plot 3c  MSD per Z-slab — MSD split into N_SLICES Z-layers. Answers the
                            question: is there a layer where mobility is
                            suppressed (close to headgroups) vs. bulk-like?

Input:   1_Analysis/trajectory.xyz   (produced by trajectory_to_xyz.py)
Outputs: figures/3a.msd.pdf / .csv
         figures/3b.vacf.pdf / .csv
         figures/3c.msd_per_slice.pdf / .csv

Color palette (matching existing scripts):
  MSD  : #4878CF  (blue  — translational)
  VACF : #6ACC65  (green — velocity)
  Slabs: matplotlib default cycle (up to N_SLICES distinct hues)
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
FIG_DIR   = os.path.join(BASE_DIR, '../figures')
TRAJ_FILE = os.path.join(BASE_DIR, '../1_Analysis/trajectory.xyz')

style_candidates = [
    os.path.join(BASE_DIR, 'lib/science.mplstyle'),
    os.path.join(BASE_DIR, '../mplstyle/science.mplstyle'),
]
for style_file in style_candidates:
    if os.path.exists(style_file):
        plt.style.use(style_file)
        break

os.makedirs(FIG_DIR, exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────────────

OXYGEN_LABEL = 'OH2'   # atom name for water oxygen in TIP3P / CHARMM

# Timestep between consecutive frames in the trajectory (ps).
# DCDfreq=10 and timestep=2 fs → one frame every 10×2 fs = 20 fs = 0.02 ps.
DT_PS    = 0.02
MAX_DT   = 500         # maximum lag in frames
N_SLICES = 5           # number of Z-slabs for per-layer MSD

C_MSD  = '#4878CF'     # blue  — translational
C_VACF = '#6ACC65'     # green — velocity


# ── XYZ reader ────────────────────────────────────────────────────────────────

def read_xyz(filename):
    """
    Parse trajectory.xyz, keeping only oxygen (OXYGEN_LABEL) positions.

    Returns
    -------
    frames : list of (N_oxygens, 3) float arrays.
             N can vary between frames because the shell selection is
             distance-based and waters enter/leave the 9–15 Å shell.
    """
    frames = []
    with open(filename) as f:
        while True:
            nline = f.readline()
            if not nline:
                break
            try:
                n = int(nline.strip())
            except ValueError:
                continue
            f.readline()          # frame header
            coords = []
            for _ in range(n):
                p = f.readline().split()
                if len(p) >= 4 and p[0] == OXYGEN_LABEL:
                    coords.append([float(p[1]), float(p[2]), float(p[3])])
            if coords:
                frames.append(np.array(coords, dtype=float))
            else:
                frames.append(np.empty((0, 3), dtype=float))
    return frames


# ── MSD ───────────────────────────────────────────────────────────────────────

def msd(frames, max_dt):
    """
    Mean squared displacement averaged over all molecules and all t_0 origins.

    MSD(dt) = < |r(t0+dt) - r(t0)|^2 >

    Returns t_ps (ps), y (Å²), err (SEM in Å²).
    """
    non_empty = [f for f in frames if len(f) > 0]
    min_n = min((len(f) for f in non_empty), default=0)
    if min_n == 0:
        return np.array([]), np.array([]), np.array([])

    n_dt = min(max_dt, len(frames) - 1)
    t_ps = np.arange(1, n_dt + 1) * DT_PS
    y    = np.zeros(n_dt)
    err  = np.zeros(n_dt)

    for idx, dt in enumerate(range(1, n_dt + 1)):
        vals = []
        for t0 in range(len(frames) - dt):
            n = min(len(frames[t0]), len(frames[t0 + dt]), min_n)
            if n == 0:
                continue
            d = frames[t0 + dt][:n] - frames[t0][:n]
            vals.append(np.sum(d * d, axis=1))
        if vals:
            v        = np.concatenate(vals)
            y[idx]   = v.mean()
            err[idx] = v.std() / np.sqrt(len(v))

    return t_ps, y, err


# ── VACF ──────────────────────────────────────────────────────────────────────

def vacf(frames, max_dt):
    """
    Velocity autocorrelation function via central finite differences.

    v(t) ≈ [r(t+1) - r(t-1)] / (2 * DT_PS)
    VACF(dt) = < v(t0+dt) · v(t0) > / < v(t0) · v(t0) >

    Returns t_ps (ps) and normalised C (C(0) = 1).
    """
    non_empty = [f for f in frames if len(f) > 0]
    min_n = min((len(f) for f in non_empty), default=0)
    if min_n == 0 or len(frames) < 3:
        return np.array([]), np.array([])

    vels = []
    for t in range(1, len(frames) - 1):
        n = min(len(frames[t - 1]), len(frames[t + 1]), min_n)
        if n == 0:
            continue
        vels.append((frames[t + 1][:n] - frames[t - 1][:n]) / (2 * DT_PS))

    if not vels:
        return np.array([]), np.array([])

    c0   = np.mean(np.concatenate([np.sum(v * v, axis=1) for v in vels]))
    n_dt = min(max_dt, len(vels) - 1)
    t_ps = np.arange(0, n_dt + 1) * DT_PS
    C    = np.zeros(n_dt + 1)

    for dt in range(0, n_dt + 1):
        vals = []
        for i in range(len(vels) - dt):
            n = min(len(vels[i]), len(vels[i + dt]))
            vals.extend(np.sum(vels[i][:n] * vels[i + dt][:n], axis=1))
        if vals:
            C[dt] = np.mean(vals) / c0

    return t_ps, C


# ── Per-slab MSD ──────────────────────────────────────────────────────────────

def per_slice_msd(frames, max_dt, n_slices):
    """
    MSD computed separately for N equal Z-slabs.

    Atoms assigned to slabs by their Z coordinate at t_0.
    Returns t_ps (ps), mat (n_slices × n_dt, Å², NaN where no data),
    and edges (slab boundaries in Å).
    """
    non_empty = [f for f in frames if len(f) > 0]
    if not non_empty:
        return np.array([]), np.empty((n_slices, 0)), np.array([])

    allz  = np.concatenate([f[:, 2] for f in non_empty])
    edges = np.linspace(allz.min(), allz.max(), n_slices + 1)
    min_n = min(len(x) for x in non_empty)
    n_dt  = min(max_dt, len(frames) - 1)
    mat   = np.full((n_slices, n_dt), np.nan)

    for idx, dt in enumerate(range(1, n_dt + 1)):
        bucket = [[] for _ in range(n_slices)]
        for t0 in range(len(frames) - dt):
            n = min(len(frames[t0]), len(frames[t0 + dt]), min_n)
            if n == 0:
                continue
            z0  = frames[t0][:n, 2]
            sid = np.clip(np.digitize(z0, edges) - 1, 0, n_slices - 1)
            d   = frames[t0 + dt][:n] - frames[t0][:n]
            d2  = np.sum(d * d, axis=1)
            for s in range(n_slices):
                m = sid == s
                if np.any(m):
                    bucket[s].append(d2[m].mean())
        for s in range(n_slices):
            if bucket[s]:
                mat[s, idx] = np.mean(bucket[s])

    return np.arange(1, n_dt + 1) * DT_PS, mat, edges


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _save_msd(t_ps, y, err, outpdf):
    if len(t_ps) == 0:
        print(f'Skipping {os.path.basename(outpdf)}: no data'); return
    fig, ax = plt.subplots()
    ax.fill_between(t_ps, y - err, y + err,
                    color=C_MSD, alpha=0.20, linewidth=0)
    ax.plot(t_ps, y, color=C_MSD,
            label=r'$\langle|\Delta r|^2\rangle$')
    i0 = int(0.80 * len(t_ps))
    if i0 < len(t_ps) - 2:
        slope, _ = np.polyfit(t_ps[i0:], y[i0:], 1)
        D = slope / 6.0   # MSD = 6Dt  →  Å²/ps
        ax.annotate(
            rf'$D \approx {D:.3f}\;\mathrm{{\AA^2/ps}}$',
            xy=(t_ps[i0], y[i0]), xytext=(0.55, 0.25),
            textcoords='axes fraction', fontsize=8,
            arrowprops=dict(arrowstyle='->', lw=0.8))
    ax.set_xlabel(r'$\Delta t$ (ps)')
    ax.set_ylabel(r'MSD ($\mathrm{\AA}^2$)')
    ax.legend(loc='upper left')
    fig.tight_layout(); fig.savefig(outpdf, dpi=150); plt.close(fig)
    print(f'Saved {os.path.relpath(outpdf)}')


def _save_vacf(t_ps, C, outpdf):
    if len(t_ps) == 0:
        print(f'Skipping {os.path.basename(outpdf)}: no data'); return
    fig, ax = plt.subplots()
    ax.axhline(0, color='black', linewidth=0.6, linestyle=':')
    ax.plot(t_ps, C, color=C_VACF, label=r'$C^{\rm vel}(t)$')
    ax.set_xlabel(r'$\Delta t$ (ps)')
    ax.set_ylabel(r'VACF (normalised)')
    ax.set_ylim(bottom=min(-0.15, C.min() - 0.05))
    ax.legend(loc='upper right')
    fig.tight_layout(); fig.savefig(outpdf, dpi=150); plt.close(fig)
    print(f'Saved {os.path.relpath(outpdf)}')


def _save_slice_msd(t_ps, mat, edges, outpdf):
    if len(t_ps) == 0 or mat.size == 0:
        print(f'Skipping {os.path.basename(outpdf)}: no data'); return
    fig, ax = plt.subplots()
    for s in range(mat.shape[0]):
        m = ~np.isnan(mat[s])
        if np.any(m):
            ax.plot(t_ps[m], mat[s, m],
                    label=rf'${edges[s]:.1f} < z < {edges[s+1]:.1f}\;\mathrm{{\AA}}$')
    ax.set_xlabel(r'$\Delta t$ (ps)')
    ax.set_ylabel(r'MSD ($\mathrm{\AA}^2$)')
    ax.legend(loc='upper left', fontsize=7)
    fig.tight_layout(); fig.savefig(outpdf, dpi=150); plt.close(fig)
    print(f'Saved {os.path.relpath(outpdf)}')


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f'Reading: {TRAJ_FILE}')
    frames = read_xyz(TRAJ_FILE)
    print(f'  {len(frames)} frames loaded')

    n_oxy = [len(f) for f in frames if len(f) > 0]
    if n_oxy:
        print(f'  {OXYGEN_LABEL} atoms: min={min(n_oxy)}  '
              f'mean={np.mean(n_oxy):.0f}  max={max(n_oxy)}')
    else:
        print(f'  WARNING: no atoms labelled {OXYGEN_LABEL!r} found — '
              'check OXYGEN_LABEL in the script.')

    # 3a — MSD
    print('Computing MSD...')
    t_msd, y_msd, e_msd = msd(frames, MAX_DT)
    if len(t_msd):
        np.savetxt(os.path.join(FIG_DIR, '3a.msd.csv'),
                   np.column_stack([t_msd, y_msd, e_msd]),
                   header='t_ps MSD_A2 SEM', comments='# ')
    _save_msd(t_msd, y_msd, e_msd, os.path.join(FIG_DIR, '3a.msd.pdf'))

    # 3b — VACF
    print('Computing VACF...')
    t_vacf, C_vacf = vacf(frames, MAX_DT)
    if len(t_vacf):
        np.savetxt(os.path.join(FIG_DIR, '3b.vacf.csv'),
                   np.column_stack([t_vacf, C_vacf]),
                   header='t_ps VACF', comments='# ')
    _save_vacf(t_vacf, C_vacf, os.path.join(FIG_DIR, '3b.vacf.pdf'))

    # 3c — Per-slab MSD
    print(f'Computing per-slab MSD ({N_SLICES} slices)...')
    t_slc, mat_slc, edges_slc = per_slice_msd(frames, MAX_DT, N_SLICES)
    if len(t_slc):
        np.savetxt(os.path.join(FIG_DIR, '3c.msd_per_slice.csv'),
                   np.column_stack([t_slc] + [mat_slc[s] for s in range(N_SLICES)]),
                   header='t_ps ' + ' '.join(
                       [f'slab{s}_{edges_slc[s]:.1f}-{edges_slc[s+1]:.1f}A'
                        for s in range(N_SLICES)]),
                   comments='# ')
    _save_slice_msd(t_slc, mat_slc, edges_slc,
                    os.path.join(FIG_DIR, '3c.msd_per_slice.pdf'))

    print('Done.')