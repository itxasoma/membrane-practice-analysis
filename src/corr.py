#!/usr/bin/env python3
"""
corr.py — Dipolar rotational autocorrelation function C_rot(t) for water
          molecules in the hydration shell of a DMPC membrane.

Definition (Informe_practica, eq. 1):

    C_rot(t) = < mu_hat(t0+t) · mu_hat(t0) >
               --------------------------------
                 < mu_hat(t0) · mu_hat(t0) >

where mu_hat(t) is the unit dipole vector of a water molecule at time t,
defined as the direction bisecting the H–O–H angle:

    mu(t) = r_O(t) - 0.5 * (r_H1(t) + r_H2(t))
    mu_hat(t) = mu(t) / |mu(t)|

The double average is over:
  1. All water molecules in the selected shell (9–15 Å from any lipid atom)
  2. All t_0 starting times (valid because the system is at equilibrium)

Input:  trajectory.xyz  (produced by trajectory_to_xyz.py)
Output: figures/2.corr.pdf
        figures/2.corr.csv   (t_ps, C_rot columns)

Color conventions (matching science.mplstyle cycle):
  C_rot : violet  #845B97  (same as T — a rotational/thermal quantity)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os


BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
FIG_DIR   = os.path.join(BASE_DIR, '../figures')
TRAJ_FILE = os.path.join(BASE_DIR, '../1_Analysis/trajectory.xyz')

plt.style.use('lib/science.mplstyle')

os.makedirs(FIG_DIR, exist_ok=True)

# Timestep between consecutive frames in the trajectory (ps).
# DCDfreq=10 and timestep=2fs → one frame every 10×2 fs = 20 fs = 0.02 ps.
DT_PS = 0.02

C_CORR = '#845B97'   # violet


def darken(hex_color, factor=0.55):
    r, g, b = mcolors.to_rgb(hex_color)
    return (r * factor, g * factor, b * factor)


# ── XYZ reader ────────────────────────────────────────────────────────────────

def read_xyz(filename):
    """
    Parse trajectory.xyz produced by trajectory_to_xyz.py.

    Returns
    -------
    frames : list of dicts  {atom_name: list_of_positions}
             Each frame is a dict mapping atom name (e.g. 'OH2', 'H1', 'H2')
             to an (N, 3) array of positions in Å.
             Because the number of selected waters can vary between frames,
             we store per-residue dipole vectors separately (see below).
    """
    frames = []
    with open(filename) as f:
        while True:
            line = f.readline()
            if not line:
                break
            n_atoms = int(line.strip())
            _ = f.readline()  # frame header

            atoms = []
            for _ in range(n_atoms):
                parts = f.readline().split()
                name = parts[0]
                xyz  = np.array([float(parts[1]),
                                  float(parts[2]),
                                  float(parts[3])])
                atoms.append((name, xyz))
            frames.append(atoms)
    return frames


def build_dipoles(frames):
    """
    For each frame, group atoms into TIP3P water residues (O + 2 H) and
    compute the unit dipole vector for each water.

    TIP3P atom names in CHARMM: OH2 (oxygen), H1, H2.
    The dipole direction bisects H–O–H:

        mu = r_O - 0.5*(r_H1 + r_H2)

    Returns
    -------
    dipoles : list of (N_waters, 3) arrays, one per frame.
              N_waters can differ between frames because the shell
              selection is distance-based.
    """
    dipoles = []
    for atoms in frames:
        # Collect into residues: consecutive triplets OH2, H1, H2
        frame_dipoles = []
        i = 0
        while i < len(atoms) - 2:
            name0, r0 = atoms[i]
            name1, r1 = atoms[i+1]
            name2, r2 = atoms[i+2]

            # Accept any ordering that contains one O and two H
            group = {name0: r0, name1: r1, name2: r2}
            o_keys = [k for k in group if k.upper() in ('OH2', 'O')]
            h_keys = [k for k in group if k.upper() in ('H1', 'H2', 'H')]

            if len(o_keys) == 1 and len(h_keys) == 2:
                r_O  = group[o_keys[0]]
                r_H1 = group[h_keys[0]]
                r_H2 = group[h_keys[1]]
                mu   = r_O - 0.5 * (r_H1 + r_H2)
                norm = np.linalg.norm(mu)
                if norm > 1e-10:
                    frame_dipoles.append(mu / norm)
                i += 3
            else:
                i += 1  # skip misaligned atom, try next

        if frame_dipoles:
            dipoles.append(np.array(frame_dipoles))   # shape (N_w, 3)
        else:
            dipoles.append(np.zeros((0, 3)))

    return dipoles


# ── Correlation function ──────────────────────────────────────────────────────

def compute_crot(dipoles, max_lag=None):
    """
    Compute C_rot(t) averaged over all molecules and all t_0.

    For each lag dt (in frames):
        C_rot(dt) = mean over all t0 of:
                    mean over molecules present in BOTH t0 and t0+dt of:
                    mu_hat(t0+dt) · mu_hat(t0)

    Only molecules that appear in both frames contribute (the number of
    waters in the shell can fluctuate slightly between frames).

    Parameters
    ----------
    dipoles  : list of (N_i, 3) arrays
    max_lag  : int, maximum lag in frames (default: half the trajectory)

    Returns
    -------
    t_ps   : (max_lag,) array   — lag times in ps
    C_rot  : (max_lag,) array   — correlation values, C_rot(0) = 1
    """
    n_frames = len(dipoles)
    if max_lag is None:
        max_lag = n_frames // 2

    C_rot = np.zeros(max_lag)
    counts = np.zeros(max_lag, dtype=int)

    for t0 in range(n_frames - 1):
        d0 = dipoles[t0]           # (N0, 3)
        if d0.shape[0] == 0:
            continue
        n0 = d0.shape[0]

        for dt in range(1, min(max_lag, n_frames - t0)):
            d1 = dipoles[t0 + dt]  # (N1, 3)
            if d1.shape[0] == 0:
                continue

            # Use the minimum number of molecules present in both frames
            n_common = min(n0, d1.shape[0])
            dot = np.sum(d0[:n_common] * d1[:n_common], axis=1)  # (n_common,)
            C_rot[dt]  += dot.mean()
            counts[dt] += 1

    # Normalise by number of t0 origins; set C_rot(0) = 1 by definition
    mask = counts > 0
    C_rot[mask] /= counts[mask]
    C_rot[0] = 1.0

    t_ps = np.arange(max_lag) * DT_PS
    return t_ps, C_rot


# ── Plot & save ───────────────────────────────────────────────────────────────

def save_crot(t_ps, C_rot):
    fig, ax = plt.subplots()
    ax.plot(t_ps, C_rot, color=C_CORR,
            label=r'$C^{\rm rot}_{\rm sim}(t)$')
    ax.axhline(0, color='black', linewidth=0.6, linestyle=':')
    ax.set_xlabel(r'$t$ (ps)')
    ax.set_ylabel(r'$C^{\rm rot}_{\rm sim}(t)$')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='best')
    fig.tight_layout()
    out = os.path.join(FIG_DIR, '2.corr.pdf')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved {os.path.relpath(out)}')


def save_csv(t_ps, C_rot):
    out = os.path.join(FIG_DIR, '2.corr.csv')
    np.savetxt(out,
               np.column_stack([t_ps, C_rot]),
               header='t_ps C_rot',
               comments='# ')
    print(f'Saved {os.path.relpath(out)}')


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f'Reading: {TRAJ_FILE}')
    frames  = read_xyz(TRAJ_FILE)
    print(f'  {len(frames)} frames loaded')

    print('Building dipole vectors...')
    dipoles = build_dipoles(frames)
    n_waters = [d.shape[0] for d in dipoles if d.shape[0] > 0]
    print(f'  Waters per frame: min={min(n_waters)}  '
          f'mean={np.mean(n_waters):.0f}  max={max(n_waters)}')

    print('Computing C_rot(t)...')
    t_ps, C_rot = compute_crot(dipoles)
    print(f'  Max lag: {t_ps[-1]:.2f} ps  ({len(t_ps)} points)')
    print(f'  C_rot at t=0:       {C_rot[0]:.4f}  (should be 1.0)')
    print(f'  C_rot at t_max/2:   {C_rot[len(C_rot)//2]:.4f}')
    print(f'  C_rot at t_max:     {C_rot[-1]:.4f}')

    save_crot(t_ps, C_rot)
    save_csv(t_ps, C_rot)