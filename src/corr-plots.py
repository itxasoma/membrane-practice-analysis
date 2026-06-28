#!/usr/bin/env python3
"""
corr-plots.py — Plot shell-conditioned rotational correlation functions.

Reads:
    figures/2.corr_<shell>_<TEMP_TAG>.csv   (produced by corr.py)

Writes:
    figures/2.corr.pdf   — C_rot(t) averaged over all shells, one curve per
                           temperature, coloured with the inferno colourmap.
"""

import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR  = os.path.join(BASE_DIR, '../figures')

os.makedirs(FIG_DIR, exist_ok=True)
plt.style.use('lib/science.mplstyle')

MAX_LAG_PS = 10.0

SHELL_ORDER = ['0_3A', '3_5A', '5_10A', '10_15A']


def load_csv(path):
    data = np.loadtxt(path, comments='#')
    if data.ndim == 1:
        data = data[None, :]
    return data[:, 0], data[:, 1]   # t_ps, C_rot


def discover_temps():
    """Return sorted list of temperature tags found in figures/."""
    pattern = os.path.join(FIG_DIR, '2.corr_0_3A_*.csv')
    tags = []
    for p in glob.glob(pattern):
        m = re.search(r'2\.corr_0_3A_(.+)\.csv$', os.path.basename(p))
        if m:
            tags.append(m.group(1))
    return sorted(tags, key=lambda s: float(s))


def average_shells(temp):
    """
    Load all available shells for a given temperature tag and return
    their point-wise average on a common t_ps grid.

    Returns (t_ps, C_avg) or (None, None) if no shells are found.
    """
    curves = []
    t_ref  = None

    for shell_tag in SHELL_ORDER:
        path = os.path.join(FIG_DIR, f'2.corr_{shell_tag}_{temp}.csv')
        if not os.path.exists(path):
            print(f'  [skip] {os.path.basename(path)}')
            continue

        t_ps, C_rot = load_csv(path)

        # Restrict to MAX_LAG_PS
        mask = t_ps <= MAX_LAG_PS
        t_ps  = t_ps[mask]
        C_rot = C_rot[mask]

        if t_ref is None:
            t_ref = t_ps
        else:
            # Interpolate onto the reference grid if lengths differ
            if len(t_ps) != len(t_ref):
                C_rot = np.interp(t_ref, t_ps, C_rot)

        curves.append(C_rot)
        print(f'  Loaded {os.path.basename(path)}')

    if not curves:
        return None, None

    C_avg = np.mean(curves, axis=0)
    return t_ref, C_avg


if __name__ == '__main__':
    temps = discover_temps()
    if not temps:
        print(f'No CSV files found in {FIG_DIR}.')
        print('Run corr.py for each temperature first.')
        raise SystemExit(1)

    print(f'Temperatures found: {temps}')

    # Inferno colours: sample from [0.15, 0.85] to avoid near-black/near-white ends.
    n_temps = len(temps)
    sample_points = np.linspace(0.15, 0.85, n_temps)
    inferno = cm.get_cmap('inferno')
    temp_colors = {t: inferno(s) for t, s in zip(temps, sample_points)}

    fig, ax = plt.subplots()

    for temp in temps:
        print(f'\nT = {temp} K')
        t_ps, C_avg = average_shells(temp)
        if t_ps is None:
            print(f'  No data for T = {temp} K, skipping.')
            continue

        color = temp_colors[temp]
        ax.plot(t_ps, C_avg, color=color, lw=1.6, label=f'$T = {temp}$ K')

    ax.set_xlabel(r'$t$ (ps)')
    ax.set_ylabel(r'$C^{\rm rot}(t)$')
    ax.set_xlim(0, MAX_LAG_PS)
    ax.set_ylim(0, 1.0)
    ax.legend(title='Temperature', loc='upper right', framealpha=0.9)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, '2.corr.pdf')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'\nSaved {os.path.relpath(out)}')