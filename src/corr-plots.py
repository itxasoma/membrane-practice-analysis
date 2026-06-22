#!/usr/bin/env python3
"""
corr-plots.py — Plot shell-conditioned rotational correlation functions.

Reads:
    figures/2.corr_<shell>_<TEMP_TAG>.csv

Writes:
    figures/2.corr_<TEMP_TAG>.pdf        — C_rot(t) for all shells
    figures/2.corr_pairs_<TEMP_TAG>.pdf  — contributing molecule pairs per lag
"""

import os
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR  = os.path.join(BASE_DIR, '../figures')

os.makedirs(FIG_DIR, exist_ok=True)
plt.style.use('lib/science.mplstyle')

MAX_LAG_PS = 10.0

# ─────────────────────────────────────────────────────────────────────────────
TEMP_TAG = "307.5"
# ─────────────────────────────────────────────────────────────────────────────

SHELLS = [
    (r'$0$-$3\,\AA$',   '0_3A',   '#F0A500'),   # orange
    (r'$3$-$5\,\AA$',   '3_5A',   '#E87D72'),   # red
    (r'$5$-$10\,\AA$',  '5_10A',  '#56A0D3'),   # blue
    (r'$10$-$15\,\AA$', '10_15A', '#845B97'),   # violet
]


def load_csv(path):
    data = np.loadtxt(path, comments='#')
    if data.ndim == 1:
        data = data[None, :]
    t_ps       = data[:, 0]
    C_rot      = data[:, 1]
    pair_count = data[:, 2]
    t0_count   = data[:, 3]
    return t_ps, C_rot, pair_count, t0_count


if __name__ == '__main__':
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    any_loaded = False

    for label, tag, color in SHELLS:
        path = os.path.join(FIG_DIR, f'2.corr_{tag}_{TEMP_TAG}.csv')
        if not os.path.exists(path):
            print(f'[skip] {os.path.basename(path)} not found')
            continue

        t_ps, C_rot, pair_count, t0_count = load_csv(path)
        any_loaded = True

        plt.rcParams['lines.markersize'] = 0.3
        ax1.plot(t_ps, C_rot, color=color, lw=2, label=label)
        ax2.plot(t_ps, pair_count, color=color, lw=2, label=label)

        print(f'Loaded {os.path.basename(path)}  '
              f'(C(0)={C_rot[0]:.3f}, C(end)={C_rot[-1]:.3f})')

    if not any_loaded:
        print('\nNo CSV files found. Check that:')
        print(f'  1. corr.py has been run with TEMP_TAG = "{TEMP_TAG}"')
        print(f'  2. The CSVs are in {FIG_DIR}')
        print(f'  3. This script uses the same TEMP_TAG = "{TEMP_TAG}"')
    else:
        ax1.set_xlabel(r'$t$ (ps)')
        ax1.set_ylabel(r'$C^{\rm rot}(t)$')
        ax1.set_xlim(0, MAX_LAG_PS)
        ax1.set_ylim(0, 1.0)
        ax1.set_title(f'Rotational correlation for T = {TEMP_TAG} K')
        ax1.legend(loc='upper right')
        fig1.tight_layout()

        out1 = os.path.join(FIG_DIR, f'2.corr_{TEMP_TAG}.pdf')
        fig1.savefig(out1, dpi=150)
        plt.close(fig1)
        print(f'\nSaved {os.path.relpath(out1)}')

        ax2.set_xlabel(r'$t$ (ps)')
        ax2.set_ylabel('Contributing molecule pairs')
        ax2.set_xlim(0, MAX_LAG_PS)
        ax2.set_title(f'Shell population — T = {TEMP_TAG} K')
        ax2.legend(loc='upper right')
        fig2.tight_layout()

        out2 = os.path.join(FIG_DIR, f'2.corr_pairs_{TEMP_TAG}.pdf')
        fig2.savefig(out2, dpi=150)
        plt.close(fig2)
        print(f'Saved {os.path.relpath(out2)}')