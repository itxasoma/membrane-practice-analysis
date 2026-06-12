#!/usr/bin/env python3
"""
corr-plots.py — Plot shell-conditioned rotational correlation functions.

Reads:
    figures/2.corr_<tag>.csv

Writes:
    figures/2.corr.pdf
    figures/2.corr_pairs.pdf
"""

import os
import numpy as np
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR  = os.path.join(BASE_DIR, '../figures')

os.makedirs(FIG_DIR, exist_ok=True)

plt.style.use('lib/science.mplstyle')

MAX_LAG_PS = 10.0

SHELLS = [
    (r'$0$-$3\,\AA$',   '0_3A',   '#F0A500'),
    (r'$3$-$5\,\AA$',   '3_5A',   '#E87D72'),
    (r'$5$-$10\,\AA$',  '5_10A',  '#845B97'),
    (r'$10$-$15\,\AA$', '10_15A', '#56A0D3'),
]


def load_csv(path):
    data = np.loadtxt(path, comments='#')
    if data.ndim == 1:
        data = data[None, :]
    t_ps = data[:, 0]
    C_rot = data[:, 1]
    pair_count = data[:, 2]
    t0_count = data[:, 3]
    return t_ps, C_rot, pair_count, t0_count


if __name__ == '__main__':
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    for label, tag, color in SHELLS:
        path = os.path.join(FIG_DIR, f'2.corr_{tag}.csv')
        if not os.path.exists(path):
            print(f'[skip] {os.path.basename(path)} not found')
            continue

        t_ps, C_rot, pair_count, t0_count = load_csv(path)

        plt.rcParams['lines.markersize'] = 0.1
        ax1.plot(t_ps, C_rot, color=color, lw=2, label=label)
        ax2.plot(t_ps, pair_count, color=color, lw=2, label=label)

        print(f'Loaded {os.path.basename(path)}')

    ax1.axhline(0, color='black', lw=0.7, ls=':')
    ax1.set_xlabel(r'$t$ (ps)')
    ax1.set_ylabel(r'$C^{\rm rot}(t)$')
    ax1.set_xlim(0, MAX_LAG_PS)
    ax1.set_ylim(0, 1.0)
    ax1.legend(loc='best')
    fig1.tight_layout()

    out1 = os.path.join(FIG_DIR, '2.corr.pdf')
    fig1.savefig(out1, dpi=150)
    plt.close(fig1)
    print(f'Saved {os.path.relpath(out1)}')

    ax2.set_xlabel(r'$t$ (ps)')
    ax2.set_ylabel('Contributing molecule pairs')
    ax2.set_xlim(0, MAX_LAG_PS)
    ax2.legend(loc='best')
    fig2.tight_layout()

    out2 = os.path.join(FIG_DIR, '2.corr_pairs.pdf')
    fig2.savefig(out2, dpi=150)
    plt.close(fig2)
    print(f'Saved {os.path.relpath(out2)}')
