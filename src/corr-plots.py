#!/usr/bin/env python3
"""
corr-plots.py — Read the CSV files produced by corr.py and generate all figures.

Input:  figures/2.corr_<tag>.csv           — per-shell full-average C_rot
        figures/2.corr_lastframe_<tag>.csv — per-shell last-frame C_rot
Output: figures/2.corr.pdf                 — all shells overlaid (full average)
        figures/2.corr_lastframe_<tag>.pdf — per-shell: full avg vs last-frame

Run corr.py first to produce the CSVs.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR  = os.path.join(BASE_DIR, '../figures')

plt.style.use('lib/science.mplstyle')
os.makedirs(FIG_DIR, exist_ok=True)

MAX_LAG_PS = 10.0   # x-axis limit for all plots

# Must match the tags produced by corr.py's _shell_tag()
SHELLS = [
    (r'$0$-$3\,\AA$',   '0_3A',   '#E87D72'),   # red
    (r'$3$-$5\,\AA$',   '3_5A',   '#F0A500'),   # orange
    (r'$5$-$10\,\AA$',  '5_10A',  '#845B97'),   # violet
    (r'$10$-$15\,\AA$', '10_15A', '#56A0D3'),   # blue
]


def darken(hex_color, factor=0.55):
    r, g, b = mcolors.to_rgb(hex_color)
    return (r * factor, g * factor, b * factor)


def load_csv(path):
    data = np.loadtxt(path, comments='#')
    return data[:, 0], data[:, 1]


def _csv_path(tag, suffix=''):
    return os.path.join(FIG_DIR, f'2.corr{suffix}_{tag}.csv')


# ── Plot 1: all shells overlaid (full circular average) ───────────────────────

def plot_all_shells(results):
    fig, ax = plt.subplots()
    for label, color, t_ps, C_rot in results:
        ax.plot(t_ps, C_rot, color=color, label=label)
    ax.axhline(0, color='black', linewidth=0.6, linestyle=':')
    ax.set_xlabel(r'$t$ (ps)')
    ax.set_ylabel(r'$C^{\rm rot}_{\rm sim}(t)$')
    ax.set_xlim(0, MAX_LAG_PS)
    ax.set_ylim(0, 0.1)
    ax.legend(loc='best')
    fig.tight_layout()
    out = os.path.join(FIG_DIR, '2.corr.pdf')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved {os.path.relpath(out)}')


# ── Plot 2: per-shell full avg vs last-frame ──────────────────────────────────

def plot_comparison(label, tag, color):
    path_avg  = _csv_path(tag, suffix='')
    path_last = _csv_path(tag, suffix='_lastframe')

    missing = [p for p in (path_avg, path_last) if not os.path.exists(p)]
    if missing:
        for p in missing:
            print(f'  [skip] {os.path.basename(p)} not found')
        return

    t_avg,  C_avg  = load_csv(path_avg)
    t_last, C_last = load_csv(path_last)

    fig, ax = plt.subplots()
    ax.plot(t_avg,  C_avg,  color=color,
            label=r'$C^{\rm rot}_{\rm avg}(t)$ (circular $t_0$)')
    ax.plot(t_last, C_last, color=darken(color), linestyle='--',
            label=r'$C^{\rm rot}_{\rm last}(t)$ (last frame $t_0$)')
    ax.axhline(0, color='black', linewidth=0.6, linestyle=':')
    ax.set_xlabel(r'$t$ (ps)')
    ax.set_ylabel(r'$C^{\rm rot}_{\rm sim}(t)$')
    ax.set_xlim(0, MAX_LAG_PS)
    ax.set_ylim(0, 1.0)
    ax.set_title(label)
    ax.legend(loc='best')
    fig.tight_layout()
    out = os.path.join(FIG_DIR, f'2.corr_lastframe_{tag}.pdf')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved {os.path.relpath(out)}')


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    results = []

    for label, tag, color in SHELLS:
        path_avg = _csv_path(tag, suffix='')
        if not os.path.exists(path_avg):
            print(f'[skip] {os.path.basename(path_avg)} not found — run corr.py first')
            continue
        t_ps, C_rot = load_csv(path_avg)
        results.append((label, color, t_ps, C_rot))
        print(f'Loaded {os.path.basename(path_avg)}')

    if results:
        plot_all_shells(results)

    for label, tag, color in SHELLS:
        plot_comparison(label, tag, color)

    print('\nDone.')
