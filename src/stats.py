#!/usr/bin/env python3
"""
stats.py — Fit rotational correlation times tau_rot from C_rot(t) CSVs.

For each (temperature, shell) pair, fits a single-exponential decay

    C_rot(t) = A * exp(-t / tau)

over the range [T_MIN_FIT, T_MAX_FIT] (default 0.5 – 10 ps) using
scipy.optimize.curve_fit.

Reads:
    figures/2.corr_<shell>_<TEMP_TAG>.csv   (produced by corr.py)

Writes:
    figures/3.tau.csv    — table of (T, shell, tau, tau_err, A, A_err, R2)
    figures/3.tau.pdf    — tau_rot vs temperature for each shell
"""

import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR  = os.path.join(BASE_DIR, '../figures')

os.makedirs(FIG_DIR, exist_ok=True)
plt.style.use('lib/science.mplstyle')

# Fit window (ps)
T_MIN_FIT = 0.5
T_MAX_FIT = 10.0

SHELL_ORDER = ['0_3A', '3_5A', '5_10A', '10_15A']
SHELL_LABELS = {
    '0_3A':   r'$0$–$3\,\AA$',
    '3_5A':   r'$3$–$5\,\AA$',
    '5_10A':  r'$5$–$10\,\AA$',
    '10_15A': r'$10$–$15\,\AA$',
}
SHELL_LINESTYLES = ['-', '--', '-.', ':']


# ── helpers ───────────────────────────────────────────────────────────────────

def load_csv(path):
    data = np.loadtxt(path, comments='#')
    if data.ndim == 1:
        data = data[None, :]
    return data[:, 0], data[:, 1]   # t_ps, C_rot


def discover_temps():
    pattern = os.path.join(FIG_DIR, '2.corr_0_3A_*.csv')
    tags = []
    for p in glob.glob(pattern):
        m = re.search(r'2\.corr_0_3A_(.+)\.csv$', os.path.basename(p))
        if m:
            tags.append(m.group(1))
    return sorted(tags, key=lambda s: float(s))


def exp_decay(t, A, tau):
    return A * np.exp(-t / tau)


def fit_tau(t_ps, C_rot, t_min=T_MIN_FIT, t_max=T_MAX_FIT):
    """
    Fit A*exp(-t/tau) over [t_min, t_max].
    Returns (tau, tau_err, A, A_err, R2) or (nan,)*5 on failure.
    """
    mask = (t_ps >= t_min) & (t_ps <= t_max) & np.isfinite(C_rot)
    t = t_ps[mask]
    C = C_rot[mask]
    if len(t) < 5:
        return (np.nan,) * 5

    try:
        p0     = [C[0], 2.0]
        bounds = ([0, 0.01], [2.0, 200.0])
        popt, pcov = curve_fit(exp_decay, t, C, p0=p0, bounds=bounds, maxfev=10000)
        perr       = np.sqrt(np.diag(pcov))
        A,   tau   = popt
        A_e, tau_e = perr

        C_pred = exp_decay(t, A, tau)
        ss_res = np.sum((C - C_pred) ** 2)
        ss_tot = np.sum((C - C.mean()) ** 2)
        R2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        return tau, tau_e, A, A_e, R2
    except (RuntimeError, ValueError):
        return (np.nan,) * 5


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    temps = discover_temps()
    if not temps:
        print(f'No CSV files found in {FIG_DIR}.')
        print('Run corr.py for each temperature first.')
        raise SystemExit(1)

    print(f'Temperatures: {temps}')
    print(f'Fit window  : [{T_MIN_FIT}, {T_MAX_FIT}] ps\n')

    rows = []   # (T_float, shell_tag, tau, tau_err, A, A_err, R2)

    for temp in temps:
        T = float(temp)
        for shell_tag in SHELL_ORDER:
            path = os.path.join(FIG_DIR, f'2.corr_{shell_tag}_{temp}.csv')
            if not os.path.exists(path):
                print(f'  [skip] {os.path.basename(path)}')
                continue

            t_ps, C_rot = load_csv(path)
            tau, tau_e, A, A_e, R2 = fit_tau(t_ps, C_rot)

            rows.append((T, shell_tag, tau, tau_e, A, A_e, R2))
            status = (
                f'tau={tau:.3f}±{tau_e:.3f} ps  A={A:.3f}  R²={R2:.4f}'
                if np.isfinite(tau) else 'FIT FAILED'
            )
            print(f'  T={temp} K  {SHELL_LABELS[shell_tag]:20s}  {status}')

    # ── save CSV ──────────────────────────────────────────────────────────────
    csv_out = os.path.join(FIG_DIR, '3.tau.csv')
    with open(csv_out, 'w') as fh:
        fh.write('# T_K shell tau_ps tau_err_ps A A_err R2\n')
        for T, shell, tau, tau_e, A, A_e, R2 in rows:
            fh.write(f'{T:.1f} {shell} {tau:.6f} {tau_e:.6f} '
                     f'{A:.6f} {A_e:.6f} {R2:.6f}\n')
    print(f'\nSaved {os.path.relpath(csv_out)}')

    # ── plot tau vs T ─────────────────────────────────────────────────────────
    shell_data = {s: {'T': [], 'tau': [], 'err': []} for s in SHELL_ORDER}
    for T, shell, tau, tau_e, A, A_e, R2 in rows:
        if np.isfinite(tau) and shell in shell_data:
            shell_data[shell]['T'].append(T)
            shell_data[shell]['tau'].append(tau)
            shell_data[shell]['err'].append(tau_e)

    shell_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, ax = plt.subplots()
    for i, (shell_tag, ls) in enumerate(zip(SHELL_ORDER, SHELL_LINESTYLES)):
        d = shell_data[shell_tag]
        if not d['T']:
            continue
        color = shell_colors[i % len(shell_colors)]
        ax.errorbar(
            d['T'], d['tau'], yerr=d['err'],
            fmt='o', color=color, ls=ls, lw=1.4, capsize=3,
            label=SHELL_LABELS[shell_tag]
        )

    ax.set_xlabel(r'$T$ (K)')
    ax.set_ylabel(r'$\tau_{\rm rot}$ (ps)')
    ax.legend(title='Shell', framealpha=0.9)
    fig.tight_layout()

    pdf_out = os.path.join(FIG_DIR, '3.tau.pdf')
    fig.savefig(pdf_out, dpi=150)
    plt.close(fig)
    print(f'Saved {os.path.relpath(pdf_out)}')