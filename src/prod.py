#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR  = os.path.join(BASE_DIR, '../figures')
#LOG_FILE = os.path.join(BASE_DIR, '../0_Simulation/Produccio_NVT/Vctt/nvt.log')
#LOG_FILE = os.path.join(BASE_DIR, '../TSCAN/304.5-itziar/Produccio_NVT/Vctt/168096/nvt.log')
#LOG_FILE = os.path.join(BASE_DIR, '../TSCAN/301.5-Andrea/Produccio_NVT/Vctt/168097/nvt.log')
LOG_FILE = os.path.join(BASE_DIR, '../TSCAN/293.5-sadhbh/Produccio_NVT/Vctt/168098/nvt.log')

plt.style.use('lib/science.mplstyle')
os.makedirs(FIG_DIR, exist_ok=True)

T_TARGET = 293.5   # K
P_TARGET = 1.01325 # bar
BLOCK_COUNTS = [5, 8, 10, 12, 15, 20, 25, 30, 40, 50]

C_K    = '#0C5DA5'
C_ETOT = '#00B945'
C_UPOT = '#FF9500'
C_T    = '#845B97'
C_V    = '#474747'
C_P    = '#FF2C00'

COLUMNS = [
    'TS', 'BOND', 'ANGLE', 'DIHED', 'IMPRP',
    'ELECT', 'VDW', 'BOUNDARY', 'MISC', 'KINETIC',
    'TOTAL', 'TEMP', 'POTENTIAL', 'TOTAL3', 'TEMPAVG',
    'PRESSURE', 'GPRESSURE', 'VOLUME', 'PRESSAVG', 'GPRESSAVG'
]

def darken(hex_color, factor=0.55):
    r, g, b = mcolors.to_rgb(hex_color)
    return (r * factor, g * factor, b * factor)

def load_log(filename):
    data = {c: [] for c in COLUMNS}
    with open(filename) as f:
        for line in f:
            if line.startswith('ENERGY:'):
                parts = line.split()
                if len(parts) >= len(COLUMNS) + 1:
                    for i, c in enumerate(COLUMNS):
                        data[c].append(float(parts[i + 1]))
    for c in COLUMNS:
        data[c] = np.array(data[c], dtype=float)
    data['TIME_PS'] = data['TS'] * 2e-3
    return data

def running_stats(x):
    n   = np.arange(1, len(x) + 1, dtype=float)
    mu  = np.cumsum(x) / n
    var = np.maximum(np.cumsum(x**2) / n - mu**2, 0.0)
    return mu, np.sqrt(var)

def zscore_ylim(x, n_sigma=3.5):
    half = len(x) // 2
    mu, sigma = np.mean(x[half:]), np.std(x[half:])
    pad = n_sigma * sigma if sigma > 0 else 1.0
    return mu - pad, mu + pad

def block_sem(x, n_blocks):
    block_len = len(x) // n_blocks
    if block_len < 1 or n_blocks < 2:
        return np.nan
    xb = x[:block_len * n_blocks].reshape(n_blocks, block_len).mean(axis=1)
    return float(np.std(xb, ddof=1) / np.sqrt(n_blocks))

def best_block_sem(x):
    sems = [block_sem(x, n) for n in BLOCK_COUNTS]
    sems = [s for s in sems if np.isfinite(s)]
    return float(np.median(sems[-3:])) if sems else float(np.std(x, ddof=1) / np.sqrt(len(x)))

def zscore_plot(t, x, color, target, ylabel, outpath, target_label):
    mu, sigma = running_stats(x)
    fig, ax = plt.subplots()
    ax.fill_between(t, mu - 2*sigma, mu + 2*sigma, color=color, alpha=0.15, label=r'$\pm2\sigma$')
    ax.fill_between(t, mu - sigma,   mu + sigma,   color=color, alpha=0.40, label=r'$\pm1\sigma$')
    ax.plot(t, mu, color=darken(color), lw=1.4, ls='--', label=r'running mean')
    ax.axhline(target, color='black', lw=0.8, ls=':', label=target_label)
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel(ylabel)
    ax.set_ylim(zscore_ylim(x))
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f'Saved {os.path.relpath(outpath)}')

def raw_plot(t, x, color, target, ylabel, outpath, target_label):
    lo, hi = np.min(x), np.max(x)
    pad = 0.05 * (hi - lo) if hi != lo else 1.0
    fig, ax = plt.subplots()
    ax.plot(t, x, marker='.', color=color, label=ylabel.split('(')[0].strip())
    ax.axhline(target, color='black', lw=0.8, ls=':', label=target_label)
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel(ylabel)
    ax.set_ylim(lo - pad, hi + pad)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f'Saved {os.path.relpath(outpath)}')

def save_energies(t, kinetic, total, potential, outpath):
    fig, ax = plt.subplots()
    ax.plot(t, kinetic,   marker='.', color=C_K,    label=r'$K$')
    ax.plot(t, total,     marker='.', color=C_ETOT, label=r'$E_{\rm tot}$')
    ax.plot(t, potential, marker='.', color=C_UPOT, label=r'$U_{\rm pot}$')
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('Energy (kcal/mol)')
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f'Saved {os.path.relpath(outpath)}')

if __name__ == '__main__':
    print(f'Reading: {LOG_FILE}')
    data = load_log(LOG_FILE)
    t = data['TIME_PS']
    print(f'{len(t)} frames  (TS {int(data["TS"][0])}–{int(data["TS"][-1])})')

    v0 = float(data['VOLUME'][0])

    # ── Plots ────────────────────────────────────────────────────────────────

    save_energies(t, data['KINETIC'], data['TOTAL'], data['POTENTIAL'],
                  os.path.join(FIG_DIR, '1.energies_293.pdf'))

    raw_plot(t, data['TEMP'], C_T, T_TARGET,
             r'Temperature (K)',
             os.path.join(FIG_DIR, '1.temperature_293.pdf'),
             rf'$T_{{\rm target}} = {T_TARGET}$ K')

    zscore_plot(t, data['TEMP'], C_T, T_TARGET,
                r'Temperature (K)',
                os.path.join(FIG_DIR, '1.temperature-zscore_293.pdf'),
                rf'$T_{{\rm target}} = {T_TARGET}$ K')

    raw_plot(t, data['VOLUME'], C_V, v0,
             r'Volume ($\mathrm{\AA}^3$)',
             os.path.join(FIG_DIR, '1.volume_293.pdf'),
             rf'$V_0 = {v0:.1f}\ \mathrm{{\AA}}^3$')

    zscore_plot(t, data['PRESSURE'], C_P, P_TARGET,
                r'Pressure (bar)',
                os.path.join(FIG_DIR, '1.pressure-zscore_293.pdf'),
                rf'$P_{{\rm target}} = {P_TARGET}$ bar')

    raw_plot(t, data['PRESSURE'], C_P, P_TARGET,
             r'Pressure (bar)',
             os.path.join(FIG_DIR, '1.pressure_293.pdf'),
             rf'$P_{{\rm target}} = {P_TARGET}$ bar')

    # ── Block-averaged uncertainties to txt ───────────────────────────────────

    half = len(t) // 2
    out_txt = os.path.join(FIG_DIR, '1.block_stats_293.txt')
    with open(out_txt, 'w') as f:
        f.write(f'# Block-averaged stats — second half of run\n')
        f.write(f'# log: {LOG_FILE}\n')
        f.write(f'# {"variable":<10}  {"mean":>12}  {"±sem":>12}  {"unit":<6}  target\n')
        for var, unit, target in [
            ('TEMP',     'K',    T_TARGET),
            ('PRESSURE', 'bar',  P_TARGET),
            ('VOLUME',   'Ang3', v0),
        ]:
            x    = data[var][half:]
            mean = np.mean(x)
            sem  = best_block_sem(x)
            f.write(f'  {var:<10}  {mean:>12.4f}  {sem:>12.4f}  {unit:<6}  {target}\n')
    print(f'Saved {os.path.relpath(out_txt)}')