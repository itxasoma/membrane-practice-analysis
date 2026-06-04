#!/usr/bin/env python3
"""
equi.py — NPT equilibration analysis for membrane MD simulations.

The z-score / thermalized-band approach used in save_*_zscore() functions
is adapted from:
  Muñoz-Aldalur, I. (2025). BirthMonth.ipynb.
  GitHub: https://github.com/itxasoma/Education-Big-Data-Project/blob/master/Notebooks/BirthMonth.ipynb

The idea: compute a running mean and running std from the simulation data itself
(the "null" distribution of expected fluctuations), then check whether the
target value (T_TARGET or P_TARGET) falls inside the 1σ / 2σ bands.
A target z-score |z| < 1 means the target is well within normal fluctuations.

Color conventions (matching science.mplstyle cycle):
  K (kinetic)   : blue    #0C5DA5
  Etot (total)  : green   #00B945
  Upot (pot.)   : yellow  #FF9500
  T (temp.)     : violet  #845B97
  P (pressure)  : red     #FF2C00
  V (volume)    : gray    #474747
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR  = os.path.join(BASE_DIR, '../figures')
LOG_FILE = os.path.join(BASE_DIR, '../0_Simulation/Equilibrat_NPT/npt.log')

style_candidates = [
    os.path.join(BASE_DIR, 'lib/science.mplstyle'),
    os.path.join(BASE_DIR, '../mplstyle/science.mplstyle'),
]
for style_file in style_candidates:
    if os.path.exists(style_file):
        plt.style.use(style_file)
        break

os.makedirs(FIG_DIR, exist_ok=True)

# Edit to match your simulation targets
T_TARGET = 209.5    # K
P_TARGET = 1.01325  # bar

# Base colors per variable
C_K    = '#0C5DA5'  # blue   — kinetic
C_ETOT = '#00B945'  # green  — total energy
C_UPOT = '#FF9500'  # yellow — potential
C_T    = '#845B97'  # violet — temperature
C_P    = '#FF2C00'  # red    — pressure
C_V    = '#474747'  # gray   — volume


def darken(hex_color, factor=0.55):
    """Return a darker version of hex_color by scaling RGB towards black."""
    r, g, b = mcolors.to_rgb(hex_color)
    return (r * factor, g * factor, b * factor)


COLUMNS = [
    'TS', 'BOND', 'ANGLE', 'DIHED', 'IMPRP',
    'ELECT', 'VDW', 'BOUNDARY', 'MISC', 'KINETIC',
    'TOTAL', 'TEMP', 'POTENTIAL', 'TOTAL3', 'TEMPAVG',
    'PRESSURE', 'GPRESSURE', 'VOLUME', 'PRESSAVG', 'GPRESSAVG'
]


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
    data['TIME_NS'] = data['TS'] * 2e-6
    return data


def running_stats(x):
    """
    Cumulative running mean and std.
    The band narrows as more data accumulates, as in BirthMonth.ipynb.
    """
    n       = np.arange(1, len(x) + 1, dtype=float)
    cumsum  = np.cumsum(x)
    cumsum2 = np.cumsum(x ** 2)
    mu      = cumsum / n
    var     = np.maximum(cumsum2 / n - mu ** 2, 0.0)
    sigma   = np.sqrt(var)
    return mu, sigma


def _shared_ylim(x):
    """y-limits with 5% padding."""
    lo, hi = np.min(x), np.max(x)
    pad = 0.05 * (hi - lo) if hi != lo else 1.0
    return lo - pad, hi + pad


def _zscore_ylim(x, n_sigma=3.5):
    """
    Tighter y-limits for z-score plots: centre on the final running mean
    and show ± n_sigma * final_sigma, so the bands fill the plot nicely
    without the ±2σ region dominating when early transients are large.
    """
    mu, sigma = running_stats(x)
    # Use the second-half stats to avoid early-transient inflation
    half = len(x) // 2
    mu_stable    = np.mean(x[half:])
    sigma_stable = np.std(x[half:])
    pad = n_sigma * sigma_stable
    return mu_stable - pad, mu_stable + pad


# ── Temperature ──────────────────────────────────────────────────────────────

def save_temperature(t, temp, temp_avg, ylim):
    fig, ax = plt.subplots()
    ax.plot(t, temp, marker='.', color=C_T, label=r'$T$')
    ax.axhline(temp_avg[-1], color=darken(C_T), linewidth=1.2, linestyle='--',
               label=rf'$T_{{\mathrm{{avg,final}}}} = {temp_avg[-1]:.2f}\ \mathrm{{K}}$')
    ax.axhline(T_TARGET, color='black', linewidth=0.8, linestyle=':',
               label=rf'$T_{{\mathrm{{target}}}} = {T_TARGET}\ \mathrm{{K}}$')
    ax.set_xlabel(r'Time (ns)')
    ax.set_ylabel(r'Temperature (K)')
    ax.set_ylim(ylim)
    ax.legend(loc='best')
    fig.tight_layout()
    out = os.path.join(FIG_DIR, '0.temperature.pdf')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved {os.path.relpath(out)}')


def save_temperature_zscore(t, temp, ylim):
    """
    Thermalized-band plot for T (no instantaneous points).
    Ref: Muñoz-Aldalur, BirthMonth.ipynb (2025).
    """
    mu, sigma = running_stats(temp)

    fig, ax = plt.subplots()

    ax.fill_between(t, mu - 2*sigma, mu + 2*sigma,
                    color=C_T, alpha=0.15, label=r'$\pm 2\sigma$ ($\sim$95\%)')
    ax.fill_between(t, mu - sigma, mu + sigma,
                    color=C_T, alpha=0.40, label=r'$\pm 1\sigma$ ($\sim$68\%)')
    ax.plot(t, mu, color=darken(C_T), linewidth=1.4, linestyle='--',
            label=r'running mean $\langle T \rangle$')
    ax.axhline(T_TARGET, color='black', linewidth=0.8, linestyle=':',
               label=rf'$T_{{\rm target}} = {T_TARGET}\ \mathrm{{K}}$')

    ax.set_xlabel(r'Time (ns)')
    ax.set_ylabel(r'Temperature (K)')
    ax.set_ylim(ylim)
    ax.legend(loc='lower right')
    fig.tight_layout()
    out = os.path.join(FIG_DIR, '0.temperature-zscore.pdf')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    z_final = (mu[-1] - T_TARGET) / sigma[-1] if sigma[-1] > 0 else float('nan')
    print(f'Saved {os.path.relpath(out)}  (final z = {z_final:.3f})')


# ── Pressure ─────────────────────────────────────────────────────────────────

def save_pressure(t, press, press_avg, ylim):
    fig, ax = plt.subplots()
    ax.plot(t, press, marker='.', color=C_P, label=r'$P$')
    ax.plot(t, press_avg, color=darken(C_P), linewidth=1.2, linestyle='--',
            label=r'$P_{\mathrm{avg}}$')
    ax.axhline(P_TARGET, color='black', linewidth=0.8, linestyle=':',
               label=r'$P_{\mathrm{target}} = 1.01325\ \mathrm{bar}$')
    ax.set_xlabel(r'Time (ns)')
    ax.set_ylabel(r'Pressure (bar)')
    ax.set_ylim(ylim)
    ax.legend(loc='best')
    fig.tight_layout()
    out = os.path.join(FIG_DIR, '0.pressure.pdf')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved {os.path.relpath(out)}')


def save_pressure_zscore(t, press, ylim):
    """
    Thermalized-band plot for P (no instantaneous points).
    y-axis clipped to ±3.5σ of the stable second half so the bands
    are visible and the legend is not buried.
    Ref: Muñoz-Aldalur, BirthMonth.ipynb (2025).
    """
    mu, sigma = running_stats(press)

    fig, ax = plt.subplots()

    ax.fill_between(t, mu - 2*sigma, mu + 2*sigma,
                    color=C_P, alpha=0.15, label=r'$\pm 2\sigma$ ($\sim$95\%)')
    ax.fill_between(t, mu - sigma, mu + sigma,
                    color=C_P, alpha=0.40, label=r'$\pm 1\sigma$ ($\sim$68\%)')
    ax.plot(t, mu, color=darken(C_P), linewidth=1.4, linestyle='--',
            label=r'running mean $\langle P \rangle$')
    ax.axhline(P_TARGET, color='black', linewidth=0.8, linestyle=':',
               label=rf'$P_{{\rm target}} = {P_TARGET:.3f}\ \mathrm{{bar}}$')

    ax.set_xlabel(r'Time (ns)')
    ax.set_ylabel(r'Pressure (bar)')
    ax.set_ylim(ylim)
    ax.legend(loc='best')
    fig.tight_layout()
    out = os.path.join(FIG_DIR, '0.pressure-zscore.pdf')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    z_final = (mu[-1] - P_TARGET) / sigma[-1] if sigma[-1] > 0 else float('nan')
    print(f'Saved {os.path.relpath(out)}  (final z = {z_final:.3f})')


# ── Volume ───────────────────────────────────────────────────────────────────

def save_volume(t, volume):
    fig, ax = plt.subplots()
    ax.plot(t, volume, marker='.', color=C_V, label=r'$V$')
    ax.set_xlabel(r'Time (ns)')
    ax.set_ylabel(r'Volume ($\mathrm{A}^3$)')
    ax.legend(loc='best')
    fig.tight_layout()
    out = os.path.join(FIG_DIR, '0.volume.pdf')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved {os.path.relpath(out)}')


# ── Energies ─────────────────────────────────────────────────────────────────

def save_energies(t, kinetic, total, potential):
    fig, ax = plt.subplots()
    ax.plot(t, kinetic,   marker='.', color=C_K,    label=r'$K$')
    ax.plot(t, total,     marker='.', color=C_ETOT, label=r'$E_{\mathrm{tot}}$')
    ax.plot(t, potential, marker='.', color=C_UPOT, label=r'$U_{\mathrm{pot}}$')
    ax.set_xlabel(r'Time (ns)')
    ax.set_ylabel(r'Energy (kcal/mol)')
    ax.legend(loc='best')
    fig.tight_layout()
    out = os.path.join(FIG_DIR, '0.energies.pdf')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved {os.path.relpath(out)}')


def save_total_energy(t, total):
    fig, ax = plt.subplots()
    ax.plot(t, total, marker='.', color=C_ETOT, label=r'$E_{\mathrm{tot}}$')
    ax.set_xlabel(r'Time (ns)')
    ax.set_ylabel(r'Total energy (kcal/mol)')
    ax.legend(loc='best')
    fig.tight_layout()
    out = os.path.join(FIG_DIR, '0.total_energy.pdf')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved {os.path.relpath(out)}')


def save_bonded(t, bond, angle, dihed):
    fig, ax = plt.subplots()
    ax.plot(t, bond,  marker='.', label=r'$E_{\mathrm{bond}}$')
    ax.plot(t, angle, marker='.', label=r'$E_{\mathrm{angle}}$')
    ax.plot(t, dihed, marker='.', label=r'$E_{\mathrm{dihed}}$')
    ax.set_xlabel(r'Time (ns)')
    ax.set_ylabel(r'Energy (kcal/mol)')
    ax.legend(loc='best')
    fig.tight_layout()
    out = os.path.join(FIG_DIR, '0.bonded.pdf')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved {os.path.relpath(out)}')


# ── Summary ───────────────────────────────────────────────────────────────────

def save_summary(data):
    out = os.path.join(FIG_DIR, 'equilibration_summary.dat')
    arr = np.column_stack([
        data['TIME_NS'], data['TEMP'], data['PRESSURE'], data['VOLUME'],
        data['KINETIC'], data['POTENTIAL'], data['TOTAL']
    ])
    np.savetxt(out, arr,
               header='time_ns temp_K pressure_bar volume_A3 kinetic potential total')
    print(f'Saved {os.path.relpath(out)}')


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f'Reading: {LOG_FILE}')
    data = load_log(LOG_FILE)
    print(f'Found {len(data["TS"])} energy frames'
          f'  (TS range: {int(data["TS"][0])}-{int(data["TS"][-1])})')

    t = data['TIME_NS']

    # Raw plots: full data range so all fluctuations are visible
    ylim_T_raw = _shared_ylim(data['TEMP'])
    ylim_P_raw = _shared_ylim(data['PRESSURE'])

    # Z-score plots: tighter range centred on stable second-half stats
    ylim_T_z = _zscore_ylim(data['TEMP'],     n_sigma=3.5)
    ylim_P_z = _zscore_ylim(data['PRESSURE'], n_sigma=3.5)

    save_temperature(t, data['TEMP'], data['TEMPAVG'], ylim_T_raw)
    save_temperature_zscore(t, data['TEMP'], ylim_T_z)

    save_pressure(t, data['PRESSURE'], data['PRESSAVG'], ylim_P_raw)
    save_pressure_zscore(t, data['PRESSURE'], ylim_P_z)

    save_volume(t, data['VOLUME'])
    save_energies(t, data['KINETIC'], data['TOTAL'], data['POTENTIAL'])
    save_total_energy(t, data['TOTAL'])
    save_bonded(t, data['BOND'], data['ANGLE'], data['DIHED'])
    save_summary(data)

    half = len(t) // 2
    print('\n-- Equilibration stats (second half of the run) --')
    for var, unit in [
        ('TOTAL', 'kcal/mol'), ('POTENTIAL', 'kcal/mol'),
        ('TEMP', 'K'), ('PRESSURE', 'bar'), ('VOLUME', 'A3')
    ]:
        vals = data[var][half:]
        print(f'  {var:12s}: mean = {np.mean(vals):10.2f}'
              f'   std = {np.std(vals):8.2f}   {unit}')