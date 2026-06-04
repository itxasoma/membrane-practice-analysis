#!/usr/bin/env python3
"""
prod.py — NVT production run analysis for membrane MD simulations.

Checks that the NVT ensemble is satisfied:
  - Temperature should fluctuate around T_TARGET within ~1σ.
  - Volume should be constant (fixed in NVT) — any drift signals a problem.
  - Energies should be stable (no drift in total or potential energy).

The z-score / thermalized-band approach is adapted from:
  Muñoz-Aldalur, I. (2025). BirthMonth.ipynb.
  GitHub: https://github.com/itxasoma/Education-Big-Data-Project/blob/master/Notebooks/BirthMonth.ipynb

Color conventions (matching science.mplstyle cycle):
  K (kinetic)   : blue    #0C5DA5
  Etot (total)  : green   #00B945
  Upot (pot.)   : yellow  #FF9500
  T (temp.)     : violet  #845B97
  V (volume)    : gray    #474747
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR  = os.path.join(BASE_DIR, '../figures')
LOG_FILE = os.path.join(BASE_DIR, '../0_Simulation/Produccio_NVT/168015/nvt.log')

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
V_TARGET = None     # Å³ — set to e.g. 251210.9 to draw a target line (read from TS=0)

# Base colors
C_K    = '#0C5DA5'  # blue   — kinetic
C_ETOT = '#00B945'  # green  — total energy
C_UPOT = '#FF9500'  # yellow — potential
C_T    = '#845B97'  # violet — temperature
C_V    = '#474747'  # gray   — volume

COLUMNS = [
    'TS', 'BOND', 'ANGLE', 'DIHED', 'IMPRP',
    'ELECT', 'VDW', 'BOUNDARY', 'MISC', 'KINETIC',
    'TOTAL', 'TEMP', 'POTENTIAL', 'TOTAL3', 'TEMPAVG',
    'PRESSURE', 'GPRESSURE', 'VOLUME', 'PRESSAVG', 'GPRESSAVG'
]


def darken(hex_color, factor=0.55):
    """Return a darker shade of hex_color by scaling RGB towards black."""
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
    data['TIME_PS'] = data['TS'] * 2e-3   # timestep 2 fs → ps
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
    """Full y-range with 5% padding — for raw plots."""
    lo, hi = np.min(x), np.max(x)
    pad = 0.05 * (hi - lo) if hi != lo else 1.0
    return lo - pad, hi + pad


def _zscore_ylim(x, n_sigma=3.5):
    """
    Tighter y-limits for z-score plots: centred on the stable second-half
    mean, showing ±n_sigma * second-half std so the bands are clearly visible
    without early transients dominating the scale.
    """
    half         = len(x) // 2
    mu_stable    = np.mean(x[half:])
    sigma_stable = np.std(x[half:])
    pad = n_sigma * sigma_stable
    return mu_stable - pad, mu_stable + pad


# ── Energies ─────────────────────────────────────────────────────────────────

def save_energies(t, kinetic, total, potential):
    """
    K, E_tot, U_pot on a single axes.
    In NVT the total energy should be stationary; any drift signals
    incomplete equilibration or a thermostat problem.
    """
    fig, ax = plt.subplots()
    ax.plot(t, kinetic,   marker='.', color=C_K,    label=r'$K$')
    ax.plot(t, total,     marker='.', color=C_ETOT, label=r'$E_{\mathrm{tot}}$')
    ax.plot(t, potential, marker='.', color=C_UPOT, label=r'$U_{\mathrm{pot}}$')
    ax.set_xlabel(r'Time (ps)')
    ax.set_ylabel(r'Energy (kcal/mol)')
    ax.legend(loc='best')
    fig.tight_layout()
    out = os.path.join(FIG_DIR, '1.energies.pdf')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved {os.path.relpath(out)}')


# ── Temperature ──────────────────────────────────────────────────────────────

def save_temperature(t, temp, temp_avg, ylim):
    """
    Raw temperature trace with the block average and target line.
    In NVT the Langevin thermostat keeps <T> ≈ T_TARGET.
    """
    fig, ax = plt.subplots()
    ax.plot(t, temp, marker='.', color=C_T, label=r'$T$')
    ax.axhline(temp_avg[-1], color=darken(C_T), linewidth=1.2, linestyle='--',
               label=rf'$T_{{\mathrm{{avg,final}}}} = {temp_avg[-1]:.2f}\ \mathrm{{K}}$')
    ax.axhline(T_TARGET, color='black', linewidth=0.8, linestyle=':',
               label=rf'$T_{{\mathrm{{target}}}} = {T_TARGET}\ \mathrm{{K}}$')
    ax.set_xlabel(r'Time (ps)')
    ax.set_ylabel(r'Temperature (K)')
    ax.set_ylim(ylim)
    ax.legend(loc='best')
    fig.tight_layout()
    out = os.path.join(FIG_DIR, '1.temperature.pdf')
    #fig.savefig(out, dpi=150)
    #plt.close(fig)
    print(f'Saved {os.path.relpath(out)}')


def save_temperature_zscore(t, temp, ylim):
    """
    Thermalized-band plot for T (no instantaneous points).
    ±1σ and ±2σ running bands show whether T_TARGET is inside
    the normal fluctuation range throughout the production run.
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
    ax.set_xlabel(r'Time (ps)')
    ax.set_ylabel(r'Temperature (K)')
    ax.set_ylim(ylim)
    ax.legend(loc='best')
    fig.tight_layout()
    out = os.path.join(FIG_DIR, '1.temperature-zscore.pdf')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    z_final = (mu[-1] - T_TARGET) / sigma[-1] if sigma[-1] > 0 else float('nan')
    print(f'Saved {os.path.relpath(out)}  (final z = {z_final:.3f})')


# ── Volume ───────────────────────────────────────────────────────────────────

def save_volume(t, volume, ylim, v_target):
    """
    Raw volume trace.
    In NVT the box is fixed, so V should be constant (flat line).
    Any variation here would point to a configuration error.
    v_target: if not None, draws a dotted target line (initial volume from TS=0).
    """
    fig, ax = plt.subplots()
    ax.plot(t, volume, marker='.', color=C_V, label=r'$V$')
    if v_target is not None:
        ax.axhline(v_target, color='black', linewidth=0.8, linestyle=':',
                   label=rf'$V_{{\mathrm{{target}}}} = {v_target:.1f}\ \mathrm{{\AA}}^3$')
    ax.set_xlabel(r'Time (ps)')
    ax.set_ylabel(r'Volume ($\mathrm{\AA}^3$)')
    ax.set_ylim(ylim)
    ax.legend(loc='best')
    fig.tight_layout()
    out = os.path.join(FIG_DIR, '1.volume.pdf')
    #fig.savefig(out, dpi=150)
    #plt.close(fig)
    print(f'Saved {os.path.relpath(out)}')


def save_volume_zscore(t, volume, ylim, v_target):
    """
    Thermalized-band plot for V.
    In NVT the volume is fixed, so the running mean should be flat and
    the bands should be very narrow (essentially just numerical noise).
    Any widening of the bands or drift in the mean signals a problem.
    Ref: Muñoz-Aldalur, BirthMonth.ipynb (2025).
    """
    mu, sigma = running_stats(volume)

    fig, ax = plt.subplots()
    ax.fill_between(t, mu - 2*sigma, mu + 2*sigma,
                    color=C_V, alpha=0.15, label=r'$\pm 2\sigma$ ($\sim$95\%)')
    ax.fill_between(t, mu - sigma, mu + sigma,
                    color=C_V, alpha=0.40, label=r'$\pm 1\sigma$ ($\sim$68\%)')
    ax.plot(t, mu, color=darken(C_V), linewidth=1.4, linestyle='--',
            label=r'running mean $\langle V \rangle$')
    if v_target is not None:
        ax.axhline(v_target, color='black', linewidth=0.8, linestyle=':',
                   label=rf'$V_{{\mathrm{{target}}}} = {v_target:.1f}\ \mathrm{{\AA}}^3$')
    ax.set_xlabel(r'Time (ps)')
    ax.set_ylabel(r'Volume ($\mathrm{\AA}^3$)')
    ax.set_ylim(ylim)
    ax.legend(loc='best')
    fig.tight_layout()
    out = os.path.join(FIG_DIR, '1.volume-zscore.pdf')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved {os.path.relpath(out)}')


# ── Summary ───────────────────────────────────────────────────────────────────

def save_summary(data):
    out = os.path.join(FIG_DIR, '1.production_summary.dat')
    arr = np.column_stack([
        data['TIME_PS'], data['TEMP'], data['VOLUME'],
        data['KINETIC'], data['POTENTIAL'], data['TOTAL']
    ])
    np.savetxt(out, arr,
               header='time_ps temp_K volume_A3 kinetic potential total')
    print(f'Saved {os.path.relpath(out)}')


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f'Reading: {LOG_FILE}')
    data = load_log(LOG_FILE)
    print(f'Found {len(data["TS"])} energy frames'
          f'  (TS range: {int(data["TS"][0])}-{int(data["TS"][-1])})')

    t = data['TIME_PS']

    # V_TARGET: use first frame volume if not set manually above
    v_target = V_TARGET if V_TARGET is not None else float(data['VOLUME'][0])
    print(f'V_target = {v_target:.2f} Å³  (from {"config" if V_TARGET else "first frame"})')

    # Raw plots: full fluctuation range
    ylim_T_raw = _shared_ylim(data['TEMP'])
    ylim_V_raw = _shared_ylim(data['VOLUME'])

    # Z-score plots: tighter range centred on stable second-half stats
    ylim_T_z = _zscore_ylim(data['TEMP'],   n_sigma=3.5)
    ylim_V_z = _zscore_ylim(data['VOLUME'], n_sigma=3.5)

    save_energies(t, data['KINETIC'], data['TOTAL'], data['POTENTIAL'])
    save_temperature(t, data['TEMP'], data['TEMPAVG'], ylim_T_raw)
    save_temperature_zscore(t, data['TEMP'], ylim_T_z)
    save_volume(t, data['VOLUME'], ylim_V_raw, v_target)
    save_volume_zscore(t, data['VOLUME'], ylim_V_z, v_target)
    save_summary(data)

    half = len(t) // 2
    print('\n-- Production stats (second half of the run) --')
    for var, unit in [
        ('TOTAL',    'kcal/mol'),
        ('POTENTIAL','kcal/mol'),
        ('TEMP',     'K'),
        ('VOLUME',   'A3'),
    ]:
        vals = data[var][half:]
        print(f'  {var:12s}: mean = {np.mean(vals):10.2f}'
              f'   std = {np.std(vals):8.2f}   {unit}')