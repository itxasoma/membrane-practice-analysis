#!/usr/bin/env python3
"""
prod.py — NVT production run analysis for membrane MD simulations.

Checks that the NVT ensemble is satisfied:
  - Temperature should fluctuate around T_TARGET within thermal fluctuations.
  - Volume should be constant (fixed in NVT) — any drift signals a problem.
  - Energies should be stable (no drift in total or potential energy).

Added temperature diagnostics:
  - Production-window mean temperature
  - Block-averaged uncertainty on <T>
  - First-half vs second-half comparison inside production
  - Linear drift slope dT/dt
  - z-score of production mean vs T_TARGET
  - Blocking plot: SEM(<T>) vs block length
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR  = os.path.join(BASE_DIR, '../figures')
LOG_FILE = os.path.join(BASE_DIR, '../0_Simulation/Produccio_NVT/Vctt/nvt.log')

try:
    plt.style.use('lib/science.mplstyle')
except OSError:
    pass

os.makedirs(FIG_DIR, exist_ok=True)

# Edit to match your simulation targets
T_TARGET = 290.5    # K
V_TARGET = None     # Å³ — set to e.g. 251210.9 to draw a target line (read from TS=0)
P_TARGET = 1.01325  # bar

# Production window:
#   None  -> use second half of the run
#   float -> use all frames with t >= PROD_START_PS
PROD_START_PS = None

# Candidate numbers of blocks for blocking analysis
BLOCK_COUNTS = [5, 8, 10, 12, 15, 20, 25, 30, 40, 50]

# Base colors
C_K    = '#0C5DA5'  # blue   — kinetic
C_ETOT = '#00B945'  # green  — total energy
C_UPOT = '#FF9500'  # yellow — potential
C_T    = '#845B97'  # violet — temperature
C_V    = '#474747'  # gray   — volume
C_P    = '#FF2C00'  # red    — pressure

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
    data['TIME_PS'] = data['TS'] * 2e-3   # timestep 2 fs -> ps
    return data


def running_stats(x):
    n       = np.arange(1, len(x) + 1, dtype=float)
    cumsum  = np.cumsum(x)
    cumsum2 = np.cumsum(x ** 2)
    mu      = cumsum / n
    var     = np.maximum(cumsum2 / n - mu ** 2, 0.0)
    sigma   = np.sqrt(var)
    return mu, sigma


def _shared_ylim(x):
    lo, hi = np.min(x), np.max(x)
    pad = 0.05 * (hi - lo) if hi != lo else 1.0
    return lo - pad, hi + pad


def _zscore_ylim(x, n_sigma=3.5):
    half         = len(x) // 2
    mu_stable    = np.mean(x[half:])
    sigma_stable = np.std(x[half:])
    pad = n_sigma * sigma_stable if sigma_stable > 0 else 1.0
    return mu_stable - pad, mu_stable + pad


def get_prod_slice(t):
    if PROD_START_PS is None:
        i0 = len(t) // 2
        label = 'second half of the run'
    else:
        i0 = int(np.searchsorted(t, PROD_START_PS))
        if i0 >= len(t) - 1:
            i0 = len(t) // 2
            label = 'second half of the run (fallback)'
        else:
            label = f't >= {PROD_START_PS:.3f} ps'
    return slice(i0, len(t)), label


def block_mean_sem(x, n_blocks):
    n = len(x)
    if n_blocks < 2 or n < n_blocks:
        return np.nan, np.nan, np.nan, 0, 0

    block_len = n // n_blocks
    if block_len < 1:
        return np.nan, np.nan, np.nan, 0, 0

    n_trim = block_len * n_blocks
    xb = x[:n_trim].reshape(n_blocks, block_len).mean(axis=1)

    mean = float(np.mean(xb))
    std_blocks = float(np.std(xb, ddof=1)) if n_blocks > 1 else np.nan
    sem = std_blocks / np.sqrt(n_blocks) if n_blocks > 1 else np.nan
    return mean, std_blocks, sem, n_blocks, block_len


def blocking_curve(x, dt_ps, block_counts=BLOCK_COUNTS):
    rows = []
    for n_blocks in block_counts:
        mean, std_blocks, sem, nb, block_len = block_mean_sem(x, n_blocks)
        if np.isfinite(sem):
            rows.append([block_len * dt_ps, sem, nb, block_len, mean, std_blocks])
    return np.array(rows, dtype=float) if rows else np.empty((0, 6), dtype=float)


def recommended_block_sem(curve, fallback):
    if curve.size == 0:
        return float(fallback)
    sems = curve[:, 1]
    k = min(3, len(sems))
    return float(np.median(sems[-k:]))


def linear_drift(t, x):
    if len(t) < 2:
        return np.nan, np.nan
    slope, intercept = np.polyfit(t, x, 1)
    return float(slope), float(intercept)


def temperature_diagnostics(t, temp):
    prod_sl, prod_label = get_prod_slice(t)
    tp = t[prod_sl]
    xp = temp[prod_sl]

    if len(tp) < 4:
        raise ValueError('Not enough production frames for temperature diagnostics.')

    dt_ps = float(np.median(np.diff(tp))) if len(tp) > 1 else float(np.median(np.diff(t)))

    mean = float(np.mean(xp))
    std = float(np.std(xp, ddof=1))
    naive_sem = std / np.sqrt(len(xp))

    curve = blocking_curve(xp, dt_ps)
    block_sem = recommended_block_sem(curve, naive_sem)

    slope, intercept = linear_drift(tp, xp)

    mid = len(xp) // 2
    x1, x2 = xp[:mid], xp[mid:]
    t1, t2 = tp[:mid], tp[mid:]

    mean1 = float(np.mean(x1))
    mean2 = float(np.mean(x2))

    curve1 = blocking_curve(x1, dt_ps)
    curve2 = blocking_curve(x2, dt_ps)

    sem1 = recommended_block_sem(curve1, np.std(x1, ddof=1) / np.sqrt(len(x1))) if len(x1) > 1 else np.nan
    sem2 = recommended_block_sem(curve2, np.std(x2, ddof=1) / np.sqrt(len(x2))) if len(x2) > 1 else np.nan

    delta_halves = mean2 - mean1
    delta_halves_sem = np.sqrt(sem1**2 + sem2**2) if np.isfinite(sem1) and np.isfinite(sem2) else np.nan
    z_halves = delta_halves / delta_halves_sem if np.isfinite(delta_halves_sem) and delta_halves_sem > 0 else np.nan
    z_target = (mean - T_TARGET) / block_sem if block_sem > 0 else np.nan

    return {
        'prod_label': prod_label,
        't_prod': tp,
        'temp_prod': xp,
        'mean': mean,
        'std': std,
        'naive_sem': float(naive_sem),
        'block_sem': float(block_sem),
        'slope_K_per_ps': slope,
        'intercept_K': intercept,
        'first_half_mean': mean1,
        'second_half_mean': mean2,
        'first_half_sem': sem1,
        'second_half_sem': sem2,
        'delta_halves': float(delta_halves),
        'delta_halves_sem': float(delta_halves_sem) if np.isfinite(delta_halves_sem) else np.nan,
        'z_halves': float(z_halves) if np.isfinite(z_halves) else np.nan,
        'z_target': float(z_target) if np.isfinite(z_target) else np.nan,
        'curve': curve,
    }


# ── Energies ─────────────────────────────────────────────────────────────────


def save_energies(t, kinetic, total, potential):
    fig, ax = plt.subplots()
    ax.plot(t, kinetic,   marker='.', color=C_K,    label=r'$K$')
    ax.plot(t, total,     marker='.', color=C_ETOT, label=r'$E_{\mathrm{tot}}$')
    ax.plot(t, potential, marker='.', color=C_UPOT, label=r'$U_{\mathrm{pot}}$')
    ax.set_xlabel(r'Time (ps)')
    ax.set_ylabel(r'Energy (kcal/mol)')
    ax.legend(loc='best')
    fig.tight_layout()
    out = os.path.join(FIG_DIR, '1.energies_Vctt.pdf')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved {os.path.relpath(out)}')


# ── Temperature ──────────────────────────────────────────────────────────────


def save_temperature(t, temp, temp_avg, ylim):
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
    out = os.path.join(FIG_DIR, '1.temperature_Vctt.pdf')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved {os.path.relpath(out)}')


def save_temperature_zscore(t, temp, ylim):
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
    out = os.path.join(FIG_DIR, '1.temperature-zscore_Vctt.pdf')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    z_final = (mu[-1] - T_TARGET) / sigma[-1] if sigma[-1] > 0 else float('nan')
    print(f'Saved {os.path.relpath(out)}  (final z = {z_final:.3f})')


def save_temperature_blocking(diag):
    curve = diag['curve']
    if curve.size == 0:
        print('Skipping blocking plot: not enough data')
        return

    block_len_ps = curve[:, 0]
    sem = curve[:, 1]

    fig, ax = plt.subplots()
    ax.plot(block_len_ps, sem, marker='o', color=C_T)
    ax.axhline(diag['block_sem'], color=darken(C_T), linestyle='--', linewidth=1.2,
               label=rf'recommended block SEM = {diag["block_sem"]:.4f} K')
    ax.set_xlabel(r'Block length (ps)')
    ax.set_ylabel(r'Block SEM of $\langle T \rangle$ (K)')
    ax.legend(loc='best')
    fig.tight_layout()
    out = os.path.join(FIG_DIR, '1.temperature-blocking_Vctt.pdf')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved {os.path.relpath(out)}')


def save_temperature_block_means(diag, n_blocks=10):
    tp = diag['t_prod']
    xp = diag['temp_prod']

    n = len(xp)
    if n < n_blocks:
        n_blocks = max(2, n // 2)
    if n_blocks < 2:
        print('Skipping block-means plot: not enough data')
        return

    block_len = n // n_blocks
    n_trim = block_len * n_blocks
    if n_trim < 2:
        print('Skipping block-means plot: not enough trimmed data')
        return

    t_blocks = tp[:n_trim].reshape(n_blocks, block_len).mean(axis=1)
    x_blocks = xp[:n_trim].reshape(n_blocks, block_len).mean(axis=1)

    fig, ax = plt.subplots()
    ax.plot(t_blocks, x_blocks, marker='o', color=C_T, label='block means')
    ax.axhline(T_TARGET, color='black', linewidth=0.8, linestyle=':',
               label=rf'$T_{{\rm target}} = {T_TARGET}\ \mathrm{{K}}$')
    ax.axhline(diag['mean'], color=darken(C_T), linewidth=1.2, linestyle='--',
               label=rf'production mean = {diag["mean"]:.3f} K')
    ax.set_xlabel(r'Time (ps)')
    ax.set_ylabel(r'Block-averaged temperature (K)')
    ax.legend(loc='best')
    fig.tight_layout()
    out = os.path.join(FIG_DIR, '1.temperature-blockmeans_Vctt.pdf')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved {os.path.relpath(out)}')


def save_temperature_diagnostics(diag):
    out = os.path.join(FIG_DIR, '1.temperature_ctt_check_Vctt.dat')
    with open(out, 'w') as f:
        f.write('# Temperature constant-T diagnostics\n')
        f.write(f'production_window = {diag["prod_label"]}\n')
        f.write(f'mean_K = {diag["mean"]:.8f}\n')
        f.write(f'std_K = {diag["std"]:.8f}\n')
        f.write(f'naive_sem_K = {diag["naive_sem"]:.8f}\n')
        f.write(f'block_sem_K = {diag["block_sem"]:.8f}\n')
        f.write(f'target_K = {T_TARGET:.8f}\n')
        f.write(f'z_target = {diag["z_target"]:.8f}\n')
        f.write(f'drift_slope_K_per_ps = {diag["slope_K_per_ps"]:.8e}\n')
        f.write(f'first_half_mean_K = {diag["first_half_mean"]:.8f}\n')
        f.write(f'second_half_mean_K = {diag["second_half_mean"]:.8f}\n')
        f.write(f'first_half_sem_K = {diag["first_half_sem"]:.8f}\n')
        f.write(f'second_half_sem_K = {diag["second_half_sem"]:.8f}\n')
        f.write(f'delta_halves_K = {diag["delta_halves"]:.8f}\n')
        f.write(f'delta_halves_sem_K = {diag["delta_halves_sem"]:.8f}\n')
        f.write(f'z_halves = {diag["z_halves"]:.8f}\n')
        f.write('#\n')
        f.write('# block_length_ps  sem_K  n_blocks  block_len_frames  block_mean_K  std_block_means_K\n')
        for row in diag['curve']:
            f.write(f'{row[0]:.8f} {row[1]:.8f} {int(row[2])} {int(row[3])} {row[4]:.8f} {row[5]:.8f}\n')
    print(f'Saved {os.path.relpath(out)}')


# ── Volume ───────────────────────────────────────────────────────────────────


def save_volume(t, volume, ylim, v_target):
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
    out = os.path.join(FIG_DIR, '1.volume_Vctt.pdf')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved {os.path.relpath(out)}')


def save_volume_zscore(t, volume, ylim, v_target):
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
    out = os.path.join(FIG_DIR, '1.volume-zscore_Vctt.pdf')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved {os.path.relpath(out)}')


# ── Pressure ─────────────────────────────────────────────────────────────────


def save_pressure(t, press, press_avg, ylim):
    fig, ax = plt.subplots()
    ax.plot(t, press, marker='.', color=C_P, label=r'$P$')
    ax.plot(t, press_avg, color=darken(C_P), linewidth=1.2, linestyle='--',
            label=r'$P_{\mathrm{avg}}$')
    ax.axhline(P_TARGET, color='black', linewidth=0.8, linestyle=':',
               label=r'$P_{\mathrm{target}} = 1.01325\ \mathrm{bar}$')
    ax.set_xlabel(r'Time (ps)')
    ax.set_ylabel(r'Pressure (bar)')
    ax.set_ylim(ylim)
    ax.legend(loc='best')
    fig.tight_layout()
    out = os.path.join(FIG_DIR, '0.pressure_Vctt.pdf')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved {os.path.relpath(out)}')


def save_pressure_zscore(t, press, ylim):
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

    ax.set_xlabel(r'Time (ps)')
    ax.set_ylabel(r'Pressure (bar)')
    ax.set_ylim(ylim)
    ax.legend(loc='best')
    fig.tight_layout()
    out = os.path.join(FIG_DIR, '1.pressure-zscore_Vctt.pdf')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    z_final = (mu[-1] - P_TARGET) / sigma[-1] if sigma[-1] > 0 else float('nan')
    print(f'Saved {os.path.relpath(out)}  (final z = {z_final:.3f})')


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

    v_target = V_TARGET if V_TARGET is not None else float(data['VOLUME'][0])
    print(f'V_target = {v_target:.2f} Å³  (from {"config" if V_TARGET else "first frame"})')

    ylim_T_raw = _shared_ylim(data['TEMP'])
    ylim_V_raw = _shared_ylim(data['VOLUME'])

    ylim_T_z = _zscore_ylim(data['TEMP'], n_sigma=3.5)
    ylim_V_z = _zscore_ylim(data['VOLUME'], n_sigma=3.5)
    ylim_P_z = _zscore_ylim(data['PRESSURE'], n_sigma=3.5)

    save_energies(t, data['KINETIC'], data['TOTAL'], data['POTENTIAL'])
    save_temperature(t, data['TEMP'], data['TEMPAVG'], ylim_T_raw)
    save_temperature_zscore(t, data['TEMP'], ylim_T_z)
    save_volume(t, data['VOLUME'], ylim_V_raw, v_target)
    save_volume_zscore(t, data['VOLUME'], ylim_V_z, v_target)
    save_pressure_zscore(t, data['PRESSURE'], ylim_P_z)
    save_summary(data)

    diag = temperature_diagnostics(t, data['TEMP'])
    save_temperature_blocking(diag)
    save_temperature_block_means(diag, n_blocks=10)
    save_temperature_diagnostics(diag)

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

    print('\n-- Constant-T diagnostics --')
    print(f'  production window       : {diag["prod_label"]}')
    print(f'  <T>_prod                : {diag["mean"]:.4f} K')
    print(f'  std(T)_prod             : {diag["std"]:.4f} K')
    print(f'  naive SEM(<T>)          : {diag["naive_sem"]:.5f} K')
    print(f'  block SEM(<T>)          : {diag["block_sem"]:.5f} K')
    print(f'  z_target                : {diag["z_target"]:.4f}')
    print(f'  drift slope dT/dt       : {diag["slope_K_per_ps"]:.6e} K/ps')
    print(f'  first-half mean         : {diag["first_half_mean"]:.4f} K')
    print(f'  second-half mean        : {diag["second_half_mean"]:.4f} K')
    print(f'  delta(second-first)     : {diag["delta_halves"]:.4f} K')
    print(f'  SEM(delta)              : {diag["delta_halves_sem"]:.5f} K')
    print(f'  z_halves                : {diag["z_halves"]:.4f}')