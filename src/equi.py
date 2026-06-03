#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
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


def save_temperature(t, temp, temp_avg):
    fig, ax = plt.subplots()
    ax.plot(t, temp, marker='.', label=r'$T$')
    ax.axhline(temp_avg[-1], color='red', linestyle='--',
               label=rf'$T_{{\mathrm{{avg,final}}}} = {temp_avg[-1]:.2f}\ \mathrm{{K}}$')
    ax.set_xlabel(r'Time (ns)')
    ax.set_ylabel(r'Temperature (K)')
    ax.legend(loc='best')

    out = os.path.join(FIG_DIR, 'Temperature.pdf')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved {os.path.relpath(out)}')


def save_pressure(t, press, press_avg):
    fig, ax = plt.subplots()
    ax.plot(t, press, marker='.', label=r'$P$')
    ax.plot(t, press_avg, label=r'$P_{\mathrm{avg}}$')
    ax.axhline(1.01325, color='red', linestyle='--',
               label=r'$P_{\mathrm{target}} = 1.01325\ \mathrm{bar}$')
    ax.set_xlabel(r'Time (ns)')
    ax.set_ylabel(r'Pressure (bar)')
    ax.legend(loc='best')

    out = os.path.join(FIG_DIR, 'Pressure.pdf')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved {os.path.relpath(out)}')


def save_volume(t, volume):
    fig, ax = plt.subplots()
    ax.plot(t, volume, marker='.', label=r'$V$')
    ax.set_xlabel(r'Time (ns)')
    ax.set_ylabel(r'Volume ($\mathrm{A}^3$)')
    ax.legend(loc='best')

    out = os.path.join(FIG_DIR, 'Volume.pdf')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved {os.path.relpath(out)}')


def save_energies(t, kinetic, total, potential):
    fig, ax = plt.subplots()
    ax.plot(t, kinetic, marker='.', label=r'$K$')
    ax.plot(t, total, marker='.', label=r'$E_{\mathrm{tot}}$')
    ax.plot(t, potential, marker='.', label=r'$U_{\mathrm{pot}}$')
    ax.set_xlabel(r'Time (ns)')
    ax.set_ylabel(r'Energy (kcal/mol)')
    ax.legend(loc='best')

    out = os.path.join(FIG_DIR, 'Energies.pdf')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved {os.path.relpath(out)}')


def save_total_energy(t, total):
    fig, ax = plt.subplots()
    ax.plot(t, total, marker='.', label=r'$E_{\mathrm{tot}}$')
    ax.set_xlabel(r'Time (ns)')
    ax.set_ylabel(r'Total energy (kcal/mol)')
    ax.legend(loc='best')

    out = os.path.join(FIG_DIR, 'TotalEnergy.pdf')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved {os.path.relpath(out)}')


def save_bonded(t, bond, angle, dihed):
    fig, ax = plt.subplots()
    ax.plot(t, bond, marker='.', label=r'$E_{\mathrm{bond}}$')
    ax.plot(t, angle, marker='.', label=r'$E_{\mathrm{angle}}$')
    ax.plot(t, dihed, marker='.', label=r'$E_{\mathrm{dihed}}$')
    ax.set_xlabel(r'Time (ns)')
    ax.set_ylabel(r'Energy (kcal/mol)')
    ax.legend(loc='best')

    out = os.path.join(FIG_DIR, 'Bonded.pdf')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved {os.path.relpath(out)}')


def save_summary(data):
    out = os.path.join(FIG_DIR, 'equilibration_summary.dat')
    arr = np.column_stack([
        data['TIME_NS'],
        data['TEMP'],
        data['PRESSURE'],
        data['VOLUME'],
        data['KINETIC'],
        data['POTENTIAL'],
        data['TOTAL']
    ])
    np.savetxt(
        out,
        arr,
        header='time_ns temp_K pressure_bar volume_A3 kinetic potential total'
    )
    print(f'Saved {os.path.relpath(out)}')


if __name__ == '__main__':
    print(f'Reading: {LOG_FILE}')
    data = load_log(LOG_FILE)
    print(f'Found {len(data["TS"])} energy frames  (TS range: {int(data["TS"][0])}-{int(data["TS"][-1])})')

    t = data['TIME_NS']
    save_temperature(t, data['TEMP'], data['TEMPAVG'])
    save_pressure(t, data['PRESSURE'], data['PRESSAVG'])
    save_volume(t, data['VOLUME'])
    save_energies(t, data['KINETIC'], data['TOTAL'], data['POTENTIAL'])
    save_total_energy(t, data['TOTAL'])
    save_bonded(t, data['BOND'], data['ANGLE'], data['DIHED'])
    save_summary(data)

    half = len(t) // 2
    print('\n-- Equilibration stats (second half of the run) --')
    for var, unit in [
        ('TOTAL', 'kcal/mol'),
        ('POTENTIAL', 'kcal/mol'),
        ('TEMP', 'K'),
        ('PRESSURE', 'bar'),
        ('VOLUME', 'A3')
    ]:
        vals = data[var][half:]
        print(f'  {var:12s}: mean = {np.mean(vals):10.2f}   std = {np.std(vals):8.2f}   {unit}')