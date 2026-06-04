#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os


BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
FIG_DIR   = os.path.join(BASE_DIR, '../figures')
DATA_FILE = os.path.join(BASE_DIR, '../1_Analysis/trajectory.xyz')

style_candidates = [
    os.path.join(BASE_DIR, 'lib/science.mplstyle'),
    os.path.join(BASE_DIR, '../mplstyle/science.mplstyle'),
]
for style_file in style_candidates:
    if os.path.exists(style_file):
        plt.style.use(style_file)
        break

os.makedirs(FIG_DIR, exist_ok=True)

OXYGEN_LABEL = 'OH2'
DT_PS = 20.0
MAX_DT = 500
N_SLICES = 5


def read_xyz(filename):
    frames = []

    with open(filename) as f:
        while True:
            nline = f.readline()
            if not nline:
                break

            try:
                n = int(nline.strip())
            except ValueError:
                continue

            f.readline()
            coords = []

            for _ in range(n):
                p = f.readline().split()
                if len(p) >= 4 and p[0] == OXYGEN_LABEL:
                    coords.append([float(p[1]), float(p[2]), float(p[3])])

            if coords:
                frames.append(np.array(coords, dtype=float))
            else:
                frames.append(np.empty((0, 3), dtype=float))

    return frames


def average_snapshot_correlation(frames, max_dt):
    non_empty = [f for f in frames if len(f) > 0]
    min_n = min((len(f) for f in non_empty), default=0)

    if min_n == 0:
        return np.array([]), np.array([])

    dtv, cv = [], []

    for dt in range(0, min(max_dt, len(frames) - 1) + 1):
        num_vals = []
        den_vals = []

        for t0 in range(len(frames) - dt):
            n = min(len(frames[t0]), len(frames[t0 + dt]), min_n)
            if n == 0:
                continue

            r0 = frames[t0][:n]
            r1 = frames[t0 + dt][:n]

            num_vals.extend(np.sum(r0 * r1, axis=1))
            den_vals.extend(np.sum(r0 * r0, axis=1))

        if num_vals and den_vals:
            dtv.append(dt * DT_PS)
            cv.append(np.mean(num_vals) / np.mean(den_vals))

    return np.array(dtv), np.array(cv)


def msd(frames, max_dt):
    non_empty = [f for f in frames if len(f) > 0]
    min_n = min((len(f) for f in non_empty), default=0)

    if min_n == 0:
        return np.array([]), np.array([]), np.array([])

    dtv, yv, ev = [], [], []

    for dt in range(1, min(max_dt, len(frames) - 1) + 1):
        vals = []

        for t0 in range(len(frames) - dt):
            n = min(len(frames[t0]), len(frames[t0 + dt]), min_n)
            if n == 0:
                continue

            d = frames[t0 + dt][:n] - frames[t0][:n]
            vals.append(np.sum(d * d, axis=1))

        if vals:
            vals = np.concatenate(vals)
            dtv.append(dt * DT_PS)
            yv.append(vals.mean())
            ev.append(vals.std() / np.sqrt(len(vals)))

    return np.array(dtv), np.array(yv), np.array(ev)


def vacf(frames, max_dt):
    non_empty = [f for f in frames if len(f) > 0]
    min_n = min((len(f) for f in non_empty), default=0)

    if min_n == 0 or len(frames) < 3:
        return np.array([]), np.array([])

    vels = []
    for t in range(1, len(frames) - 1):
        n = min(len(frames[t - 1]), len(frames[t + 1]), min_n)
        if n == 0:
            continue
        vels.append((frames[t + 1][:n] - frames[t - 1][:n]) / (2 * DT_PS))

    if not vels:
        return np.array([]), np.array([])

    c0 = np.mean(np.concatenate([np.sum(v * v, axis=1) for v in vels]))

    dtv, yv = [], []
    for dt in range(0, min(max_dt, len(vels) - 1) + 1):
        vals = []

        for i in range(len(vels) - dt):
            n = min(len(vels[i]), len(vels[i + dt]))
            vals.extend(np.sum(vels[i][:n] * vels[i + dt][:n], axis=1))

        if vals:
            dtv.append(dt * DT_PS)
            yv.append(np.mean(vals) / c0)

    return np.array(dtv), np.array(yv)


def per_slice_msd(frames, max_dt, n_slices):
    non_empty = [f for f in frames if len(f) > 0]
    if not non_empty:
        return np.array([]), np.empty((n_slices, 0)), np.array([])

    allz = np.concatenate([f[:, 2] for f in non_empty])
    edges = np.linspace(allz.min(), allz.max(), n_slices + 1)

    min_n = min(len(x) for x in non_empty)
    n_dt = min(max_dt, len(frames) - 1)
    mat = np.full((n_slices, n_dt), np.nan)

    for dt in range(1, n_dt + 1):
        bucket = [[] for _ in range(n_slices)]

        for t0 in range(len(frames) - dt):
            n = min(len(frames[t0]), len(frames[t0 + dt]), min_n)
            if n == 0:
                continue

            z0 = frames[t0][:n, 2]
            sid = np.clip(np.digitize(z0, edges) - 1, 0, n_slices - 1)

            d = frames[t0 + dt][:n] - frames[t0][:n]
            d2 = np.sum(d * d, axis=1)

            for s in range(n_slices):
                m = sid == s
                if np.any(m):
                    bucket[s].append(np.mean(d2[m]))

        for s in range(n_slices):
            if bucket[s]:
                mat[s, dt - 1] = np.mean(bucket[s])

    return np.arange(1, n_dt + 1) * DT_PS, mat, edges


def save_curve(x, y, title, ylabel, outfile, yerr=None, hline=None, hline_label=None):
    if len(x) == 0:
        print(f'Skipping {os.path.basename(outfile)}: no data found')
        return

    fig, ax = plt.subplots()

    if yerr is not None and len(yerr):
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.2)

    ax.plot(x, y, marker='.', label=title)

    if hline is not None:
        ax.axhline(hline, color='red', linestyle='--', label=hline_label)

    ax.set_xlabel(r'Lag time (ps)')
    ax.set_ylabel(ylabel)
    ax.legend(loc='best')
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f'Saved {os.path.relpath(outfile)}')


def save_slice_plot(x, mat, edges, outfile):
    if len(x) == 0 or mat.size == 0 or len(edges) == 0:
        print(f'Skipping {os.path.basename(outfile)}: no slice data found')
        return

    fig, ax = plt.subplots()

    for i in range(mat.shape[0]):
        m = ~np.isnan(mat[i])
        if np.any(m):
            ax.plot(
                x[m],
                mat[i, m],
                marker='.',
                label=rf'${edges[i]:.1f} < z < {edges[i+1]:.1f}$'
            )

    ax.set_xlabel(r'Lag time (ps)')
    ax.set_ylabel(r'MSD ($\mathrm{A}^2$)')
    ax.legend(loc='best')
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f'Saved {os.path.relpath(outfile)}')


if __name__ == '__main__':
    print(f'Reading: {DATA_FILE}')
    frames = read_xyz(DATA_FILE)
    print(f'Found {len(frames)} frames')

    n_oxy = [len(f) for f in frames if len(f) > 0]
    if n_oxy:
        print(f'Found {len(n_oxy)} frames with {OXYGEN_LABEL} atoms; mean count = {np.mean(n_oxy):.1f}')
    else:
        print(f'Warning: no atoms with label {OXYGEN_LABEL} were found')

    xc, yc = average_snapshot_correlation(frames, MAX_DT)
    if len(xc):
        np.savetxt(
            os.path.join(FIG_DIR, 'corr.csv'),
            np.c_[xc, yc],
            delimiter=',',
            header='dt_ps,C_snapshot',
            comments=''
        )
    save_curve(
        xc, yc,
        r'Snapshot correlation',
        r'$C(\Delta t)$',
        os.path.join(FIG_DIR, 'corr.pdf'),
        hline=0.0,
        hline_label=r'$0$'
    )

    x, y, e = msd(frames, MAX_DT)
    if len(x):
        np.savetxt(
            os.path.join(FIG_DIR, 'msd.csv'),
            np.c_[x, y, e],
            delimiter=',',
            header='dt_ps,MSD_A2,SEM',
            comments=''
        )
    save_curve(
        x, y,
        r'MSD',
        r'MSD ($\mathrm{A}^2$)',
        os.path.join(FIG_DIR, 'msd.pdf'),
        yerr=e
    )

    xv, yv = vacf(frames, MAX_DT)
    if len(xv):
        np.savetxt(
            os.path.join(FIG_DIR, 'vacf.csv'),
            np.c_[xv, yv],
            delimiter=',',
            header='dt_ps,VACF',
            comments=''
        )
    save_curve(
        xv, yv,
        r'VACF',
        r'VACF($\Delta t$)',
        os.path.join(FIG_DIR, 'vacf.pdf'),
        hline=0.0,
        hline_label=r'$0$'
    )

    xs, mat, edges = per_slice_msd(frames, MAX_DT, N_SLICES)
    if len(xs):
        np.savetxt(
            os.path.join(FIG_DIR, 'msd_per_slice.csv'),
            np.column_stack([xs] + [mat[i] for i in range(mat.shape[0])]),
            delimiter=','
        )
    save_slice_plot(
        xs, mat, edges,
        os.path.join(FIG_DIR, 'msd_per_slice.pdf')
    )

    print('Done.')