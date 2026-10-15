#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H2/H4 3- and 5-sample smoothing + exploratory H2 temporal-shift bootstrap.

Run main.py first. Then Run this file in PyCharm; settings in constant_variable.py.
Reads per-subject *_rsa.npz (H2 and H4) rather than figures or group averages.
Centered time smoothing is applied to the ALREADY computed per-sample contrasts,
not to EEG voltages and not to the neural RDM prior to calculating distances.
No latency, cluster onset, or compensatory processing claims are warranted from
this exploratory bootstrap analysis.
"""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import constant_variable as cfg


def load_subject_data(folder):
    folder = Path(folder)
    files = sorted(folder.glob('*_rsa.npz'))
    if not files:
        raise FileNotFoundError(f'No *_rsa.npz in {folder}. Run main.py on the complete sample first.')
    records = []
    times = None
    for file in files:
        with np.load(file, allow_pickle=True) as d:
            if not {'h2', 'h4', 'times', 'group'}.issubset(d.files):
                raise ValueError(f'{file}: missing h2/h4/times/group; use the H1-H4 main.py')
            t = np.asarray(d['times'], dtype=float).reshape(-1)
            if times is None:
                times = t
            elif not np.allclose(times, t, atol=1e-9, rtol=0):
                raise ValueError(f'Incompatible time axis: {file}')
            h2 = np.asarray(d['h2'], dtype=float)
            h4 = np.asarray(d['h4'], dtype=float)
            if h2.shape != (2, len(t)) or h4.shape != (2, len(t)):
                raise ValueError(f'{file}: unexpected shapes h2 {h2.shape}, h4 {h4.shape}')
            if not np.isfinite(h2).all() or not np.isfinite(h4).all():
                raise ValueError(f'{file}: NaN/Inf in contrasts')
            group = str(np.asarray(d['group']).item())
            if group not in ('GJ', 'GA'):
                raise ValueError(f'{file}: invalid group {group}')
            sid = str(np.asarray(d['subject']).item()) if 'subject' in d else file.stem[:-4]
            records.append((sid, group, h2, h4))
    subjects = np.array([r[0] for r in records])
    if len(set(subjects)) != len(subjects):
        raise ValueError('Duplicate subject identifiers in *_rsa.npz')
    groups = np.array([r[1] for r in records])
    if not {'GJ', 'GA'}.issubset(set(groups)):
        warnings.warn('Both age groups are needed for H2 shift and age statistics')
    return subjects, groups, times * 1000, {k: np.stack([r[i] for r in records])
                                                  for k, i in [('H2', 2), ('H4', 3)]}


def smooth(data, n):
    """Centered n-sample moving average, no shrinking boundary windows.
    NaN at n//2 samples per edge (avoid artificial edge padding).
    """
    if n % 2 != 1 or n < 1:
        raise ValueError('Smoothing must use a positive odd integer')
    if n == 1:
        return np.array(data, copy=True)
    out = np.full_like(data, np.nan, dtype=float)
    for idx in range(n // 2, data.shape[-1] - n // 2):
        out[..., idx] = np.mean(data[..., idx - n // 2:idx + n // 2 + 1], axis=-1)
    return out


def t_one(x):
    n = x.shape[0]
    mean = x.mean(axis=0)
    std = x.std(axis=0, ddof=1)
    return np.divide(mean, std / np.sqrt(n), out=np.zeros_like(mean), where=std > 0)


def t_two(a, b):
    diff = a.mean(axis=0) - b.mean(axis=0)
    denom = np.sqrt(a.var(axis=0, ddof=1) / len(a) + b.var(axis=0, ddof=1) / len(b))
    return np.divide(diff, denom, out=np.zeros_like(diff), where=denom > 0)


def permutation_stats(data, groups, times, n_perm, seed):
    """For each contrast, jointly correct H2/H4, 3/5 window, and all time points.
    Each contrast (semantic, SOA, AGE, interaction) is still a separate family.
    """
    rng = np.random.default_rng(seed)
    keys = [(name, win) for name in ('H2', 'H4') for win in (3, 5)]
    stack = np.stack([smooth(data[name], win) for name, win in keys], axis=1)
    # subjects x 4 metrics x SOA x time
    n, m, _, T = stack.shape
    valid = np.isfinite(stack).all(axis=(0, 2))  # metrics x time; edge samples excluded
    dif = stack[:, :, 1] - stack[:, :, 0]
    avg = stack.mean(axis=2)
    outcomes = []
    for fam in ('semantic_mean', 'soa2_minus_soa1', 'GJ_minus_GA', 'AGE_x_SOA'):
        if fam == 'semantic_mean':
            X = avg
            obs = t_one(X)
            effect = X.mean(axis=0)
        elif fam == 'soa2_minus_soa1':
            X = dif
            obs = t_one(X)
            effect = X.mean(axis=0)
        else:
            X = avg if fam == 'GJ_minus_GA' else dif
            gj, ga = X[groups == 'GJ'], X[groups == 'GA']
            if len(gj) < 3 or len(ga) < 3:
                continue
            obs = t_two(gj, ga)
            effect = gj.mean(axis=0) - ga.mean(axis=0)
        obs[~valid] = np.nan
        nullmax = np.zeros(n_perm)
        for p in range(n_perm):
            if fam in ('semantic_mean', 'soa2_minus_soa1'):
                signs = rng.choice((-1, 1), size=n)
                stat = t_one(X * signs[:, None, None])
            else:
                shuffled = rng.permutation(groups)
                stat = t_two(X[shuffled == 'GJ'], X[shuffled == 'GA'])
            nullmax[p] = np.max(np.abs(stat[valid]))
        pvals = (np.sum(nullmax[:, None, None] >= np.abs(obs)[None, ...], axis=0) + 1) / (n_perm + 1)
        for (name, win), eff, pv, tv in zip(keys, effect, pvals, obs):
            for k in np.flatnonzero(np.isfinite(tv)):
                outcomes.append(dict(family=fam, metric=name, samples=win,
                                     time_ms=times[k], effect=eff[k], t_stat=tv[k],
                                     p_maxT_two_metrics_two_windows_all_times=pv[k],
                                     n_gj=np.sum(groups == 'GJ'), n_ga=np.sum(groups == 'GA')))
    return pd.DataFrame(outcomes)


def best_shift(young, old, times_ms, analysis_bounds, max_lag_ms, min_overlap):
    """Search rho(GJ(t), GA(t+lag)) on the SAME absolute time grid.
    A positive lag is interpreted as GA later than GJ, not a neural onset.
    """
    dt = np.median(np.diff(times_ms))
    if not np.allclose(np.diff(times_ms), dt, atol=1e-4):
        raise ValueError('Time axis must be evenly spaced')
    region = (times_ms >= analysis_bounds[0]) & (times_ms <= analysis_bounds[1])
    idx = np.flatnonzero(region)
    if len(idx) < min_overlap:
        raise ValueError('Analysis interval contains too few samples')
    nlag = int(round(max_lag_ms / dt))
    candidates = []
    for shift in range(-nlag, nlag + 1):
        i = idx[(idx + shift >= 0) & (idx + shift < len(times_ms))]
        j = i + shift
        if len(i) < min_overlap:
            continue
        x, y = young[i], old[j]
        keep = np.isfinite(x) & np.isfinite(y)
        x, y = x[keep], y[keep]
        if len(x) < min_overlap:
            continue
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            continue
        rho = float(np.corrcoef(x, y)[0, 1])
        candidates.append((shift * dt, rho, len(x)))
    if not candidates:
        return np.nan, np.nan, np.nan, pd.DataFrame(columns=['lag_ms', 'r', 'n_overlap'])
    vals = pd.DataFrame(candidates, columns=['lag_ms', 'r', 'n_overlap'])
    best = vals.iloc[int(vals.r.argmax())]
    zero = vals.iloc[int(np.abs(vals.lag_ms).argmin())]
    return float(best.lag_ms), float(best.r), float(zero.r), vals


def plot_curves(data, groups, times, out):
    fig, ax = plt.subplots(2, 2, figsize=(14, 9), sharex=True, constrained_layout=True)
    for i, name in enumerate(('H2', 'H4')):
        for j, soa in enumerate((0, 1)):
            a = ax[i, j]
            for group, color in [('GJ', 'tab:blue'), ('GA', 'tab:orange')]:
                for win, style in [(1, ':'), (3, '-'), (5, '--')]:
                    sub = smooth(data[name][groups == group], win)[:, soa]
                    mu = np.divide(np.nansum(sub, axis=0), np.isfinite(sub).sum(axis=0),
                                   out=np.full(sub.shape[-1], np.nan), where=np.isfinite(sub).sum(axis=0)>0)
                    a.plot(times, mu, linestyle=style, color=color, alpha=0.6 if win == 1 else 0.9,
                           linewidth=1.1 if win == 1 else 1.7, label=f'{group} {win} pt')
            a.axhline(0, c='grey', lw=.75)
            a.set_title(f'{name} / SOA code {soa + 1}')
            a.set_ylabel('Crossvalidated contrast / rank')
            a.legend(ncol=2, fontsize=8)
    for a in ax[-1]: a.set_xlabel('Time from target (ms)')
    fig.savefig(out / 'H2_H4_original_3pt_5pt_curves.png', dpi=170)
    plt.close(fig)


def plot_lag(df, lag_boot, times, yng, old, optimal, output):
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    axs[0].plot(times, yng, label='GJ / SOA code 1')
    axs[0].plot(times, old, label='GA / SOA code 1')
    axs[0].plot(times - optimal, old, '--', label=f'GA shifted earlier by {optimal:.1f} ms')
    axs[0].set_xlim(*cfg.SENS_SHIFT_ANALYSIS_MS)
    axs[0].axhline(0, color='gray', lw=.7)
    axs[0].set_xlabel('Target-relative time (ms)')
    axs[0].set_ylabel('H2 crossvalidated contrast / rank')
    axs[0].legend(fontsize=8)
    axs[1].plot(df.lag_ms, df.r)
    axs[1].axvline(0, c='gray', ls=':')
    axs[1].axvline(optimal, c='black', ls='--', label=f'best lag {optimal:.1f} ms')
    axs[1].set_xlabel('Candidate GA lag relative to GJ (ms)')
    axs[1].set_ylabel('Correlation of group mean H2 curves')
    axs[1].legend(fontsize=8)
    fig.savefig(output / 'H2_GJ_GA_SOA1_lag_correlation.png', dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.hist(lag_boot, bins=np.arange(-cfg.SENS_MAX_LAG_MS - 7.81, cfg.SENS_MAX_LAG_MS + 16, 7.81), edgecolor='white')
    ax.axvline(0, color='gray', lw=1)
    ax.axvline(optimal, color='black', lw=1.4, ls='--')
    ax.set_xlabel('Bootstrap optimal lag (ms), positive = GA later')
    ax.set_ylabel('Subject bootstrap replicates')
    ax.set_title('H2 lag uncertainty; exploratory, not a significance test')
    fig.savefig(output / 'H2_lag_bootstrap_distribution.png', dpi=170)
    plt.close(fig)


def main():
    input_dir = Path(cfg.SENS_INPUT_DIR).expanduser().resolve()
    output = Path(cfg.SENS_OUTPUT_DIR).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    subjects, groups, times, data = load_subject_data(input_dir)
    n_gj, n_ga = np.sum(groups == 'GJ'), np.sum(groups == 'GA')
    print(f'Loaded {len(subjects)} subjects: GJ={n_gj}, GA={n_ga}; timepoints={len(times)}')
    if tuple(cfg.SENS_SMOOTH_SAMPLES) != (3, 5):
        raise ValueError('This analysis specifies jointly corrected 3- and 5-sample windows; set SENS_SMOOTH_SAMPLES=(3,5)')
    plot_curves(data, groups, times, output)
    if len(subjects) >= 5:
        stats = permutation_stats(data, groups, times, cfg.SENS_N_PERM, cfg.SENS_SEED)
        stats.to_csv(output / 'H2_H4_smoothed_maxT_stats.csv', index=False)
        print('Smoothed significance rows:', len(stats), 'minimum corrected p:', stats['p_maxT_two_metrics_two_windows_all_times'].min() if len(stats) else 'none')
    if n_gj >= 3 and n_ga >= 3:
        rng = np.random.default_rng(cfg.SENS_SEED + 17)
        gj = data['H2'][groups == 'GJ', cfg.SENS_SOA_INDEX]
        ga = data['H2'][groups == 'GA', cfg.SENS_SOA_INDEX]
        # Use 3-point smoothing as a fixed lag-estimation choice to reduce single-point noise.
        gj = smooth(gj, 3); ga = smooth(ga, 3)
        mean_gj, mean_ga = gj.mean(axis=0), ga.mean(axis=0)
        lag, rho, rho0, lag_curve = best_shift(mean_gj, mean_ga, times,
             cfg.SENS_SHIFT_ANALYSIS_MS, cfg.SENS_MAX_LAG_MS, cfg.SENS_MIN_OVERLAP)
        lag_curve.to_csv(output / 'H2_SOA1_lag_correlation_grid.csv', index=False)
        boots = []
        for _ in range(cfg.SENS_N_BOOT):
            yg = gj[rng.integers(0, len(gj), size=len(gj))].mean(axis=0)
            og = ga[rng.integers(0, len(ga), size=len(ga))].mean(axis=0)
            lag_b, _, _, _ = best_shift(yg, og, times,
                cfg.SENS_SHIFT_ANALYSIS_MS, cfg.SENS_MAX_LAG_MS, cfg.SENS_MIN_OVERLAP)
            boots.append(lag_b)
        boots = np.asarray(boots)
        np.savetxt(output / 'H2_SOA1_bootstrap_lags_ms.csv', boots, delimiter=',', header='lag_ms', comments='')
        result = dict(best_lag_ms=lag, best_r=rho, unshifted_r=rho0,
            relative_corr_gain=rho-rho0, bootstrap_ci95_ms=np.percentile(boots, [2.5, 97.5]).tolist(),
            bootstrap_median_ms=float(np.median(boots)), bootstrap_positive_fraction=float(np.mean(boots > 0)),
            window_smoothing_samples=3, analysis_time_range_ms=list(cfg.SENS_SHIFT_ANALYSIS_MS),
            maximum_absolute_lag_ms=cfg.SENS_MAX_LAG_MS,
            interpretation='Exploratory best-match lag; does not test neural onset or compensation. '
              'Scanning lags and selecting a maximum biases observed correlation upward.')
        (output / 'H2_SOA1_shift_summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
        plot_lag(lag_curve, boots, times, mean_gj, mean_ga, lag, output)
        print('H2 exploratory shift:', result)
    else:
        print('Time-shift analysis skipped: need at least 3 participants in both groups')
    (output / 'analysis_settings.json').write_text(json.dumps({
        'source_folder':str(input_dir), 'n_subjects':len(subjects), 'n_GJ':int(n_gj), 'n_GA':int(n_ga),
        'smoothing':[3,5], 'n_perm':cfg.SENS_N_PERM,'n_boot':cfg.SENS_N_BOOT,
        'seed':cfg.SENS_SEED,'soa_index':cfg.SENS_SOA_INDEX,
        'family_definition':'Separate families by semantic/SOA/AGE/AGE_x_SOA; within each family jointly correct H2,H4,3pt,5pt and time',
        'edge_rule':'trim half-window from both epoch ends; no padding',
        'statistical_note':'Post hoc exploratory. No familywise correction across four statistical contrasts.'
    }, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Saved output to', output)

if __name__ == '__main__':
    main()
