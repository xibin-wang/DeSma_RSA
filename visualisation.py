#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plots and group-level time-resolved statistical summaries.

Updated version:
- Q time series remain pointwise in time.
- Heatmaps now show pair-by-time dynamics across the FULL epoch:
  rows = condition pairs, columns = time points.
- Static 6x6 whole-epoch heatmaps are no longer the default summary.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from constant_variable import SEMANTIC, SOA_CODES


# -----------------------------
# Pair helpers
# -----------------------------
PAIR_INDEX = [(i, j) for i in range(len(SEMANTIC)) for j in range(i+1, len(SEMANTIC))]
PAIR_LABELS = [f'{SEMANTIC[i]}-{SEMANTIC[j]}' for i, j in PAIR_INDEX]
REL_LABELS = SEMANTIC[:5]


def _curve_mean_sem(x):
    """Descriptive mean/SEM across subjects, not multiple-comparison corrected."""
    x = np.asarray(x, dtype=float)
    good = np.isfinite(x).all(axis=1)
    x = x[good]
    if x.ndim == 1:
        x = x[:, None]
    if not len(x):
        return np.full(x.shape[1], np.nan), np.full(x.shape[1], np.nan)
    sem = x.std(axis=0, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.zeros(x.shape[1])
    return x.mean(axis=0), sem


def _pair_by_time_from_rdms(rdms):
    """Convert (..., time, 6, 6) RDMs to (..., pair, time)."""
    arr = np.asarray(rdms)
    pairs = [arr[..., i, j] for i, j in PAIR_INDEX]
    return np.stack(pairs, axis=-2)


def _pair_time_heatmap(ax, mat, times_ms, pair_labels, title, symmetric=False, cmap=None):
    """mat: pair x time"""
    mat = np.asarray(mat)
    if cmap is None:
        cmap = 'RdBu_r' if symmetric else 'viridis'
    if symmetric:
        lim = max(float(np.nanmax(np.abs(mat))), 1e-12)
        im = ax.imshow(mat, aspect='auto', origin='lower', cmap=cmap,
                       vmin=-lim, vmax=lim,
                       extent=[times_ms[0], times_ms[-1], -0.5, len(pair_labels)-0.5])
    else:
        im = ax.imshow(mat, aspect='auto', origin='lower', cmap=cmap,
                       extent=[times_ms[0], times_ms[-1], -0.5, len(pair_labels)-0.5])
    ax.set_yticks(np.arange(len(pair_labels)))
    ax.set_yticklabels(pair_labels)
    ax.set_xlabel('Time relative to target event (ms)')
    ax.set_ylabel('Condition pair')
    ax.set_title(title)
    return im


def _save_pair_heatmap(out_path, mat, times_ms, pair_labels, title, colorbar_label, symmetric=False):
    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    im = _pair_time_heatmap(ax, mat, times_ms, pair_labels, title, symmetric=symmetric)
    fig.colorbar(im, ax=ax, label=colorbar_label)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# -----------------------------
# Group overview
# -----------------------------

def plot_group(all_data, out, method='q_crossnobis'):
    """General group Q curves only (no static 6x6 RDMs)."""
    vals = np.stack([x[method] for x in all_data])  # n x 2 x time x 3
    ts = all_data[0]['times'] * 1e3
    names = ('PA/PC shared', 'A/AC shared', 'C/AC shared')
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
    for j, ax in enumerate(axes):
        for soa in range(2):
            v = vals[:, soa, :, j]
            mean, sem = _curve_mean_sem(v)
            ax.plot(ts, mean, label=f'SOA code {soa+1}')
            ax.fill_between(ts, mean-1.96*sem, mean+1.96*sem, alpha=.15)
        ax.axhline(0, color='gray', linewidth=.8)
        ax.set_ylabel(names[j])
        ax.legend(loc='best')
    axes[-1].set_xlabel('Time relative to event (ms)')
    fig.suptitle(f'Exploratory {method} (uncorrected descriptive 95% CI)')
    fig.savefig(out / f'{method}_timeseries.png', dpi=160)
    plt.close(fig)


# -----------------------------
# Permutation stats
# -----------------------------

def permutation_signflip(data, n_perm=2000, seed=42):
    """Max-|t| sign flip across time; within-subject differences or null=0.

    Corrects temporal multiplicity for ONE predefined metric. Run it separately
    for each of 3 Q metrics and each planned contrast; further correction needed
    across metrics/contrasts, e.g. Holm.
    """
    rng = np.random.default_rng(seed)
    if not np.isfinite(data).all():
        return np.full(data.shape[1], np.nan)
    n, k = data.shape
    if n < 5:
        return np.full(k, np.nan)

    def calc_t(a):
        mu = a.mean(0)
        sd = a.std(0, ddof=1)
        return np.divide(mu, sd/np.sqrt(len(a)), out=np.zeros_like(mu), where=sd > 0)

    obs = np.abs(calc_t(data))
    maximal = np.zeros(n_perm)
    for i in range(n_perm):
        signs = rng.choice([-1, 1], size=n)
        maximal[i] = np.max(np.abs(calc_t(data * signs[:, None])))
    return (np.sum(maximal[:, None] >= obs[None, :], axis=0) + 1) / (n_perm + 1)


def group_analysis(all_data, groups, out, n_perm=2000):
    ts = all_data[0]['times']
    groups = np.array(groups)
    q = np.stack([d['q_crossnobis'] for d in all_data])  # subject x SOA x time x metric
    rows = []
    for j, name in enumerate(('QP', 'QA', 'QC')):
        avg = q[:, :, :, j].mean(axis=1)
        diff = q[:, 1, :, j] - q[:, 0, :, j]
        for contrast, dat in [('semantic_mean', avg), ('soa2_minus_soa1', diff)]:
            if np.isfinite(dat).all():
                pc = permutation_signflip(dat, n_perm=n_perm, seed=42+j)
                for ti, time in enumerate(ts):
                    rows.append(dict(metric=name, contrast=contrast, time_ms=time*1000,
                                     effect=float(np.mean(dat[:, ti])), p_maxT=float(pc[ti]), n=len(dat)))
        valid = np.isfinite(avg).all(axis=1)
        av = avg[valid]
        gr = groups[valid]
        if len(np.unique(gr)) == 2 and np.min(np.bincount(gr)) >= 3:
            rng = np.random.default_rng(173+j)

            def tcurve(lab):
                a = av[lab == 0]
                b = av[lab == 1]
                return ttest_ind(a, b, axis=0, equal_var=False).statistic

            obs = np.abs(np.nan_to_num(tcurve(gr)))
            null = np.zeros(n_perm)
            for b in range(n_perm):
                null[b] = np.max(np.abs(np.nan_to_num(tcurve(rng.permutation(gr)))))
            p = (np.sum(null[:, None] >= obs[None, :], axis=0) + 1) / (n_perm + 1)
            for ti, time in enumerate(ts):
                rows.append(dict(metric=name, contrast='GJ_minus_GA', time_ms=time*1000,
                                 effect=float(av[gr == 0, ti].mean() - av[gr == 1, ti].mean()),
                                 p_maxT=float(p[ti]), n=len(av)))
    stats = pd.DataFrame(rows)
    stats.to_csv(out / 'group_timepoint_stats.csv', index=False)
    stats[stats['contrast'] == 'GJ_minus_GA'].to_csv(out / 'age_timepoint_stats.csv', index=False)
    stats[stats['contrast'] == 'soa2_minus_soa1'].to_csv(out / 'soa_timepoint_stats.csv', index=False)
    stats[stats['contrast'] == 'semantic_mean'].to_csv(out / 'semantic_timepoint_stats.csv', index=False)
    (out / 'STATISTICAL_WARNING.txt').write_text(
        'Max-T permutation p values correct across time ONLY, separately per metric and contrast.\n'
        'These 9 test families (3 Q metrics x Semantic/SOA/AGE) require additional\n'
        'family multiplicity control or prespecified hypothesis hierarchy.\n'
        'Baseline of zero Q is an overlap/dot-product hypothesis, not proof of semantic priming.\n'
        'No AGE x SOA interaction is tested. SOA codes 1/2 not independently confirmed as 150/300 ms.\n'
        'Cluster onset timing is NOT an unbiased onset-latency estimate.\n', encoding='utf-8')


# -----------------------------
# AGE plots
# -----------------------------

def plot_age_analysis(all_data, groups, out):
    """AGE (GJ vs GA) pointwise Q curves and pair-by-time heatmaps."""
    groups = np.asarray(groups)
    q = np.stack([d['q_crossnobis'] for d in all_data])
    rdms = np.stack([d['crossnobis'] for d in all_data])  # N,2,T,6,6
    times_ms = np.asarray(all_data[0]['times']) * 1000
    if len(np.unique(groups)) < 2:
        print('AGE plots skipped: only one group represented')
        return
    metrics = ('QP (PA-PC)', 'QA (A-AC)', 'QC (C-AC)')
    data = q.mean(axis=1)  # subject-level SOA average

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, constrained_layout=True)
    for j, ax in enumerate(axes):
        for grp, label in ((0, 'GJ'), (1, 'GA')):
            vals = data[groups == grp, :, j]
            mean, sem = _curve_mean_sem(vals)
            ax.plot(times_ms, mean, label=f'{label} (n={np.isfinite(vals).all(axis=1).sum()})')
            ax.fill_between(times_ms, mean-sem, mean+sem, alpha=.2)
        ax.axhline(0, linewidth=.8, color='gray')
        ax.set_ylabel(metrics[j]); ax.legend(loc='best')
    axes[-1].set_xlabel('Time relative to target event (ms)')
    fig.suptitle('AGE: GJ vs GA (SOA averaged; shading = descriptive +/- 1 SEM)')
    fig.savefig(out / 'age_GJ_vs_GA_Q_timeseries.png', dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, constrained_layout=True)
    for j, ax in enumerate(axes):
        young = data[groups == 0, :, j]
        aged = data[groups == 1, :, j]
        young = young[np.isfinite(young).all(axis=1)]
        aged = aged[np.isfinite(aged).all(axis=1)]
        if not len(young) or not len(aged):
            continue
        delta = young.mean(axis=0) - aged.mean(axis=0)
        se1 = young.var(axis=0, ddof=1) / len(young) if len(young) > 1 else np.full(len(times_ms), np.nan)
        se2 = aged.var(axis=0, ddof=1) / len(aged) if len(aged) > 1 else np.full(len(times_ms), np.nan)
        se = np.sqrt(se1 + se2)
        ax.plot(times_ms, delta, label='GJ - GA')
        ax.fill_between(times_ms, delta-se, delta+se, alpha=.2)
        ax.axhline(0, color='gray', linewidth=.8)
        ax.set_ylabel(metrics[j]); ax.legend(loc='best')
    axes[-1].set_xlabel('Time relative to target event (ms)')
    fig.suptitle('AGE difference (GJ - GA; shading = descriptive +/- 1 SEM)')
    fig.savefig(out / 'age_GJ_minus_GA_Q_difference.png', dpi=170)
    plt.close(fig)

    mean_soa = rdms.mean(axis=1)  # N,T,6,6
    pair_time = _pair_by_time_from_rdms(mean_soa)  # N,15,T
    gj = np.nanmean(pair_time[groups == 0], axis=0) if (groups == 0).any() else None
    ga = np.nanmean(pair_time[groups == 1], axis=0) if (groups == 1).any() else None
    if gj is not None:
        _save_pair_heatmap(out / 'age_pair_by_time_GJ_heatmap.png', gj, times_ms, PAIR_LABELS,
                           'AGE: GJ mean pair-by-time RDM (SOA averaged)', 'Crossnobis distance', symmetric=False)
    if ga is not None:
        _save_pair_heatmap(out / 'age_pair_by_time_GA_heatmap.png', ga, times_ms, PAIR_LABELS,
                           'AGE: GA mean pair-by-time RDM (SOA averaged)', 'Crossnobis distance', symmetric=False)
    if gj is not None and ga is not None:
        _save_pair_heatmap(out / 'age_pair_by_time_difference_heatmap.png', gj-ga, times_ms, PAIR_LABELS,
                           'AGE: GJ - GA pair-by-time difference', 'Crossnobis difference', symmetric=True)


# -----------------------------
# SOA plots
# -----------------------------

def plot_soa_analysis(all_data, out):
    """SOA-specific Q curves, paired difference, and pair-by-time heatmaps."""
    q = np.stack([d['q_crossnobis'] for d in all_data])  # N,2,T,3
    rdms = np.stack([d['crossnobis'] for d in all_data])  # N,2,T,6,6
    ts = np.asarray(all_data[0]['times']) * 1000
    names = ('QP: PA-PC shared', 'QA: A-AC shared', 'QC: C-AC shared')

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, constrained_layout=True)
    for j, ax in enumerate(axes):
        for oi, soa in enumerate(SOA_CODES):
            mu, se = _curve_mean_sem(q[:, oi, :, j])
            ax.plot(ts, mu, label=f'SOA code {soa}')
            ax.fill_between(ts, mu-se, mu+se, alpha=.18)
        ax.axhline(0, color='gray', lw=.8)
        ax.set_ylabel(names[j]); ax.legend(loc='best')
    axes[-1].set_xlabel('Time relative to epoch event (ms)')
    fig.suptitle('SOA-specific RSA Q metrics at every EEG sample')
    fig.savefig(out / 'soa_Q_timeseries.png', dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, constrained_layout=True)
    for j, ax in enumerate(axes):
        dif = q[:, 1, :, j] - q[:, 0, :, j]
        mu, se = _curve_mean_sem(dif)
        ax.plot(ts, mu, label='SOA2 - SOA1')
        ax.fill_between(ts, mu-se, mu+se, alpha=.18)
        ax.axhline(0, color='gray', lw=.8)
        ax.set_ylabel(names[j]); ax.legend(loc='best')
    axes[-1].set_xlabel('Time relative to epoch event (ms)')
    fig.suptitle('Paired SOA Q differences (shading = +/- 1 SEM)')
    fig.savefig(out / 'soa_Q_difference.png', dpi=170)
    plt.close(fig)

    pair_time = _pair_by_time_from_rdms(rdms)  # N,2,15,T
    soa1 = np.nanmean(pair_time[:, 0], axis=0)
    soa2 = np.nanmean(pair_time[:, 1], axis=0)
    diff = np.nanmean(pair_time[:, 1] - pair_time[:, 0], axis=0)
    _save_pair_heatmap(out / 'soa_pair_by_time_soa1_heatmap.png', soa1, ts, PAIR_LABELS,
                       'SOA 1 pair-by-time RDM', 'Crossnobis distance', symmetric=False)
    _save_pair_heatmap(out / 'soa_pair_by_time_soa2_heatmap.png', soa2, ts, PAIR_LABELS,
                       'SOA 2 pair-by-time RDM', 'Crossnobis distance', symmetric=False)
    _save_pair_heatmap(out / 'soa_pair_by_time_difference_heatmap.png', diff, ts, PAIR_LABELS,
                       'SOA 2 - SOA 1 pair-by-time difference', 'Crossnobis difference', symmetric=True)
    # Keep old file name for backward compatibility.
    _save_pair_heatmap(out / 'soa_RDM_differences.png', diff, ts, PAIR_LABELS,
                       'SOA 2 - SOA 1 pair-by-time difference', 'Crossnobis difference', symmetric=True)


# -----------------------------
# Semantic plots
# -----------------------------

def plot_semantic_analysis(all_data, out):
    """Semantic relationship contrasts, SOA/subject averaged.

    Semantic has 6 categories, so there is no single group subtraction.
    We therefore plot:
    1) D(relation, U) curves and relation-specific deviations.
    2) Full pair-by-time heatmap across all 15 condition pairs.
    3) A 5xT heatmap of D(relation,U)-mean_related D(.,U), clearly labeled as NOT a full RDM.
    """
    rdms = np.stack([d['crossnobis'] for d in all_data])  # N,2,T,6,6
    ts = np.asarray(all_data[0]['times']) * 1000
    mean_soa = rdms.mean(axis=1)  # N,T,6,6
    d_u = mean_soa[:, :, :5, 5]  # N,T,5

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for j, name in enumerate(REL_LABELS):
        mu, se = _curve_mean_sem(d_u[:, :, j])
        ax.plot(ts, mu, label=f'D({name}, U)')
        ax.fill_between(ts, mu-se, mu+se, alpha=.12)
    ax.axhline(0, color='grey', lw=.8)
    ax.set_xlabel('Time relative to epoch event (ms)')
    ax.set_ylabel('Crossnobis squared distance')
    ax.set_title('SEMANTIC: each related condition vs unrelated U (SOA averaged)')
    ax.legend(ncol=3)
    fig.savefig(out / 'semantic_relation_timeseries.png', dpi=170)
    plt.close(fig)

    deviations = d_u - d_u.mean(axis=-1, keepdims=True)
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for j, name in enumerate(REL_LABELS):
        mu, se = _curve_mean_sem(deviations[:, :, j])
        ax.plot(ts, mu, label=f'D({name},U) - mean related D(.,U)')
        ax.fill_between(ts, mu-se, mu+se, alpha=.12)
    ax.axhline(0, color='grey', lw=.8)
    ax.set_xlabel('Time relative to epoch event (ms)')
    ax.set_ylabel('Distance difference')
    ax.set_title('SEMANTIC: relation-specific deviations from related-condition mean')
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(out / 'semantic_relation_difference_timeseries.png', dpi=170)
    plt.close(fig)

    # Full pair-by-time heatmap over all 15 pairs.
    pair_time = _pair_by_time_from_rdms(mean_soa)  # N,15,T
    semantic_mean = np.nanmean(pair_time, axis=0)
    _save_pair_heatmap(out / 'semantic_pair_by_time_heatmap.png', semantic_mean, ts, PAIR_LABELS,
                       'SEMANTIC: pair-by-time RDM (SOA averaged)', 'Crossnobis distance', symmetric=False)
    # Keep previous file name for backward compatibility.
    _save_pair_heatmap(out / 'semantic_RDM_heatmaps.png', semantic_mean, ts, PAIR_LABELS,
                       'SEMANTIC: pair-by-time RDM (SOA averaged)', 'Crossnobis distance', symmetric=False)

    # U-relative difference heatmap (5 x T): NOT a full RDM.
    dev_mean = np.nanmean(np.transpose(deviations, (0, 2, 1)), axis=0)  # 5 x T
    _save_pair_heatmap(out / 'semantic_related_vs_U_difference_heatmap.png', dev_mean, ts, REL_LABELS,
                       'SEMANTIC: D(relation,U) - mean related D(.,U) (NOT full RDM)',
                       'Distance difference', symmetric=True)
    # Keep previous file name for backward compatibility.
    _save_pair_heatmap(out / 'semantic_U_relative_difference_heatmaps.png', dev_mean, ts, REL_LABELS,
                       'SEMANTIC: D(relation,U) - mean related D(.,U) (NOT full RDM)',
                       'Distance difference', symmetric=True)


# -----------------------------
# Semantic pairwise stats
# -----------------------------

def semantic_pairwise_statistics(all_data, out, n_perm=2000):
    """Exploratory paired contrasts for 10 distinct D(relation,U) curves.

    maxT corrects time per contrast, not the 10 contrast families.
    """
    rdms = np.stack([d['crossnobis'] for d in all_data]).mean(axis=1)
    d = rdms[:, :, :5, 5]
    ts = np.asarray(all_data[0]['times']) * 1000
    rows = []
    for i in range(5):
        for j in range(i+1, 5):
            dif = d[:, :, i] - d[:, :, j]
            pv = permutation_signflip(dif, n_perm=n_perm, seed=233+i*5+j)
            for ti, time in enumerate(ts):
                rows.append({'relation_a': SEMANTIC[i], 'relation_b': SEMANTIC[j],
                             'time_ms': time, 'effect': float(np.nanmean(dif[:, ti])),
                             'p_maxT_time_only': float(pv[ti]), 'n': len(dif)})
    pd.DataFrame(rows).to_csv(out / 'semantic_pairwise_U_distance_stats.csv', index=False)


# ================================================================
# RELATED-ONLY PRIMARY ANALYSIS: PA, PC, A, C, AC (exclude U)
# ================================================================
RELATED_PAIR_INDEX = [(i,j) for i in range(5) for j in range(i+1,5)]
RELATED_PAIR_LABELS = [f"{SEMANTIC[i]}-{SEMANTIC[j]}" for i,j in RELATED_PAIR_INDEX]


def related_pair_data(all_data):
    """Return N x SOA x related-pair x T: no U-based comparison."""
    r = np.stack([d['crossnobis'] for d in all_data])  # N,2,T,6,6
    return np.stack([r[...,i,j] for i,j in RELATED_PAIR_INDEX], axis=-2)


def _related_curves(data, times_ms, outpath, title, ylabel):
    # data N x 10 x T
    fig, ax = plt.subplots(figsize=(12,7), constrained_layout=True)
    for pi, name in enumerate(RELATED_PAIR_LABELS):
        mu, se = _curve_mean_sem(data[:,pi,:])
        ax.plot(times_ms,mu,label=name,lw=1.3)
        ax.fill_between(times_ms,mu-se,mu+se,alpha=.06)
    ax.axhline(0,color='gray',lw=.7)
    ax.set(xlabel='Time relative to event (ms)',ylabel=ylabel,title=title)
    ax.legend(ncol=5,fontsize=8)
    fig.savefig(outpath,dpi=180)
    plt.close(fig)


def plot_related_primary(all_data,groups,out):
    """Main semantic, SOA, AGE plots based ONLY on 10 Related-Related pairs."""
    arr=related_pair_data(all_data)  # N,2,10,T
    groups=np.asarray(groups)
    ts=all_data[0]['times']*1000
    # Semantic: mean the two SOAs within each subject; group average
    subj=arr.mean(axis=1)
    _related_curves(subj,ts,out/'semantic_related_pair_timeseries.png',
                    'Semantic: 10 Related-Related crossnobis distances', 'Crossnobis squared distance')
    _save_pair_heatmap(out/'semantic_related_pair_by_time_heatmap.png',subj.mean(axis=0),ts,
                       RELATED_PAIR_LABELS,'Semantic: Related-only pair-by-time RDM','Crossnobis distance')
    # Semantic relative pair contrast: deviations from mean across the 10 pair distances
    centered=subj-subj.mean(axis=1,keepdims=True)
    _related_curves(centered,ts,out/'semantic_related_pair_difference_timeseries.png',
                    'Related pair distance minus average of 10 Related pairs','Relative distance')
    _save_pair_heatmap(out/'semantic_related_pair_relative_heatmap.png',centered.mean(axis=0),ts,
                       RELATED_PAIR_LABELS,'Related pair distance minus Related-pair mean (NOT an RDM)',
                       'Relative distance',symmetric=True)
    # SOA: within-participant paired difference
    soa_diff=arr[:,1]-arr[:,0]
    _related_curves(soa_diff,ts,out/'soa_related_pair_difference_timeseries.png',
                    'SOA 2 - SOA 1: Related-only pair distances','Crossnobis difference')
    for oi in range(2):
        _save_pair_heatmap(out/f'soa_related_pairs_soa{oi+1}_heatmap.png',arr[:,oi].mean(axis=0),ts,
                           RELATED_PAIR_LABELS,f'SOA {oi+1}: Related-only pair-by-time','Crossnobis distance')
    _save_pair_heatmap(out/'soa_related_pair_difference_heatmap.png',soa_diff.mean(axis=0),ts,
                       RELATED_PAIR_LABELS,'SOA 2 - SOA 1: Related-only pairs','Crossnobis difference',symmetric=True)
    # AGE: SOA averaged per participant, then unpaired contrast
    if (groups==0).any() and (groups==1).any():
        gj=subj[groups==0];ga=subj[groups==1]
        _related_curves(gj,ts,out/'age_related_pair_GJ_timeseries.png','AGE GJ: Related-only pairs','Crossnobis distance')
        _related_curves(ga,ts,out/'age_related_pair_GA_timeseries.png','AGE GA: Related-only pairs','Crossnobis distance')
        _save_pair_heatmap(out/'age_related_pair_GJ_heatmap.png',gj.mean(axis=0),ts,RELATED_PAIR_LABELS,
                           'AGE GJ: Related-only pairs','Crossnobis distance')
        _save_pair_heatmap(out/'age_related_pair_GA_heatmap.png',ga.mean(axis=0),ts,RELATED_PAIR_LABELS,
                           'AGE GA: Related-only pairs','Crossnobis distance')
        _save_pair_heatmap(out/'age_related_pair_difference_heatmap.png',gj.mean(axis=0)-ga.mean(axis=0),ts,
                           RELATED_PAIR_LABELS,'AGE GJ - GA: Related-only pair distances',
                           'Crossnobis difference',symmetric=True)
    # Also export numerical arrays for exact downstream tests
    np.savez_compressed(out/'related_pair_by_time.npz',
                        times=all_data[0]['times'],pairs=np.array(RELATED_PAIR_LABELS),
                        distances=arr,groups=groups)


def related_pair_statistics(all_data,groups,out,n_perm=2000):
    """Joint max-|t| correction over ALL 10 pairs x ALL time samples per family.

    Families: positive distance vs 0 (semantic), paired SOA effect,
    and between-participant AGE effect. No AGE x SOA.
    These 3 families still require family-wise interpretation/correction.
    """
    arr=related_pair_data(all_data)
    groups=np.asarray(groups)
    subj=arr.mean(axis=1)
    soa=arr[:,1]-arr[:,0]
    ts=np.asarray(all_data[0]['times'])*1000
    rng=np.random.default_rng(12345)
    rows=[]
    for family,dat in [('semantic_distance_vs_zero',subj),('soa2_minus_soa1',soa)]:
        if len(dat)<5 or not np.isfinite(dat).all():
            continue
        n=dat.shape[0]
        def tcurve(x):
            m=x.mean(axis=0);sd=x.std(axis=0,ddof=1)
            return np.divide(m,sd/np.sqrt(x.shape[0]),out=np.zeros_like(m),where=sd>0)
        observed=tcurve(dat)
        mx=np.empty(n_perm)
        for k in range(n_perm):
            flip=rng.choice((-1,1),size=n)
            mx[k]=np.max(np.abs(tcurve(dat*flip[:,None,None])))
        ps=(1+(mx[:,None,None]>=np.abs(observed)[None,:,:]).sum(axis=0))/(n_perm+1)
        for j,label in enumerate(RELATED_PAIR_LABELS):
            for ti,time in enumerate(ts):
                rows.append(dict(family=family,pair=label,time_ms=time,
                                 effect=float(dat[:,j,ti].mean()),p_maxT_pair_and_time=float(ps[j,ti]),n=n))
    if (groups==0).sum()>=3 and (groups==1).sum()>=3 and np.isfinite(subj).all():
        def welch(labs):
            a=subj[labs==0];b=subj[labs==1]
            va=a.var(0,ddof=1)/len(a);vb=b.var(0,ddof=1)/len(b)
            return np.divide(a.mean(0)-b.mean(0),np.sqrt(va+vb),out=np.zeros_like(va),where=(va+vb)>0)
        observed=welch(groups)
        mx=np.empty(n_perm)
        for k in range(n_perm):
            mx[k]=np.max(np.abs(welch(rng.permutation(groups))))
        ps=(1+(mx[:,None,None]>=np.abs(observed)[None,:,:]).sum(0))/(n_perm+1)
        for j,label in enumerate(RELATED_PAIR_LABELS):
            for ti,time in enumerate(ts):
                rows.append(dict(family='GJ_minus_GA',pair=label,time_ms=time,
                                 effect=float(subj[groups==0,j,ti].mean()-subj[groups==1,j,ti].mean()),
                                 p_maxT_pair_and_time=float(ps[j,ti]),n=len(groups)))
    pd.DataFrame(rows,columns=['family','pair','time_ms','effect','p_maxT_pair_and_time','n']).to_csv(
        out/'related_pair_timepoint_stats.csv',index=False)


# ===== H1-H4 THEORY-DRIVEN RDM/INTERACTION INFERENCE =====
import constant_variable as cfg
THEORY_METRICS=('h1','h1_distinct','h2','h3','h4')
METRICS=list(THEORY_METRICS)
TITLES={'h1_distinct':'H1b: PA-PC discriminability', 'h1':'H1a: PA-PC relative proximity','h2':'H2: A/C/AC late organization','h3':'H3: A-specific representational organization','h4':'H4: AC-A-C+U crossvalidated interaction'}
PAIRS=[(i,j) for i in range(5) for j in range(i+1,5)]
LABELS=[f'{SEMANTIC[i]}-{SEMANTIC[j]}' for i,j in PAIRS]

def stat_t(X):
    mean=X.mean(axis=0); sd=X.std(axis=0,ddof=1)
    return np.divide(mean,sd/np.sqrt(len(X)),out=np.zeros_like(mean),where=sd>0)


def maxT_sign(X,n_perm,rng):
    # X: participants x metrics x time
    n,m,t=X.shape
    if n<5:return np.full((m,t),np.nan)
    obs=np.abs(stat_t(X))
    mx=np.empty(n_perm)
    for b in range(n_perm):
        signs=rng.choice([-1,1],size=n)
        mx[b]=np.max(np.abs(stat_t(X*signs[:,None,None])))
    return ((mx[:,None,None]>=obs).sum(axis=0)+1)/(n_perm+1)


def maxT_group(X,group,n_perm,rng):
    # between independent ages: t with unequal variance
    a=X[group==0];b=X[group==1]
    if min(len(a),len(b))<4:return np.full(X.shape[1:],np.nan)
    def tt(a,b):
        delta=a.mean(0)-b.mean(0)
        se=np.sqrt(a.var(0,ddof=1)/len(a)+b.var(0,ddof=1)/len(b))
        return np.divide(delta,se,out=np.zeros_like(delta),where=se>0)
    obs=np.abs(tt(a,b)); mx=np.empty(n_perm)
    for i in range(n_perm):
        p=rng.permutation(group);mx[i]=np.max(np.abs(tt(X[p==0],X[p==1])))
    return ((mx[:,None,None]>=obs).sum(axis=0)+1)/(n_perm+1)


def plot_theoretical_curves(subject_data,groups,times,out):
    ts=times*1e3
    for name in METRICS:
        v=np.stack([s[name] for s in subject_data]) # N,2,T
        fig,axs=plt.subplots(2,2,figsize=(13,8),sharex=True,constrained_layout=True)
        for ax,grp,so in zip(axs.flat,[0,0,1,1],[0,1,0,1]):
            sub=v[(groups==grp),so]
            if len(sub):
                mean=sub.mean(0);se=sub.std(0,ddof=1)/np.sqrt(len(sub)) if len(sub)>1 else 0
                ax.plot(ts,mean);ax.fill_between(ts,mean-se,mean+se,alpha=.2)
            ax.axhline(0,color='gray',lw=.7)
            ax.set_title(f'{"GJ" if grp==0 else "GA"} / SOA code {so+1} (n={len(sub)})')
            ax.set_ylabel('Crossvalidated contrast / rank')
            ax.set_xlabel('Time (ms)')
        fig.suptitle(TITLES[name]+' (descriptive, ±1 SEM)')
        fig.savefig(out/f'{name}_AGE_by_SOA_timeseries.png',dpi=170);plt.close(fig)
    rdms=np.stack([d['crossnobis'] for d in subject_data]) # N,2,T,6,6
    for description,arr in [('semantic',rdms.mean((0,1))),('soa1',rdms[:,0].mean(0)),('soa2',rdms[:,1].mean(0)),
                            ('soa2_minus_soa1',(rdms[:,1]-rdms[:,0]).mean(0))]:
        a=np.stack([arr[:,i,j] for i,j in PAIRS]);fig,ax=plt.subplots(figsize=(11,6),constrained_layout=True)
        v=max(np.max(np.abs(a)),1e-12);diff='minus' in description
        im=ax.imshow(a,origin='lower',aspect='auto',extent=[ts[0],ts[-1],-.5,9.5],cmap='RdBu_r' if diff else 'viridis',
                     vmin=-v if diff else None,vmax=v if diff else None)
        ax.set_yticks(range(10),LABELS);ax.set_xlabel('Time (ms)');ax.set_title(description+' | Related-only 10 pair RDM')
        fig.colorbar(im,ax=ax,label='Crossnobis' if not diff else 'Distance difference')
        fig.savefig(out/f'RDM_{description}_pair_by_time.png',dpi=170);plt.close(fig)
    if len(set(groups))==2:
        for label,arr in [('GJ',rdms[groups==0].mean((0,1))),('GA',rdms[groups==1].mean((0,1))),
                          ('GJ_minus_GA',rdms[groups==0].mean((0,1))-rdms[groups==1].mean((0,1)))]:
            a=np.stack([arr[:,i,j] for i,j in PAIRS]);fig,ax=plt.subplots(figsize=(11,6),constrained_layout=True)
            diff='minus' in label;v=max(np.max(np.abs(a)),1e-12)
            im=ax.imshow(a,origin='lower',aspect='auto',extent=[ts[0],ts[-1],-.5,9.5],
                         cmap='RdBu_r' if diff else 'viridis',vmin=-v if diff else None,vmax=v if diff else None)
            ax.set_yticks(range(10),LABELS);ax.set_xlabel('Time (ms)');ax.set_title(label+' | Related-only RDM')
            fig.colorbar(im,ax=ax);fig.savefig(out/f'RDM_AGE_{label}_pair_by_time.png',dpi=170);plt.close(fig)


def theoretical_group_stats(data,groups,times,out):
    rng=np.random.default_rng(getattr(cfg, "SEED", 2026))
    arrays=np.stack([np.stack([x[k] for k in METRICS],axis=1) for x in data]) # N,2,5,T
    # H1a / H1b / H2 jointly corrected across metrics and time.
    families=[('H1_H2',(0,1,2)),('H3',(3,)),('H4',(4,))]
    rows=[]
    contrasts={'semantic_mean':(arrays.mean(1),None),
               'soa2_minus_soa1':(arrays[:,1]-arrays[:,0],None),
               'AGE_x_SOA':(arrays[:,0]-arrays[:,1],groups),
               'GJ_minus_GA':(arrays.mean(1),groups)}
    for contrast,(X,group) in contrasts.items():
        for fam,indices in families:
            dat=X[:,indices,:]
            if group is None:
                p=maxT_sign(dat,cfg.N_PERM,rng)
                effect=dat.mean(0)
            elif len(set(group))==2:
                p=maxT_group(dat,group,cfg.N_PERM,rng)
                effect=dat[group==0].mean(0)-dat[group==1].mean(0)
            else:continue
            for k,idx in enumerate(indices):
                for t,ms in enumerate(times*1000):
                    rows.append({'contrast':contrast,'family':fam,'metric':METRICS[idx], 'time_ms':ms,
                                 'effect':effect[k,t], 'p_maxT_within_family_and_time':p[k,t],
                                 'n_GJ':sum(groups==0),'n_GA':sum(groups==1)})
    pd.DataFrame(rows).to_csv(out/'theoretical_contrasts_timepoint_stats.csv',index=False)
