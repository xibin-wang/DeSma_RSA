#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PyCharm entry point: configure all paths and analysis parameters in
constant_variable.py, then click Run on main.py. No CLI arguments needed.
"""
import json
import warnings
from pathlib import Path
from types import SimpleNamespace
import mne
import numpy as np
import pandas as pd
from constant_variable import SEMANTIC, EVENT_TO_PAIR, GROUPS
import constant_variable as cfg
from data_load import eeglab_meta, select_partitions
from rsa import compute_subject
from visualisation import (plot_group, group_analysis, plot_age_analysis,
                           plot_soa_analysis, plot_semantic_analysis, semantic_pairwise_statistics,
                           plot_related_primary, related_pair_statistics,
                            plot_theoretical_curves, theoretical_group_stats)

def main():
    args = SimpleNamespace(
        input_dir=Path(cfg.INPUT_DIR).expanduser().resolve() if cfg.INPUT_DIR is not None else None,
        metadata=Path(cfg.METADATA_FILE).expanduser().resolve() if cfg.METADATA_FILE is not None else None,
        output_dir=Path(cfg.OUTPUT_DIR).expanduser().resolve(),
        counts=Path(cfg.COUNTS_FILE).expanduser().resolve() if cfg.COUNTS_FILE is not None else None,
        block_size=cfg.BLOCK_SIZE,
        window_samples=cfg.WINDOW_SAMPLES,
        step_samples=cfg.STEP_SAMPLES,
        shrink=cfg.SHRINK,
        rank_tol=cfg.RANK_TOL,
        min_per_cell=cfg.MIN_PER_CELL,
        include_mastoids=cfg.INCLUDE_MASTOIDS,
        partition_mode=cfg.PARTITION_MODE,
        n_perm=cfg.N_PERM,
        limit=cfg.LIMIT,
    )
    print(f"Manifest: {args.metadata}; output: {args.output_dir}")
    if args.min_per_cell<1:raise ValueError("MIN_PER_CELL must be >= 1")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.metadata is not None:
        manifest = pd.read_csv(args.metadata, dtype=str)
        mandatory = {'subject', 'path', 'group'}
        if not mandatory.issubset(manifest.columns):
            raise SystemExit(f'--metadata requires columns: {sorted(mandatory)}')
        if manifest[list(mandatory)].isna().any().any():
            raise SystemExit('--metadata has missing required fields')
        if manifest.subject.duplicated().any() or manifest.path.duplicated().any():
            raise SystemExit('--metadata has duplicated subject IDs or paths')
        if not manifest.group.isin(GROUPS).all():
            raise SystemExit(f'Unknown group labels: {manifest.loc[~manifest.group.isin(GROUPS), "group"].unique().tolist()}')
        entries = []
        for row in manifest.itertuples(index=False):
            given = Path(row.path)
            options = [given] if given.is_absolute() else [
                args.metadata.resolve().parent / given,
                Path.cwd() / given,
            ]
            if args.input_dir is not None:
                options.append(args.input_dir / given)
                options.append(args.input_dir / given.name)
            path = next((p.resolve() for p in options if p.is_file()), options[0].resolve())
            entries.append((path, str(row.subject), str(row.group)))
    else:
        if args.input_dir is None:
            raise SystemExit('Provide --metadata meta_data.csv or --input-dir EEG_data')
        entries = [(p, p.stem, p.name[:2]) for p in sorted(args.input_dir.glob('*.set'))]
    if args.limit is not None:
        entries = entries[:args.limit]
    if not entries:
        raise SystemExit('No datasets specified')
    missing = [str(p) for p, _, _ in entries if not p.is_file()]
    if missing:
        raise SystemExit('Missing .set files (first five):\n' + '\n'.join(missing[:5]) +
                         '\nCheck CSV paths relative to the CSV folder, or use --input-dir')
    existing=None
    if args.counts:
        existing=pd.read_csv(args.counts)
        print(f'Loaded count QC for {len(existing)} subjects')
    qc=[]; group_data=[]; group_ids=[]; group_labels=[]
    for path, subject_id, group_name in entries:
        print(f'\nProcessing: {path.name}',flush=True)
        rec=dict(filename=path.name,subject=subject_id,group=group_name,status='error')
        try:
            tab,nstim,ref=eeglab_meta(path,args.block_size)
            rec.update(n_original_stimuli=nstim,reference=ref,
                       n_epochs_total=len(tab),n_target=int(tab.semantic.notna().sum()))
            if existing is not None:
                key=existing['file_path'].astype(str).str.replace('\\','/',regex=False).str.split('/').str[-1]
                cand=existing.loc[key==path.name]
                if len(cand)==1:
                    row=cand.iloc[0]
                    if int(row.n_total)!=rec['n_target']:
                        warnings.warn(f'{path.name}: event-filtered trials {rec["n_target"]} != CSV {row.n_total}')
                        rec['count_csv_match']=False
                    else: rec['count_csv_match']=True
            target=tab[tab.semantic.notna()].copy()
            parts,mode=select_partitions(target,args.partition_mode,args.min_per_cell)
            rec.update(partition_mode=mode,partition_a=','.join(map(str,parts['A'])) if parts else '',
                       partition_b=','.join(map(str,parts['B'])) if parts else '')
            if parts is None:
                rec['status']='no_usable_partition';qc.append(rec)
                print('No complete two-partition split; skipped')
                continue
            epo=mne.read_epochs_eeglab(str(path),verbose='ERROR')
            if len(epo)!=len(tab):raise ValueError('MNE epoch count mismatch')
            if np.max(np.abs(epo.times))>3:raise ValueError('Unexpected time axis')
            result=compute_subject(epo,tab,parts,args)
            result.update(subject=subject_id,group=group_name,partition_mode=mode,
                          partition_a=parts['A'],partition_b=parts['B'])
            np.savez_compressed(args.output_dir/f'{subject_id}_rsa.npz',**result)
            rec.update(status='ok',n_eeg_channels=len(result['channels']),
                       min_cell_partition=int(result['counts'].min()),
                       rank_median=float(np.nanmedian(result['rank'])))
            group_data.append(result);group_ids.append(subject_id);group_labels.append(GROUPS[group_name])
            print(f' OK: targets={rec["n_target"]}, scalp channels={rec["n_eeg_channels"]}, '
                  f'min cell/fold={rec["min_cell_partition"]}, median rank={rec["rank_median"]}')
        except Exception as ex:
            rec['status']=f'ERROR: {type(ex).__name__}: {ex}'
            warnings.warn(f'{path.name}: {rec["status"]}')
        qc.append(rec)
    pd.DataFrame(qc).to_csv(args.output_dir/'quality_control.csv',index=False)
    if not group_data:
        print('No successful subject; see quality_control.csv')
        return
    np.savez_compressed(args.output_dir/'group_data.npz',
                        subjects=np.array(group_ids),groups=np.array(group_labels),
                        times=group_data[0]['times'],
                        full_epoch_crossnobis=np.stack([d['full_epoch_crossnobis'] for d in group_data]),
                        full_epoch_cv_squared_euclidean=np.stack([d['full_epoch_cv_squared_euclidean'] for d in group_data]),
                        full_epoch_correlation=np.stack([d['full_epoch_correlation'] for d in group_data]),
                        crossnobis=np.stack([d['crossnobis'] for d in group_data]),
                        cv_squared_euclidean=np.stack([d['cv_squared_euclidean'] for d in group_data]),
                        correlation=np.stack([d['correlation'] for d in group_data]),
                        q_crossnobis=np.stack([d['q_crossnobis'] for d in group_data]),
                        q_cveuclidean=np.stack([d['q_cveuclidean'] for d in group_data]),
                        additivity_cv_squared=np.stack([d['additivity_cv_squared'] for d in group_data]),
                        **{k: np.stack([d[k] for d in group_data]) for k in ('h1','h1_distinct','h2','h3','h4')})
    # THEORY: specified RDM geometry (H1-H3) + interaction-vector test (H4).
    plot_theoretical_curves(group_data,np.asarray(group_labels),group_data[0]['times'],args.output_dir)
    if len(group_data)>=5:
        theoretical_group_stats(group_data,np.asarray(group_labels),group_data[0]['times'],args.output_dir)
    else:
        print('Theoretical hypothesis p-values omitted: fewer than five subjects')
    # PRIMARY: direct comparisons between the five Related semantic relations.
    # U is excluded from primary plots and tests; original 6x6 RDMs remain saved.
    plot_related_primary(group_data,group_labels,args.output_dir)
    if len(group_data)>=5:
        related_pair_statistics(group_data,group_labels,args.output_dir,args.n_perm)
    else:
        print('Related-only group significance omitted: fewer than 5 participants')
    (args.output_dir/'analysis_config.json').write_text(json.dumps({
        'n_success':len(group_data),'subjects':group_ids,
        'metadata_file':str(args.metadata) if args.metadata else None,'semantic_order':SEMANTIC,'semantic_primary':'10 Related-only condition pairs (PA,PC,A,C,AC), excluding U',
        'event_to_pair':{str(k):v for k,v in EVENT_TO_PAIR.items()},
        'soa_mapping_note':'SOA code 1/2 mapping to ms must be independently confirmed',
        'include_mastoids':args.include_mastoids,
        'baseline':'none; no rereferencing or ICA rerun',
        'window_samples':args.window_samples,'heatmap_summary':'Condition-pair by time heatmaps built from pointwise RDMs across the full epoch (no fixed temporal window).', 'shrink':args.shrink,
        'rank_tol':args.rank_tol,'min_per_cell':args.min_per_cell,
        'partition_mode':args.partition_mode,
        'interpretation':'crossnobis plug-in noise covariance estimator',
    },indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'\nCompleted {len(group_data)}/{len(entries)} datasets. Results: {args.output_dir}')

if __name__ == '__main__':
    main()
