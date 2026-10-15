#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-participant time-resolved, crossvalidated EEG RDMs."""
import warnings
import numpy as np
from constant_variable import SEMANTIC, SOA_CODES, EOG_NAMES, MASTOIDS

def pool_residuals(x, labels, folds, soa):
    """Within-(semantic,fold) trial residuals, selected for one SOA.

    x is trial x channel array, computed at one time window.
    """
    residual=[]
    for f in (0,1):
        for s in SEMANTIC:
            ii=np.where((folds==f)&(labels==s))[0]
            if len(ii)>=2:
                y=x[ii]
                residual.append(y-y.mean(axis=0,keepdims=True))
    return np.concatenate(residual,axis=0) if residual else np.empty((0,x.shape[1]))

def precision_from_residuals(resid, shrink=.15, tol=1e-7):
    """Noise metric using empirically supported eigenspace + diagonal shrinkage.

    Rank is determined on condition-centered trial noise (independent of effects).
    This is a plug-in precision estimator, not guaranteed finite-sample unbiased.
    """
    n,c=resid.shape
    if n<8: raise ValueError('Too few residual trials for noise covariance')
    cov=(resid.T @ resid)/max(n-1,1)
    evals,basis=np.linalg.eigh((cov+cov.T)/2)
    max_eval=float(max(evals.max(),0))
    keep=(evals>max_eval*tol) & (evals>0)
    if not np.any(keep): raise ValueError('Noise covariance rank zero')
    vals=evals[keep]
    lam=(1-shrink)*vals+shrink*vals.mean()
    basis=basis[:,keep]
    p=(basis/lam)@basis.T
    return p,int(keep.sum()),float(vals.max()/max(vals.min(),np.finfo(float).tiny))

def compute_subject(epochs, info, partitions, args):
    """Return distances [soa,time,6,6], 3 shared-effect metrics and QC."""
    exclude=EOG_NAMES | (set() if args.include_mastoids else MASTOIDS)
    picks=[i for i,name in enumerate(epochs.ch_names) if name.upper() not in exclude]
    eeg_names=[epochs.ch_names[i] for i in picks]
    # The input file supplied to the project has 64 scalp, 2 mastoid, 4 EOG.
    if len(picks)<32: raise ValueError(f'Only {len(picks)} channels after exclusions')
    arr=epochs.get_data(picks=picks,copy=True) * 1e6  # MNE loads volts -> microvolts
    times=epochs.times.copy()
    if len(info)!=len(arr):raise ValueError('Event metadata not aligned with MNE epochs')
    selected=info.semantic.notna().to_numpy()
    arr=arr[selected]
    d=info.loc[selected].reset_index(drop=True)
    labels=d.semantic.to_numpy()
    soa_labels=d.soa.to_numpy()
    fold=np.full(len(d),-1,int)
    fold[d.block.isin(partitions['A']).to_numpy()]=0
    fold[d.block.isin(partitions['B']).to_numpy()]=1
    if (fold==-1).any():raise ValueError('Trials with unassigned block')
    win=int(args.window_samples)
    if win<1 or win>len(times):raise ValueError('Invalid time window')
    starts=list(range(0,len(times)-win+1,int(args.step_samples)))
    # Use centered timestamps, no pad: no artificial boundary samples.
    t=np.array([times[k:k+win].mean() for k in starts])
    nsoa=2
    cross=np.full((nsoa,len(t),6,6),np.nan)
    cveu=np.full_like(cross,np.nan)
    corr=np.full_like(cross,np.nan)
    ranks=np.full((nsoa,len(t)),np.nan)
    nonadd_whitened=np.full((nsoa,len(t)),np.nan)
    cond_counts=np.zeros((nsoa,2,6),int)
    for oi,soa in enumerate(SOA_CODES):
        ind=np.where(soa_labels==soa)[0]
        y=arr[ind]; lab=labels[ind]; f=fold[ind]
        for fi in (0,1):
            for si,s in enumerate(SEMANTIC):
                cond_counts[oi,fi,si]=int(np.sum((f==fi)&(lab==s)))
        if (cond_counts[oi]<args.min_per_cell).any():
            warnings.warn(f'SOA {soa} too sparse; cross distances left missing')
            continue
        for ti,start in enumerate(starts):
            chunk=y[:,:,start:start+win].mean(axis=-1)
            # Fold-specific condition mean patterns
            means=np.array([[chunk[(f==fi)&(lab==s)].mean(axis=0)
                            for s in SEMANTIC] for fi in (0,1)])
            resid=pool_residuals(chunk,lab,f,soa)
            try:
                precision,rank,cond=precision_from_residuals(
                    resid,shrink=args.shrink, tol=args.rank_tol)
            except ValueError:
                continue
            ranks[oi,ti]=rank
            # H4: signed cross-partition inner product of the interaction vector.
            # Uses the same rank-regularized noise precision as the RDM.
            # A,C,AC,U = semantic indices 2,3,4,5.
            ga=means[0,4]-means[0,2]-means[0,3]+means[0,5]
            gb=means[1,4]-means[1,2]-means[1,3]+means[1,5]
            nonadd_whitened[oi,ti]=ga @ precision @ gb / rank
            diffA=means[0,:,None,:]-means[0,None,:,:]
            diffB=means[1,:,None,:]-means[1,None,:,:]
            cross[oi,ti]=np.einsum('ijc,cd,ijd->ij',diffA,precision,diffB)/rank
            # Crossvalidated squared Euclidean with matched partitions.
            cveu[oi,ti]=np.einsum('ijc,ijc->ij',diffA,diffB)/len(picks)
            # Correlation distance of pooled condition means (not crossvalidated).
            pooled=np.array([chunk[lab==s].mean(axis=0) for s in SEMANTIC])
            cc=np.corrcoef(pooled)
            corr[oi,ti]=1-cc
            for dm in (cross[oi,ti],cveu[oi,ti],corr[oi,ti]):
                dm[:]=(dm+dm.T)/2
                np.fill_diagonal(dm,0)
    def q_index(rdm):
        u,pa,pc,a,c,ac = 5,0,1,2,3,4
        qp=(rdm[...,pa,u]+rdm[...,pc,u]-rdm[...,pa,pc])/2
        qa=(rdm[...,a,u]+rdm[...,ac,u]-rdm[...,a,ac])/2
        qc=(rdm[...,c,u]+rdm[...,ac,u]-rdm[...,c,ac])/2
        return np.stack((qp,qa,qc),axis=-1)
    # Multivariate vector-additivity deviation ||AC-A-C+U||^2 is biased upward
    # when estimated from a single noisy mean; use crossvalidated fold vectors.
    add=np.full((2,len(t)),np.nan)
    for oi,soa in enumerate(SOA_CODES):
        if (cond_counts[oi]<args.min_per_cell).any():continue
        idx=np.where(soa_labels==soa)[0]
        y=arr[idx]; lab=labels[idx];f=fold[idx]
        for ti,start in enumerate(starts):
            chunk=y[:,:,start:start+win].mean(axis=-1)
            ms=np.array([[chunk[(f==fi)&(lab==s)].mean(axis=0)
                         for s in SEMANTIC] for fi in (0,1)])
            # AC - A - C + U, unlike a zero null on an ordinary squared norm.
            vec=ms[:,4]-ms[:,2]-ms[:,3]+ms[:,5]
            add[oi,ti]=(vec[0]@vec[1])/len(picks)
    # Full-epoch RDM: aggregate independently computed single-timepoint squared
    # distances across ALL samples (NOT a voltage average or selected window).
    # With window_samples=1 / step_samples=1, every recorded epoch sample contributes.
    full_cross=np.nanmean(cross,axis=1)
    full_cveu=np.nanmean(cveu,axis=1)
    full_corr=np.nanmean(corr,axis=1)
    # Theory-driven Related-only RDM contrasts. Shape SOA x time.
    def dist(a,b): return cross[...,a,b]
    across=(dist(0,2)+dist(0,3)+dist(0,4)+dist(1,2)+dist(1,3)+dist(1,4))/6
    h1=across-dist(0,1)  # Relative PA-PC proximity
    h1_distinct=dist(0,1)  # Relative proximity does not imply identity
    h2=across-(dist(2,3)+dist(2,4)+dist(3,4))/3
    others=((0,1),(0,3),(0,4),(1,3),(1,4),(3,4))
    h3=(dist(2,0)+dist(2,1)+dist(2,3)+dist(2,4))/4 - sum(dist(a,b) for a,b in others)/6
    h4=nonadd_whitened
    return dict(h1=h1,h1_distinct=h1_distinct,h2=h2,h3=h3,h4=h4,
                full_epoch_crossnobis=full_cross,
                full_epoch_cv_squared_euclidean=full_cveu,
                full_epoch_correlation=full_corr,
                times=t, crossnobis=cross, cv_squared_euclidean=cveu,
                correlation=corr, q_crossnobis=q_index(cross),
                q_cveuclidean=q_index(cveu),
                additivity_cv_squared=add,rank=ranks,counts=cond_counts,
                channels=np.array(eeg_names))
