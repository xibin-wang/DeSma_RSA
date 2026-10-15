#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EEGLAB event parsing and blocked partition selection."""
import itertools
import re
import warnings
import numpy as np
import pandas as pd
from scipy.io import loadmat
from constant_variable import SEMANTIC, SOA_CODES, STIM_CODES, EVENT_TO_PAIR

def code_of(x):
    match = re.search(r'(\d+)\s*$', str(x))
    return int(match.group(1)) if match else None

def eeglab_meta(path, block_size=104):
    """Return 1-row-per-epoch metadata, aligned to original epoch order."""
    mat = loadmat(path, squeeze_me=True, struct_as_record=False,
                  variable_names=['event','urevent','trials','ref'])
    uev = np.atleast_1d(mat['urevent']).ravel()
    events = np.atleast_1d(mat['event']).ravel()
    if len(events) != int(mat['trials']):
        raise ValueError(f'{path.name}: event count != epochs; inspect multiple events per epoch')
    stim_index = [i for i,e in enumerate(uev, start=1) if code_of(e.type) in STIM_CODES]
    if len(stim_index) < block_size:
        raise ValueError('No complete 104-stimulus block in urevent; inspect log')
    order_lookup = {u:k for k,u in enumerate(stim_index)}
    rows=[]
    for idx,e in enumerate(events):
        code=code_of(e.type)
        ui=int(np.asarray(e.urevent).item())
        if ui not in order_lookup:
            # An epoch from an event other than the target stimulus; keep positional alignment
            rows.append(dict(epoch=idx,code=code,semantic=None,soa=None,order=np.nan,block=np.nan))
            continue
        ordinal=order_lookup[ui]
        pair=EVENT_TO_PAIR.get(code)
        rows.append(dict(epoch=idx,code=code,
                         semantic=pair[0] if pair else None,
                         soa=pair[1] if pair else None,
                         order=ordinal,block=ordinal//block_size+1))
    tab=pd.DataFrame(rows)
    if tab['order'].notna().sum() != len(events):
        warnings.warn(f'{path.name}: some epochs lack a recognized stimulus marker')
    # Avoid assuming that stimulus order maps onto blocks if urevent was incomplete.
    if len(stim_index)%block_size:
        warnings.warn(f'{path.name}: {len(stim_index)} recovered stimuli; not a multiple of {block_size}. '
                      'Inspect practice trials, partial recordings, and block boundaries.')
    return tab, len(stim_index), str(mat.get('ref','unknown'))

def select_partitions(df, method='odd_even', min_per_cell=1):
    """Pick 2 nonoverlapping groups of COMPLETE original blocks, blind to EEG."""
    blocks=sorted(df.block.dropna().astype(int).unique())
    if len(blocks)<2: return None, 'less_than_two_blocks'
    def eval_combo(a,b):
        out=[]
        for selected in (a,b):
            g=df[df.block.isin(selected)].groupby(['soa','semantic']).size()
            out.append(np.array([g.get((soa,sem),0) for soa in SOA_CODES for sem in SEMANTIC]))
        return np.stack(out)
    odd=set(b for b in blocks if b%2==1)
    even=set(blocks)-odd
    if odd and even:
        counts=eval_combo(odd,even)
        if counts.min()>=min_per_cell:
            return ({'A':sorted(odd),'B':sorted(even)},'odd_even')
    if method == 'strict_odd_even':
        return None,'odd_even_incomplete'
    # Predeclared fallback: maximize smallest cell, then minimize overall cell imbalance.
    # No neural data or p values are considered.
    candidate=[]
    for n in range(1, len(blocks)):
        for part in itertools.combinations(blocks,n):
            if blocks[0] not in part: continue  # symmetry removal
            a=set(part); b=set(blocks)-a
            counts=eval_combo(a,b)
            if counts.min()<min_per_cell: continue
            score=(int(counts.min()),-float(np.sum(np.abs(counts[0]-counts[1]))),
                   -abs(len(a)-len(b)))
            candidate.append((score,sorted(a),sorted(b)))
    if not candidate: return None,'no_complete_split'
    candidate.sort(key=lambda v:v[0],reverse=True)
    _,a,b=candidate[0]
    return {'A':a,'B':b},'fallback_block_split'
