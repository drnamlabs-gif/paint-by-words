# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""The competing baseline a reviewer will ask for: retrieval by example.

Before claiming that stating a goal in language is useful, it has to beat the obvious
alternative, which uses no language at all. Point at a handful of wells that already do
what you want, average their learned image representation, and rank every other well by
similarity to that average. Recursion publishes such a representation for this archive
(OpenPhenom), so the comparison is on their own terms.

Two settings, and they are not the same problem.

  re-mining     every candidate has already been imaged. Retrieval by example applies,
                and is a strong baseline.
  prioritising  the candidate has never been imaged, so there is no embedding to compare
                against; only a model that reads chemistry can score it at all.

The goal-conditioned model solves both, because it maps (sentence, molecule, dose) to a
probability without needing the molecule's image. The retrieval baseline solves only the
first. Reporting a single number across both would hide that, so this module reports the
first setting honestly and leaves the second to the chemistry model, where retrieval has
no entry.

There is a deeper asymmetry, and the third arm here makes it visible. Retrieval scores a
test group using measurements taken ON that same group, and the label of that group is a
function of those measurements. So retrieval is DETECTION, bounded by the reliability of
the measurement, while the chemistry model is PREDICTION from structure. The third arm
runs the identical retrieval on the eight engineered features from which the labels are
themselves computed; it reaches about 0.96 and is reported as a **leakage control**, not
as a competitor. Its job is to show how much of any retrieval score is the label being
read back rather than inferred.

Prototypes are built only from TRAINING-scaffold groups and scored on unseen-scaffold
groups, exactly as the model is, so the comparison is like for like.
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import chem, paths
from .evaluate import ACHIEVABLE
from .features import QUERY, _dose
from .instructions import load as load_instructions
from .labels import formula_prob

EMB = 'OpenPhenom_rxrx3_core_embeddings.parquet'
FEWSHOT = 5


def group_embeddings(groups):
    """Median OpenPhenom embedding per (compound, dose), aligned to the groups table."""
    cache = os.path.join(paths.WORK, 'openphenom_groups.npz')
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        if list(z['key']) == list(groups.treatment + '|' + groups.dose.round(4).astype(str)):
            return z['X'], z['plate']

    meta = pd.read_csv(paths.require(paths.METADATA, 'see README'), low_memory=False)
    meta = meta[(meta.perturbation_type == 'COMPOUND') & (meta.well_type_label == QUERY)].copy()
    meta['plate_id'] = meta.experiment_name + '_' + meta.plate.astype(str)
    meta['dose'] = _dose(meta.concentration)

    emb = pd.read_parquet(paths.require(os.path.join(paths.DATA, EMB), 'download RxRx3-core'))
    feat = [c for c in emb.columns if c.startswith('feature_')]
    emb = emb.set_index('well_id')
    meta = meta[meta.well_id.isin(emb.index)].copy()
    E = emb.loc[meta.well_id, feat].to_numpy(np.float32)

    # per-plate robust centring, the analogue of what the engineered features get; without
    # it the baseline is handicapped by batch effects it did not cause
    for _, idx in meta.groupby('plate_id').groups.items():
        rows = meta.index.get_indexer(idx)
        med = np.median(E[rows], axis=0)
        mad = 1.4826 * np.median(np.abs(E[rows] - med), axis=0) + 1e-6
        E[rows] = (E[rows] - med) / mad

    meta['_k'] = meta.treatment + '|' + meta.dose.round(4).astype(str)
    key = (groups.treatment + '|' + groups.dose.round(4).astype(str)).to_numpy()
    pos = {k: i for i, k in enumerate(key)}
    X = np.zeros((len(groups), len(feat)), np.float32)
    seen = np.zeros(len(groups), bool)
    for k, sub in meta.groupby('_k'):
        i = pos.get(k)
        if i is None:
            continue
        X[i] = np.median(E[meta.index.get_indexer(sub.index)], axis=0)
        seen[i] = True
    print(f'{seen.sum()}/{len(groups)} groups have an OpenPhenom embedding')
    np.savez_compressed(cache, X=X, plate=seen, key=key)
    return X, seen


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--definition', default='v3', choices=['v3', 'v2'])
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(a.seed)

    from .features import load_groups
    g = load_groups()
    ins = load_instructions(a.definition)
    truth = np.stack([formula_prob(g, r['lits'], a.definition) for r in ins])
    hard = truth > 0.5

    X, has_emb = group_embeddings(g)
    Xn = unit(X)
    from .features import FEATURES
    C = np.nan_to_num(g[[c for c in FEATURES]].to_numpy(np.float32))
    Cn = unit(C)

    smiles = g.groupby('treatment').SMILES.first().to_dict()
    split = chem.scaffold_split(smiles, seed=0)
    csplit = g.treatment.map(split).to_numpy()
    train = np.where((csplit == 'train') & has_emb)[0]
    test = np.where((csplit == 'test') & has_emb)[0]
    print(f'{len(train)} training and {len(test)} unseen-scaffold groups with embeddings')

    model = pd.read_csv(os.path.join(paths.RESULTS,
                                     f'summary_numerical_window_rich_{a.definition}.csv'))
    model = model[model.pool == 'unseen_comp'].set_index('text')

    rows = []
    for i, r in enumerate(ins):
        if r['split'] not in ACHIEVABLE:
            continue
        known_te = ~np.isnan(truth[i, test])
        y = hard[i, test][known_te]
        if not (0 < y.sum() < len(y)):
            continue
        pos_tr = train[hard[i, train] & ~np.isnan(truth[i, train])]
        if len(pos_tr) < 2:
            continue
        proto = unit(Xn[pos_tr].mean(0))
        s_all = Xn[test][known_te] @ proto
        few = pos_tr if len(pos_tr) <= FEWSHOT else rng.choice(pos_tr, FEWSHOT, replace=False)
        s_few = Xn[test][known_te] @ unit(Xn[few].mean(0))
        # the same retrieval, on the eight engineered features instead of the learned
        # representation: separates "retrieval by example" from "OpenPhenom"
        s_cp = Cn[test][known_te] @ unit(Cn[pos_tr].mean(0))
        m = model.auc_model.get(r['text'], np.nan)
        b = model.auc_toxicity.get(r['text'], np.nan)
        rows.append(dict(split=r['split'], text=r['text'], n_train_pos=len(pos_tr),
                         n_test_pos=int(y.sum()),
                         auc_retrieval=roc_auc_score(y, s_all),
                         auc_retrieval_fewshot=roc_auc_score(y, s_few),
                         auc_retrieval_cellprofiler=roc_auc_score(y, s_cp),
                         auc_model=m, auc_request_blind=b))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(paths.RESULTS, f'baseline_openphenom_{a.definition}.csv'), index=False)

    print()
    print(f'OpenPhenom retrieval vs the goal-conditioned model, definition {a.definition}')
    print('(both scored on the same unseen-scaffold groups and the same held-out requests)')
    print()
    t = R.groupby('split').agg(
        requests=('text', 'size'),
        retrieval_all=('auc_retrieval', 'mean'),
        retrieval_5shot=('auc_retrieval_fewshot', 'mean'),
        retrieval_cellprofiler=('auc_retrieval_cellprofiler', 'mean'),
        model=('auc_model', 'mean'),
        request_blind=('auc_request_blind', 'mean')).round(3)
    print(t.to_string())
    print()
    w = (R.auc_model > R.auc_retrieval).sum()
    print(f'model beats retrieval on {w} of {len(R)} requests '
          f'(median difference {np.median(R.auc_model - R.auc_retrieval):+.3f})')
    print()
    print('Read these columns as answering different questions. Retrieval scores a group')
    print('using measurements taken on that same group, and the label is a function of')
    print('those measurements, so it is detection with a sensor already pointed at the')
    print('answer; the CellProfiler column is the leakage control that shows this, since')
    print('the labels are computed from exactly those features. The model never sees the')
    print('test compound imaged at all. Retrieval cannot score a compound that has not')
    print('been run; the model can.')
    return R


if __name__ == '__main__':
    main()
