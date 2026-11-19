# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Scoring, with the two trivial baselines printed next to every number.

What is measured, per instruction, over compounds whose scaffold the model never saw:

  AUC    pick one group that satisfies the instruction and one that does not; the
         AUC is the probability the model ranks them the right way round. 0.5 is
         chance regardless of how rare the behaviour is, which is why it is the
         headline rather than accuracy: positives here are 1-5% of groups, so a
         model that answers "no" to everything would score 95% accuracy.
  top-1  did the single highest-ranked (compound, dose) actually satisfy the
         instruction? This is the number a screening user feels, and it is much
         harsher than AUC.
  refusal AUC
         over held-out achievable instructions AND the refuse set, how well does the
         model's own maximum probability separate "something in the archive does
         this" from "nothing does"? A system that cannot say no is not a screen.

The two baselines exist because a high AUC on a phenotype is easy to get for the
wrong reason, and reporting it alone is how a result becomes a lie by omission.

  toxicity-only  take the model's own prediction for "kill most of the cells" and
                 use that one vector as the score for EVERY instruction. It knows
                 nothing about what was asked. Under the v2 definition this reached
                 .845/.874 against the model's own .851/.896, meaning nearly all of
                 the apparent skill was "find the compounds that kill cells".
                 Reproduce that with --definition v2.
  dose-only      score by dose alone: higher dose ranks first.

A model is only interesting where it beats both. Run --definition v2 to see a
version of this work that looked much better and was worth much less.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import paths
from .features import load_groups
from .instructions import load as load_instructions
from .labels import formula_prob

ACHIEVABLE = ['para_test', 'comp_test']
TOXICITY_PROBE = 'kill most of the cells'

# The refusal statistic is the mean score of the shortlist the user is actually
# shown, with SHORTLIST fixed at the demo's default length rather than tuned.
# It is deliberately not the maximum. A maximum is an extreme-value statistic: it
# grows with the number of groups in the pool, so a threshold learned on one pool
# does not transfer to a larger one, and one lucky group is enough to answer a wish
# that nothing satisfies. Measured here, going from 1,256 to 2,670 groups, the
# median maximum for IMPOSSIBLE wishes rose .038 -> .083 while the median for
# achievable ones did not move, and refusal AUC on unseen compounds fell from .86
# (shortlist mean) to .79 (maximum).
SHORTLIST = 10


def _load_preds(labels, out, chem_kind, definition):
    pat = os.path.join(paths.RESULTS, f'pred_{labels}_{out}_{chem_kind}_{definition}_s*.npz')
    files = sorted(glob.glob(pat))
    if not files:
        raise FileNotFoundError(f'no predictions matching {pat}; run python -m pbw.train')
    return [np.load(f) for f in files], files


def evaluate(labels='numerical', out='window', chem_kind='rich', definition='v3'):
    groups = load_groups()
    instructions = load_instructions(definition)
    truth = np.stack([formula_prob(groups, r['lits'], definition) for r in instructions])
    hard = truth > 0.5
    Z, files = _load_preds(labels, out, chem_kind, definition)

    tox_i = next(i for i, r in enumerate(instructions) if r['text'] == TOXICITY_PROBE)
    ld = np.log10(groups.dose.to_numpy(float))

    rows = []
    for pool in ('seen_comp', 'val_comp', 'unseen_comp'):
        if pool not in Z[0]:
            continue
        P = np.mean([z[pool].astype(np.float32) for z in Z], 0)
        gidx = Z[0][pool + '_idx']
        for i, r in enumerate(instructions):
            known = ~np.isnan(truth[i, gidx])
            y = hard[i, gidx][known]
            n_pos = int(y.sum())
            scores = dict(model=P[i][known], toxicity=P[tox_i][known], dose=ld[gidx][known])
            rec = dict(pool=pool, split=r['split'], concept=r['concept'], text=r['text'],
                       n_pos=n_pos, n=int(known.sum()))
            for name, s in scores.items():
                rec[f'auc_{name}'] = roc_auc_score(y, s) if 0 < n_pos < len(y) else np.nan
                rec[f'top1_{name}'] = bool(y[int(np.argmax(s))]) if len(y) else False
            top = np.sort(scores['model'])[::-1][:SHORTLIST]
            rec['shortlist_p'] = float(top.mean()) if len(top) else np.nan
            rec['max_p'] = float(scores['model'].max()) if len(y) else np.nan
            rows.append(rec)
    R = pd.DataFrame(rows)
    R.attrs['n_seeds'] = len(files)
    return R


def summarise(R):
    out = []
    for pool in ('unseen_comp', 'seen_comp'):
        for split in ACHIEVABLE:
            x = R[(R.pool == pool) & (R.split == split) & (R.n_pos > 0)]
            if not len(x):
                continue
            out.append(dict(
                compounds='unseen scaffold' if pool == 'unseen_comp' else 'seen',
                instructions='unseen phrasing' if split == 'para_test' else 'unseen combination',
                n=len(x),
                auc=round(x.auc_model.mean(), 3),
                auc_toxicity_only=round(x.auc_toxicity.mean(), 3),
                auc_dose_only=round(x.auc_dose.mean(), 3),
                margin=round(x.auc_model.mean() - x.auc_toxicity.mean(), 3),
                top1=round(x.top1_model.mean(), 3),
                top1_toxicity_only=round(x.top1_toxicity.mean(), 3)))
    return pd.DataFrame(out)


def _refusal_pool(R, pool, statistic='shortlist_p'):
    x = R[(R.pool == pool) & ((R.split == 'refuse')
                              | (R.split.isin(ACHIEVABLE) & (R.n_pos > 0)))]
    return (x.split != 'refuse').to_numpy(), x[statistic].to_numpy()


def refusal(R, pool='unseen_comp', calibrate_on='val_comp'):
    """Can the model's own confidence tell achievable from impossible?

    The threshold is chosen on VALIDATION compounds: held out from training, and
    not the test pool. It must not be chosen on train compounds, where the model
    fits closely and its confidence is systematically higher than it will be on new
    chemistry; that threshold comes out far too strict and transfers badly.
    """
    achievable, p_test = _refusal_pool(R, pool)
    auc = roc_auc_score(achievable, p_test)
    auc_max = roc_auc_score(achievable, _refusal_pool(R, pool, 'max_p')[1])
    if calibrate_on not in set(R.pool):
        calibrate_on = 'seen_comp'
    ok, p = _refusal_pool(R, calibrate_on)
    best = max((np.mean(p[ok] >= c) - np.mean(p[~ok] >= c), c) for c in np.unique(p))[1]
    return dict(auc=round(auc, 3), auc_if_max_were_used=round(auc_max, 3),
                threshold=round(float(best), 4), calibrated_on=calibrate_on,
                n_achievable=int(achievable.sum()), n_refuse=int((~achievable).sum()),
                answers_answerable=round(float(np.mean(p_test[achievable] >= best)), 3),
                answers_impossible=round(float(np.mean(p_test[~achievable] >= best)), 3))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--labels', default='numerical', choices=['numerical', 'critic', 'numsub'])
    ap.add_argument('--out', default='window', choices=['window', 'beta'])
    ap.add_argument('--chem', default='rich', choices=['rich', 'morgan'])
    ap.add_argument('--definition', default='v3', choices=['v3', 'v2'])
    ap.add_argument('--per-instruction', action='store_true')
    a = ap.parse_args(argv)

    R = evaluate(a.labels, a.out, a.chem, a.definition)
    tag = f'{a.labels}_{a.out}_{a.chem}_{a.definition}'
    R.to_csv(os.path.join(paths.RESULTS, f'summary_{tag}.csv'), index=False)

    print(f'labels={a.labels}  output={a.out}  chemistry={a.chem}  '
          f'definition={a.definition}  seeds={R.attrs["n_seeds"]}')
    print()
    print(summarise(R).to_string(index=False))
    print()
    ref = refusal(R)
    print(f'refusal on unseen compounds: AUC {ref["auc"]}  '
          f'({ref["n_achievable"]} achievable vs {ref["n_refuse"]} impossible; '
          f'{ref["auc_if_max_were_used"]} if the single best score were used instead '
          f'of the top-{SHORTLIST} mean)')
    print(f'  threshold {ref["threshold"]}, calibrated on {ref["calibrated_on"]}: '
          f'answers {ref["answers_answerable"]:.0%} of answerable wishes, '
          f'and wrongly answers {ref["answers_impossible"]:.0%} of impossible ones')
    json.dump(ref, open(os.path.join(paths.RESULTS, f'refusal_{tag}.json'), 'w'), indent=1)

    if a.per_instruction:
        x = R[(R.pool == 'unseen_comp') & R.split.isin(ACHIEVABLE) & (R.n_pos > 0)]
        print()
        cols = ['split', 'text', 'n_pos', 'n', 'auc_model', 'auc_toxicity', 'top1_model']
        print(x[cols].sort_values('auc_model').to_string(index=False))


if __name__ == '__main__':
    main()
