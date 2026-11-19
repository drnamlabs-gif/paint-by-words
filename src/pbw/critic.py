# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""A vision-language model as a second measuring instrument, and its calibration.

The idea. CellProfiler measures eight numbers per well. A vision model looking at
the same well can state things no single column states, and it needs no feature
engineering. So: render each (compound, dose) group as a treated panel beside a
reference panel, ask the model to score six axes from 0 to 100, and use those
scores as an independent label source for exactly the same formulas.

The calibration problem. The model's scores are compressed and biased: it rarely
uses the ends of the scale, and each axis has its own spread. Calibrating against
the CellProfiler measurements would defeat the purpose, since the point is a label
source that does not depend on them. So calibration is label-free:

    z = (score - 50) / (1.4826 * MAD(score))

50 is the model's own stated midpoint, "no difference from the reference", and the
MAD comes from the scores themselves. The atom probability then uses the SAME rule
and the SAME fixed threshold as the measurement branch:

    p = sigmoid((sign * z - 2.0) / 0.5)

so the two branches differ only in where the numbers came from.

Two axes are bidirectional in the model's wording (brighter versus dimmer), so one
axis yields two opposite atoms. Three behaviours get no vision label at all and
stay NaN: actin at the cell edge, Golgi patchiness, and overall Golgi brightness.
The pilot found the model at chance on the first and reading the second through
brightness instead of texture, so labelling them would have manufactured noise.

The result, reported in full in MODEL_CARD.md: agreement with the measurements runs
from .93 (cell loss) and .87 (elongation) down to .58-.59 (smaller nuclei, brighter
DNA), and trained as a label source this branch ranks unseen compounds below its own
toxicity-only score. A vision model on these panels delivers toxicity, cell number
and gross shape, not subtle per-cell changes in surviving cells. That is a negative
result, and it is one of the more useful things here.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

from . import paths

MIDPOINT, THR, TEMP = 50.0, 2.0, 0.5

# axis as the model was asked about it -> (atom or (positive, negative) pair,
#                                          measurement column, sign of agreement)
AXES = {
    'fewer_cells':  ('fewer_cells', 'cfrac', -1),
    'elongated':    ('elong_up', 'elong', +1),
    'bigger_nuclei': (('nuc_up', 'nuc_dn'), 'nuc', +1),
    'dna_bright':   (('dna_up', 'dna_dn'), 'dna', +1),
    'mito_bright':  (('mito_up', 'mito_dn'), 'mito', +1),
    'actin_bright': (('actin_up', 'actin_dn'), 'actin', +1),
}
UNLABELLED = ['aedge_up', 'aedge_dn', 'gpatch_up', 'gpatch_dn', 'gmean_flat']


def load_scores(score_dir=None):
    """Every scored batch, de-duplicated with the latest score winning."""
    d = score_dir or os.path.join(paths.WORK, 'critic_scores')
    files = sorted(glob.glob(os.path.join(d, 'critic_*.json')))
    if not files:
        raise FileNotFoundError(f'no vision-model scores in {d}')
    return (pd.concat([pd.DataFrame(json.load(open(f, encoding='utf-8'))) for f in files])
            .drop_duplicates('id', keep='last'))


def calibrate(scored):
    """-> (atom probability table, per-axis calibration report). Label-free."""
    out = scored[['treatment', 'dose']].copy()
    report = []
    for axis, (atoms, column, _) in AXES.items():
        s = scored[axis].astype(float)
        mad = 1.4826 * np.nanmedian(np.abs(s - MIDPOINT)) + 1e-6
        z = (s - MIDPOINT) / mad
        pairs = [(atoms, +1)] if isinstance(atoms, str) else [(atoms[0], +1), (atoms[1], -1)]
        for atom, sign in pairs:
            out[atom] = 1 / (1 + np.exp(-(sign * z - THR) / TEMP))
            report.append(dict(axis=axis, atom=atom, mad=round(float(mad), 2),
                               n_called=int((out[atom] > 0.5).sum())))
    for atom in UNLABELLED:
        out[atom] = np.nan
    return out, pd.DataFrame(report)


def agreement(scored, atoms):
    """How well the vision scores agree with the measurements, per behaviour.

    Evaluation only. These numbers never enter training, or the calibration above
    would stop being label-free.
    """
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    from .labels import formula_prob
    rows = []
    for axis, (names, column, msign) in AXES.items():
        s = scored[axis].astype(float)
        ok = scored[column].notna() & s.notna()
        pairs = [(names, +1)] if isinstance(names, str) else [(names[0], +1), (names[1], -1)]
        for atom, sign in pairs:
            y = formula_prob(scored, [atom]) > 0.5
            n_pos = int(y[ok].sum())
            rows.append(dict(
                axis=axis, atom=atom, n_pos=n_pos,
                auc=(roc_auc_score(y[ok], sign * s[ok])
                     if 0 < n_pos < int(ok.sum()) else np.nan),
                spearman=spearmanr(s[ok], msign * scored.loc[ok, column])[0] if sign > 0 else np.nan))
    return pd.DataFrame(rows)


def main():
    paths.ensure()
    scores = load_scores()
    grp = pd.read_csv(paths.require(paths.CRITIC_GROUPS, 'see critic/README.md'),
                      keep_default_na=False, na_values=[''])
    scored = grp.merge(scores, left_on='cid', right_on='id')
    atoms, report = calibrate(scored)
    atoms.to_parquet(paths.CRITIC_ATOMS)
    print(f'{len(scored)} scored groups -> {paths.CRITIC_ATOMS}')
    print(report.to_string(index=False))
    print()
    print('agreement with the CellProfiler measurements (evaluation only):')
    print(agreement(scored, atoms).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
