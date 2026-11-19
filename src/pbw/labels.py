# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Atomic predicates over a (compound, dose) group, and the soft label of a formula.

An atom is one measurable statement about a group, for example "the mitochondria are
brighter". Each atom is a probability in [0, 1], not a hard class, because the
underlying measurement is a continuous robust z-score and a hard cut at the
threshold would throw away how far past it a group sits.

    p(atom) = sigmoid((sign * z - THR) / TEMP)

THR = 2.0 and TEMP = 0.5 were fixed before any model was trained and never tuned.
NaN propagates: a group whose measurement is missing is masked in training, not
silently treated as a negative.

Two definitions of the per-cell atoms exist, and which one is active changes every
number this repository reports.

    v2  the literal reading: "mean mitochondrial intensity per cell is high"
    v3  "... AND the cells were not killed"

v3 is the default and the honest one. CellProfiler averages over the cells it can
segment, which are the survivors, and a dying cell rounds up and reads brighter per
unit area. Measured in this archive: 37-68% of groups that satisfy a per-cell
"brighter" atom under v2 also lose cells, against a 3.5% base rate of cell loss.
Under v2 a model can score well on "make the mitochondria brighter" by learning
"kill the cells", and it does; see evaluate.py --definition v2.

The one exception: if the instruction itself asks for fewer cells, its per-cell
atoms keep the literal v2 meaning, since there is nothing to protect.
"""
import numpy as np

THR, TEMP = 2.0, 0.5
CELL_THR, CELL_TEMP = 0.6, 0.05   # surviving fraction; low-dose false-positive rate 0.5%

ATOMS = {   # name: (column in the groups table, direction)
    'fewer_cells': ('cfrac', 'cell'),
    'elong_up':   ('elong', +1),
    'nuc_up':     ('nuc', +1),    'nuc_dn':    ('nuc', -1),
    'dna_up':     ('dna', +1),    'dna_dn':    ('dna', -1),
    'mito_up':    ('mito', +1),   'mito_dn':   ('mito', -1),
    'actin_up':   ('actin', +1),  'actin_dn':  ('actin', -1),
    'aedge_up':   ('aedge', +1),  'aedge_dn':  ('aedge', -1),
    'gpatch_up':  ('gpatch', +1), 'gpatch_dn': ('gpatch', -1),
    'gmean_flat': ('gmean', 0),
}

# atoms that describe surviving cells, so v3 conjoins "and not fewer_cells"
PER_CELL = [a for a in ATOMS if a not in ('fewer_cells', 'gmean_flat')]

HUMAN = {
    'fewer_cells': 'fewer cells', 'elong_up': 'elongated cells',
    'nuc_up': 'bigger nuclei', 'nuc_dn': 'smaller nuclei',
    'dna_up': 'brighter DNA staining', 'dna_dn': 'dimmer DNA staining',
    'mito_up': 'brighter mitochondria', 'mito_dn': 'dimmer mitochondria',
    'actin_up': 'brighter actin', 'actin_dn': 'fainter actin',
    'aedge_up': 'actin pushed out to the cell edge',
    'aedge_dn': 'actin pulled away from the cell edge',
    'gpatch_up': 'patchy Golgi staining', 'gpatch_dn': 'smoother Golgi staining',
    'gmean_flat': 'unchanged Golgi brightness',
}


def atom_prob(groups, name):
    """Probability that `name` holds for each row of the groups table. NaN = unmeasured."""
    col, sign = ATOMS[name]
    x = groups[col].to_numpy(float)
    if sign == 'cell':                 # low surviving fraction = cells were killed
        return 1 / (1 + np.exp((x - CELL_THR) / CELL_TEMP))
    if sign == 0:                      # "unchanged": |z| small
        return 1 / (1 + np.exp((np.abs(x) - 1.0) / TEMP))
    return 1 / (1 + np.exp(-(sign * x - THR) / TEMP))


def combine(raw, literals, definition='v3'):
    """Soft label of a conjunction of literals. `raw(atom_name) -> array`.

    A literal is an atom name, or '!' + atom name for its negation. Taking `raw` as a
    callable is what lets the same definition logic serve both label sources: the
    CellProfiler measurements and the vision-model scores.
    """
    asks_to_kill = 'fewer_cells' in literals
    out = None
    for lit in literals:
        name = lit.lstrip('!')
        a = raw(name)
        if definition == 'v3' and name in PER_CELL and not asks_to_kill:
            a = a * (1 - np.nan_to_num(raw('fewer_cells'), nan=0.0))
        if lit.startswith('!'):
            a = 1 - a
        out = a if out is None else out * a
    return out


def formula_prob(groups, literals, definition='v3'):
    return combine(lambda n: atom_prob(groups, n), literals, definition)


def explain(groups, row, literals, definition='v3'):
    """Per-atom breakdown for one group, for the evidence card."""
    sub = groups.iloc[[row]]
    parts = []
    for lit in literals:
        name = lit.lstrip('!')
        col = ATOMS[name][0]
        parts.append(dict(literal=lit, atom=HUMAN[name], z=float(sub[col].iloc[0]),
                          p=float(atom_prob(sub, name)[0])))
    if definition == 'v3' and 'fewer_cells' not in literals:
        parts.append(dict(literal='(v3) cells survive', atom='cells not killed',
                          z=float(sub['cfrac'].iloc[0]),
                          p=float(1 - atom_prob(sub, 'fewer_cells')[0])))
    return parts
