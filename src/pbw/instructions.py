# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""The instruction set, and the four splits that make the evaluation mean something.

An instruction is an English sentence plus the formula it stands for: a conjunction
of literals over the atoms in labels.py. The sentence is what the model reads; the
formula is only used to make the label and to score the answer.

    train       phrasings and concepts the model is fitted on
    para_test   UNSEEN phrasings of a trained concept. Tests the language channel.
    comp_test   UNSEEN concepts: a pair or triple of behaviours, or a negation,
                whose combination was never a training target. Tests composition.
    refuse      an effect that NO (compound, dose) in the archive produces. The
                correct behaviour is to report that nothing matches. A system that
                always answers is not usable for screening, so this is scored.
    rare_test   a concept with too few positives in the archive to train on.

The refuse set is built by measurement, not by hand: every candidate combination of
trained behaviours is evaluated against the archive, and the ones with exactly zero
positive groups become refusals. Under the v3 definition this set grows from 25 to
37, because combinations such as "bigger nuclei and brighter mitochondria" turn out
to occur only in dying cells.

Phrasings are three train, two held out, per atom, written before any training run.
"""
import itertools
import json
import os

import numpy as np

from .labels import HUMAN, formula_prob

PHRASINGS = {
    'fewer_cells': ["kill most of the cells", "reduce the number of cells in the well",
                    "make the cell population shrink", "leave far fewer cells alive",
                    "thin out the culture by killing cells"],
    'elong_up': ["make the cells elongated", "stretch the cells into long shapes",
                 "make the cells more spindle-like", "turn the cells long and thin",
                 "give the cells an elongated outline"],
    'nuc_up': ["make the nuclei bigger", "enlarge the nuclei", "swell the cell nuclei",
               "increase nuclear size", "grow larger nuclei"],
    'nuc_dn': ["make the nuclei smaller", "shrink the nuclei", "reduce nuclear size",
               "make the nuclei compact and small", "decrease the size of the nucleus"],
    'dna_up': ["make the DNA stain brighter", "increase the DNA staining intensity",
               "make the nuclei stain more strongly for DNA",
               "condense the DNA so nuclei glow brighter",
               "raise the Hoechst signal in the nuclei"],
    'dna_dn': ["make the DNA stain dimmer", "decrease DNA staining",
               "make the nuclei stain faintly for DNA", "weaken the Hoechst signal",
               "lower the DNA intensity in nuclei"],
    'mito_up': ["make the mitochondria brighter", "increase the mitochondrial signal",
                "boost MitoTracker staining", "make the mitochondria glow more",
                "raise mitochondrial staining intensity"],
    'mito_dn': ["make the mitochondria dimmer", "reduce the mitochondrial signal",
                "weaken MitoTracker staining", "make the mitochondria fade",
                "lower mitochondrial staining intensity"],
    'actin_up': ["make the actin brighter", "increase actin staining",
                 "strengthen the phalloidin signal",
                 "make the actin cytoskeleton more intense",
                 "raise the amount of stained actin"],
    'actin_dn': ["make the actin fainter", "reduce actin staining",
                 "weaken the phalloidin signal", "make the actin cytoskeleton dimmer",
                 "lower the actin signal in the cells"],
    'aedge_up': ["push the actin out to the cell edge", "concentrate actin at the cell periphery",
                 "move the actin to the rim of the cell",
                 "redistribute actin toward the cell border",
                 "make actin accumulate at the cell margin"],
    'aedge_dn': ["pull the actin away from the cell edge", "move actin toward the cell centre",
                 "draw the actin inward from the periphery", "deplete actin at the cell border",
                 "shift actin from the rim into the interior"],
    'gpatch_up': ["make the Golgi and membrane staining patchy",
                  "make the WGA signal uneven and blotchy", "fragment the Golgi into patches",
                  "make the membrane stain look spotty", "scatter the Golgi signal into clumps"],
    'gpatch_dn': ["make the Golgi and membrane staining smoother", "make the WGA signal more even",
                  "reduce the patchiness of the Golgi stain",
                  "make the membrane stain look uniform", "smooth out the Golgi signal"],
}

# Atoms used inside composite templates. aedge_up is too rare in the archive to train
# and gmean_flat is a nuisance term, so neither is composed. The order is fixed: it
# seeds the shuffle that assigns composites to train or comp_test, so changing it
# would silently change the split and invalidate every number in README.md.
COMPOSABLE = ['elong_up', 'nuc_up', 'nuc_dn', 'dna_up', 'dna_dn', 'mito_up', 'mito_dn',
              'actin_up', 'actin_dn', 'aedge_dn', 'gpatch_up', 'gpatch_dn', 'fewer_cells']

NEGATIONS = {
    'fewer_cells': 'without killing the cells',
    'nuc_up': 'without enlarging the nuclei',
    'dna_up': 'without making the DNA brighter',
    'mito_up': 'while keeping the mitochondria normal',
    'gpatch_up': 'without making the Golgi patchy',
    'elong_up': 'without stretching the cells',
}

# hand-written refusals: a real screening wish the archive cannot grant
HAND_REFUSE = [
    ("make the Golgi staining smoother while keeping its brightness normal",
     ['gpatch_dn', 'gmean_flat']),
    ("even out the Golgi signal without changing how bright it is",
     ['gpatch_dn', 'gmean_flat']),
]

# direction pairs, used to warn when the sentence encoder cannot tell them apart
OPPOSITES = {'nuc_up': 'nuc_dn', 'dna_up': 'dna_dn', 'mito_up': 'mito_dn',
             'actin_up': 'actin_dn', 'aedge_up': 'aedge_dn', 'gpatch_up': 'gpatch_dn'}
OPPOSITES.update({v: k for k, v in OPPOSITES.items()})

MIN_POSITIVES = 10

# Behaviours the archive cannot train: too few (compound, dose) groups reach them.
# They are still evaluated, as rare_test, and language.audit warns when a query is
# closest to one of them, because the model will answer with a trained neighbour.
RARE = {'aedge_up'}
TRAINED_ATOMS = [a for a in PHRASINGS if a not in RARE]


def build(groups, definition='v3', seed=0):
    rows = []
    for atom, phrasings in PHRASINGS.items():
        for i, text in enumerate(phrasings):
            split = ('rare_test' if atom in RARE
                     else 'train' if i < 3 else 'para_test')
            rows.append(dict(text=text, lits=[atom], concept=atom, split=split))
    for text, lits in HAND_REFUSE:
        rows.append(dict(text=text, lits=lits, concept='+'.join(lits), split='refuse'))

    candidates = []
    for a, b in itertools.combinations(COMPOSABLE, 2):
        if a.split('_')[0] == b.split('_')[0]:
            continue
        candidates.append(([a, b],
                           f'give me {HUMAN[a]} and {HUMAN[b]}',
                           f'I want {HUMAN[a]} together with {HUMAN[b]}'))
    for a in COMPOSABLE:
        for neg, phrase in NEGATIONS.items():
            if definition == 'v3' and neg == 'fewer_cells':
                continue      # under v3 "X without killing the cells" is just X
            if neg == a or neg.split('_')[0] == a.split('_')[0]:
                continue
            candidates.append(([a, '!' + neg],
                               f'produce {HUMAN[a]} {phrase}',
                               f'I need {HUMAN[a]}, {phrase}'))

    n_pos = [int((formula_prob(groups, c[0], definition) > 0.5).sum()) for c in candidates]
    for cand, n in zip(candidates, n_pos):
        if n == 0:
            rows.append(dict(text=cand[1], lits=cand[0],
                             concept='+'.join(cand[0]), split='refuse'))

    rng = np.random.default_rng(seed)
    keep = [c for c, n in zip(candidates, n_pos) if n >= MIN_POSITIVES]
    rng.shuffle(keep)
    n_test = len(keep) // 3
    for j, (lits, text_a, text_b) in enumerate(keep):
        split = 'comp_test' if j < n_test else 'train'
        for text in ((text_a, text_b) if split == 'train' else (text_a,)):
            rows.append(dict(text=text, lits=lits, concept='+'.join(lits), split=split))

    triple_ok, triple_no = [], []
    for a, b, c in itertools.combinations(COMPOSABLE, 3):
        if len({x.split('_')[0] for x in (a, b, c)}) < 3:
            continue
        n = int((formula_prob(groups, [a, b, c], definition) > 0.5).sum())
        text = f'give me {HUMAN[a]}, {HUMAN[b]} and {HUMAN[c]}'
        if n >= MIN_POSITIVES:
            triple_ok.append(([a, b, c], text))
        elif n == 0:
            triple_no.append(([a, b, c], text))
    rng.shuffle(triple_ok)
    rng.shuffle(triple_no)
    for lits, text in triple_ok[:10]:
        rows.append(dict(text=text, lits=lits, concept='+'.join(lits), split='comp_test'))
    for lits, text in triple_no[:20]:
        rows.append(dict(text=text, lits=lits, concept='+'.join(lits), split='refuse'))

    for r in rows:
        r['n_pos'] = int((formula_prob(groups, r['lits'], definition) > 0.5).sum())
    return rows


def path(definition='v3'):
    from . import paths
    return os.path.join(paths.WORK, f'instructions_{definition}.json')


def load(definition='v3'):
    from . import paths
    paths.require(path(definition), f'python -m pbw.instructions {definition}')
    return json.load(open(path(definition), encoding='utf-8'))


if __name__ == '__main__':
    import sys

    import pandas as pd

    from . import paths
    from .features import load_groups

    definition = sys.argv[1] if len(sys.argv) > 1 else 'v3'
    paths.ensure()
    rows = build(load_groups(), definition)
    json.dump(rows, open(path(definition), 'w', encoding='utf-8'), indent=1)
    df = pd.DataFrame(rows)
    print(definition, len(df), df.split.value_counts().to_dict())
    print(df.groupby('split').n_pos.describe()[['count', 'min', '50%', 'max']].round(0))
