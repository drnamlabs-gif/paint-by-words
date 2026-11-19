# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""The splits have to actually hold out what they claim to hold out."""
import numpy as np
import pandas as pd
import pytest

from pbw.instructions import PHRASINGS, build
from pbw.labels import ATOMS, formula_prob


@pytest.fixture(scope='module')
def groups():
    """A small synthetic archive: one group per behaviour plus inert and toxic ones."""
    rng = np.random.default_rng(0)
    n = 400
    cols = {c: rng.normal(0, 1, n) for c, _ in ATOMS.values() if c != 'cfrac'}
    cols['cfrac'] = np.ones(n)
    g = pd.DataFrame(cols)
    for i, (short, _) in enumerate(ATOMS.values()):
        if short != 'cfrac':
            g.loc[i * 20:(i * 20 + 19), short] = 4.0      # a block that clearly fires
    g.loc[380:, 'cfrac'] = 0.2                            # a block of dead wells
    return g


def test_phrasings_are_three_train_two_held_out(groups):
    rows = build(groups)
    for atom, phrasings in PHRASINGS.items():
        if atom == 'aedge_up':
            continue
        mine = {r['text']: r['split'] for r in rows if r['lits'] == [atom]}
        assert [mine[p] for p in phrasings[:3]] == ['train'] * 3
        assert [mine[p] for p in phrasings[3:]] == ['para_test'] * 2


def test_held_out_phrasings_never_appear_in_training(groups):
    rows = build(groups)
    train = {r['text'] for r in rows if r['split'] == 'train'}
    held = {r['text'] for r in rows if r['split'] in ('para_test', 'comp_test')}
    assert not (train & held)


def test_comp_test_concepts_are_never_trained(groups):
    rows = build(groups)
    train = {r['concept'] for r in rows if r['split'] == 'train'}
    for r in rows:
        if r['split'] == 'comp_test':
            assert r['concept'] not in train


def test_refusals_really_have_no_positive_group(groups):
    rows = build(groups)
    refuse = [r for r in rows if r['split'] == 'refuse']
    assert refuse
    for r in refuse:
        p = formula_prob(groups, r['lits'])
        assert np.nansum(p > 0.5) == 0, r['text']


def test_v3_refuses_more_than_v2(groups):
    """Conjunctions that only ever happened in dying cells become impossible."""
    n2 = sum(r['split'] == 'refuse' for r in build(groups, 'v2'))
    n3 = sum(r['split'] == 'refuse' for r in build(groups, 'v3'))
    assert n3 >= n2


def test_v3_drops_the_without_killing_composites(groups):
    """Under v3 'X without killing the cells' is just X, so it is not a concept."""
    texts = {r['text'] for r in build(groups, 'v3')}
    assert not any('without killing the cells' in t for t in texts)
    assert any('without killing the cells' in r['text'] for r in build(groups, 'v2'))
