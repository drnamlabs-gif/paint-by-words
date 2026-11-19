# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""The label definition is the part of this repository that is easiest to get
quietly wrong, so it is the part with tests."""
import numpy as np
import pandas as pd
import pytest

from pbw.labels import ATOMS, CELL_THR, THR, atom_prob, combine, formula_prob


def frame(**cols):
    n = len(next(iter(cols.values())))
    base = {c: np.zeros(n) for c, _ in ATOMS.values() if c != 'cfrac'}
    base['cfrac'] = np.ones(n)
    base.update(cols)
    return pd.DataFrame(base)


def test_atom_crosses_half_at_the_threshold():
    g = frame(mito=[THR - 1, THR, THR + 1])
    p = atom_prob(g, 'mito_up')
    assert p[0] < 0.5 < p[2]
    assert p[1] == pytest.approx(0.5, abs=1e-6)
    assert np.all(np.diff(p) > 0), 'must be monotone in the measurement'


def test_direction_is_respected():
    g = frame(mito=[5.0])
    assert atom_prob(g, 'mito_up')[0] > 0.9
    assert atom_prob(g, 'mito_dn')[0] < 0.1


def test_unchanged_atom_is_two_sided():
    g = frame(gmean=[-5.0, 0.0, 5.0])
    p = atom_prob(g, 'gmean_flat')
    assert p[1] > 0.85 and p[0] < 0.01 and p[2] < 0.01


def test_cell_loss_fires_below_the_surviving_fraction_cut():
    g = frame(cfrac=[0.2, CELL_THR, 1.0])
    p = atom_prob(g, 'fewer_cells')
    assert p[0] > 0.9
    assert p[1] == pytest.approx(0.5, abs=1e-6)
    assert p[2] < 0.1


def test_missing_measurement_stays_nan_and_is_never_a_negative():
    g = frame(mito=[np.nan])
    assert np.isnan(atom_prob(g, 'mito_up')[0])
    assert np.isnan(formula_prob(g, ['mito_up'])[0])


def test_v3_cancels_a_per_cell_hit_that_came_with_cell_death():
    """The whole finding, in one assertion: brighter mitochondria in a well that
    lost its cells counts under v2 and must not count under v3."""
    g = frame(mito=[5.0], cfrac=[0.2])
    assert formula_prob(g, ['mito_up'], 'v2')[0] > 0.9
    assert formula_prob(g, ['mito_up'], 'v3')[0] < 0.1


def test_v3_keeps_a_per_cell_hit_in_a_healthy_well():
    g = frame(mito=[5.0], cfrac=[1.0])
    assert formula_prob(g, ['mito_up'], 'v3')[0] > 0.9


def test_v3_leaves_an_explicit_kill_request_alone():
    """If the wish is to kill the cells, there is nothing to protect, so the
    per-cell atoms keep their literal meaning."""
    g = frame(mito=[5.0], cfrac=[0.2])
    both = formula_prob(g, ['mito_up', 'fewer_cells'], 'v3')[0]
    assert both > 0.8


def test_negation_and_conjunction():
    g = frame(mito=[5.0], nuc=[0.0], cfrac=[1.0])
    assert combine(lambda n: atom_prob(g, n), ['mito_up', '!nuc_up'], 'v3')[0] > 0.9
    assert combine(lambda n: atom_prob(g, n), ['mito_up', 'nuc_up'], 'v3')[0] < 0.1


def test_combine_accepts_any_label_source():
    """combine() takes a callable so the same definition applies to vision-model
    scores as to measurements; this is what makes the two branches comparable."""
    fake = {'mito_up': np.array([0.9]), 'fewer_cells': np.array([0.5])}
    out = combine(lambda n: fake[n], ['mito_up'], 'v3')
    assert out[0] == pytest.approx(0.9 * 0.5)
