# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""End to end, on the real archive and real checkpoints.

Skipped automatically where the data or the trained checkpoints are absent, so the
fast tests still run anywhere. Where they are present, this is the test that would
have caught a demo that looks fine and answers nonsense.
"""
import glob
import os

import pytest

from pbw import paths

pytestmark = pytest.mark.skipif(
    not glob.glob(os.path.join(paths.RESULTS, 'ckpt_numerical_window_rich_v3_s*.pt'))
    or not os.path.exists(paths.GROUPS),
    reason='needs P2I_DATA and trained checkpoints (make all)')

WISH = 'make the mitochondria dimmer without killing the cells'


@pytest.fixture(scope='module')
def screen():
    from pbw.predict import Screen
    return Screen(device=os.environ.get('P2I_DEVICE', 'cuda'))


def test_a_wish_returns_a_ranking_with_its_evidence(screen):
    a = screen.ask(WISH, top_k=5)
    assert len(a['ranking']) == 5
    first = a['ranking'][0]
    for key in ('compound', 'dose_uM', 'p', 'p_toxicity_only', 'cell_loss_p', 'n_wells'):
        assert key in first
    assert 0 <= first['p'] <= 1
    assert a['ranking'] == sorted(a['ranking'], key=lambda r: -r['p'])


def test_the_toxicity_score_is_present_and_is_not_the_model_score(screen):
    """If these two columns were identical the evidence card would be decoration."""
    a = screen.ask(WISH, top_k=10)
    p = [r['p'] for r in a['ranking']]
    tox = [r['p_toxicity_only'] for r in a['ranking']]
    assert p != tox


def test_the_answer_depends_on_the_question(screen):
    """A ranking that ignores the sentence would be a toxicity detector with a
    text box bolted on."""
    a = screen.ask('make the cells elongated', top_k=10)
    b = screen.ask('make the nuclei smaller', top_k=10)
    assert {r['compound'] for r in a['ranking']} != {r['compound'] for r in b['ranking']}


def test_a_known_opposite_pair_raises_the_direction_warning(screen):
    """This behaviour has 8 groups in the whole archive, below the training floor,
    and the sentence encoder puts its OPPOSITE closest among the trained concepts.
    Answering it silently would be the worst thing this system could do."""
    a = screen.ask('push the actin out to the cell edge')
    warnings = ' '.join(a['language']['warnings']).lower()
    assert 'never trained' in warnings
    assert 'opposite' in warnings


def test_an_impossible_wish_is_refused(screen):
    """Under v3 this combination happens only in dying cells, so no group in the
    archive satisfies it. Answering anyway would make the refusal decorative."""
    a = screen.ask('give me bigger nuclei and brighter mitochondria', scaffold='test')
    assert a['refused']
    assert a['shortlist_score'] < a['threshold']


def test_refusal_separates_the_two_sets_at_the_rate_that_was_reported(screen):
    """The headline refusal numbers, re-measured through the public API rather than
    read off the evaluation table."""
    from pbw.instructions import load
    rows = load('v3')
    impossible = [r for r in rows if r['split'] == 'refuse']
    answerable = [r for r in rows
                  if r['split'] in ('para_test', 'comp_test') and r['n_pos'] > 0]
    refused_bad = sum(screen.ask(r['text'], scaffold='test')['refused'] for r in impossible)
    refused_good = sum(screen.ask(r['text'], scaffold='test')['refused'] for r in answerable)
    assert refused_bad / len(impossible) > 0.5
    assert refused_good / len(answerable) < 0.25


def test_the_dose_curve_covers_the_whole_ladder(screen):
    a = screen.ask(WISH, top_k=1)
    curve = screen.dose_curve(WISH, a['ranking'][0]['compound'])
    assert len(curve) == 8
    assert curve.dose_uM.is_monotonic_increasing


def test_the_demo_layer_produces_notes_a_table_and_a_figure(screen):
    from pbw.app import answer
    notes, table, fig = answer(screen, WISH, top_k=5)
    assert 'toxicity' in notes.lower()
    assert len(table) == 5
    assert fig is not None
