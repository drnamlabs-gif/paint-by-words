# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""How much screening does the language channel actually save?

AUC says how well a ranking is ordered. It does not say what a screen costs. The
question a screening group asks is narrower and harder: *given a budget of k wells,
how often do I find something that does what I asked for?*

So this module scores every held-out goal the way a buyer would:

    budget k      the top k (compound, dose) pairs the ranking proposes
    solved        at least one of those k actually satisfies the goal
    first hit     how many wells you open before the first real one
    enrichment    precision in the top k, divided by the base rate

against the same two baselines used everywhere else, and against random order.

The result is not the one a pitch would prefer, and it is the reason this module
exists rather than a slide. Random order needs a median of about 267 wells to reach
a first hit. The model needs 12. But the toxicity-only baseline needs 18, so most of
that reduction is **general activity**, not goal-specific skill, and anyone reporting
"20x fewer wells" as a result of language conditioning would be taking credit for a
baseline. Worse for the pitch: at a budget of 48 wells, ranking by DOSE alone solves
more goals than the model does (73% against 67%), because the top dose is where most
things happen and 48 wells is enough to sample it.

Goal-specific skill shows up in exactly two places, and both are real:

  1. At a very small budget. With 8 wells, one row of a 96-well plate, the model
     solves 42% of goals against 20% for toxicity-only and 12% for dose-only. Paired
     over goals: +22%, 95% CI [+11%, +33%], 15 goals solved only by the model
     against 1 solved only by the baseline, McNemar p = 0.0005. Small budgets are
     the normal case: a hypothesis rarely gets more than a few wells.
  2. At the ceiling. With 384 wells the model solves 98% against 81%. About one
     goal in five is simply not reachable along the toxicity route at any budget.

In between, from roughly 16 to 96 wells, the two are not distinguishable (at 16
wells the gap is +11%, p = 0.12). That null is part of the result.

And the composition claim, which is what makes this cheap: 38 of the 64 held-out
goals are CONJUNCTIONS whose concept was never trained. Single behaviours are
trained; their combinations are asked for in English and answered without any
screen of their own.
"""
import argparse
import os

import numpy as np
import pandas as pd

from . import paths
from .evaluate import ACHIEVABLE, TOXICITY_PROBE, _load_preds
from .features import load_groups
from .instructions import load as load_instructions
from .labels import formula_prob

BUDGETS = (8, 24, 48, 96, 384)     # a plate row, a quarter, a half, a plate, a big plate
HEADLINE_BUDGET = 8
N_RANDOM = 200
N_BOOTSTRAP = 5000


def goals(labels='numerical', out='window', chem_kind='rich', definition='v3',
          pool='unseen_comp'):
    """Every held-out goal that something in the archive satisfies, with the scores."""
    g = load_groups()
    ins = load_instructions(definition)
    truth = np.stack([formula_prob(g, r['lits'], definition) for r in ins])
    hard = truth > 0.5
    Z, files = _load_preds(labels, out, chem_kind, definition)
    P = np.mean([z[pool].astype(np.float32) for z in Z], 0)
    gidx = Z[0][pool + '_idx']
    tox = P[next(i for i, r in enumerate(ins) if r['text'] == TOXICITY_PROBE)]
    dose = np.log10(g.dose.to_numpy(float))[gidx]

    out_rows = []
    for i, r in enumerate(ins):
        if r['split'] not in ACHIEVABLE:
            continue
        known = ~np.isnan(truth[i, gidx])
        y = hard[i, gidx][known]
        if not y.sum():
            continue
        out_rows.append(dict(text=r['text'], split=r['split'], n_atoms=len(r['lits']),
                             composed=r['split'] == 'comp_test',
                             y=y, model=P[i][known], toxicity=tox[known], dose=dose[known]))
    return out_rows, len(files)


def _solved(y, score, k):
    return bool(y[np.argsort(-score)][:k].sum() > 0)


def _first_hit(y, score):
    return int(np.argmax(y[np.argsort(-score)])) + 1


def budget_table(G, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for k in BUDGETS:
        rnd = np.mean([[_solved(d['y'], rng.random(len(d['y'])), k) for d in G]
                       for _ in range(N_RANDOM)])
        rows.append(dict(budget_wells=k,
                         model=np.mean([_solved(d['y'], d['model'], k) for d in G]),
                         toxicity_only=np.mean([_solved(d['y'], d['toxicity'], k) for d in G]),
                         dose_only=np.mean([_solved(d['y'], d['dose'], k) for d in G]),
                         random=rnd))
    return pd.DataFrame(rows)


def first_hit_table(G):
    rows = []
    for name in ('model', 'toxicity', 'dose'):
        v = [_first_hit(d['y'], d[name]) for d in G]
        rows.append(dict(scorer=name, median_wells_to_first_hit=float(np.median(v))))
    rows.append(dict(scorer='random',
                     median_wells_to_first_hit=float(np.median([len(d['y']) / d['y'].sum()
                                                                for d in G]))))
    return pd.DataFrame(rows)


def paired(G, k=HEADLINE_BUDGET, seed=0):
    """Model against the toxicity baseline, on the SAME goals. Paired, so the
    comparison is not carried by which goals happen to be easy."""
    from scipy.stats import binomtest
    rng = np.random.default_rng(seed)
    m = np.array([_solved(d['y'], d['model'], k) for d in G], int)
    t = np.array([_solved(d['y'], d['toxicity'], k) for d in G], int)
    only_model = int(((m == 1) & (t == 0)).sum())
    only_tox = int(((m == 0) & (t == 1)).sum())
    diffs = [float(np.mean(m[i] - t[i]))
             for i in (rng.integers(0, len(G), len(G)) for _ in range(N_BOOTSTRAP))]
    res = dict(budget_wells=k, n_goals=len(G), model=float(m.mean()),
               toxicity_only=float(t.mean()), difference=float(m.mean() - t.mean()),
               ci_low=float(np.percentile(diffs, 2.5)),
               ci_high=float(np.percentile(diffs, 97.5)),
               solved_only_by_model=only_model, solved_only_by_toxicity=only_tox,
               mcnemar_p=(binomtest(only_model, only_model + only_tox, 0.5).pvalue
                          if only_model + only_tox else float('nan')))
    res['goals_for_80pc_power'] = _power(only_model, only_tox, len(G))
    res['wells_for_80pc_power'] = (None if res['goals_for_80pc_power'] is None
                                   else res['goals_for_80pc_power'] * 2 * k)
    return res


def _power(only_model, only_tox, n):
    """Goals needed to repeat this paired comparison prospectively at 80% power,
    assuming the discordance rate holds. That assumption is the weak point and
    should be stated wherever this number is used."""
    disc = (only_model + only_tox) / n
    if not disc:
        return None
    share = only_model / (only_model + only_tox)
    if not 0 < share < 1 or abs(share - 0.5) < 1e-9:
        return None
    needed = (1.96 * 0.5 + 0.84 * np.sqrt(share * (1 - share))) ** 2 / (share - 0.5) ** 2
    return int(np.ceil(needed / disc))


def figure(G, budgets_df):
    """The curve a screening group would read: hit rate against budget."""
    from .figures import BLUE, FAINT, GRID, INK, MUTED, ORANGE, RED, _figure, _style
    fig, ax = _figure(figsize=(8.2, 4.8), dpi=150)
    x = budgets_df.budget_wells
    for col, colour, label in (('model', BLUE, 'paint-by-words'),
                               ('toxicity_only', RED, 'toxicity-only baseline'),
                               ('dose_only', ORANGE, 'dose-only baseline'),
                               ('random', FAINT, 'random order')):
        ax.plot(x, budgets_df[col], color=colour, lw=2, marker='o', ms=7, label=label)
        for xi, yi in zip(x, budgets_df[col]):
            if xi in (HEADLINE_BUDGET, max(BUDGETS)):
                ax.text(xi, yi + 0.028, f'{yi:.0%}', ha='center', fontsize=9, color=INK)
    ax.set_xscale('log')
    ax.set_xticks(list(BUDGETS))
    ax.set_xticklabels([str(b) for b in BUDGETS])
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(lambda v, _: f'{v:.0%}')
    ax.axvspan(16, 96, color=GRID, alpha=0.5, zorder=0)
    ax.text(40, 0.045, 'the model and the baselines are not separable in this band',
            fontsize=8.5, color=MUTED, ha='center')
    _style(ax, f'Goals solved within a screening budget\n'
               f'{len(G)} held-out goals, {sum(d["composed"] for d in G)} of them composed '
               f'and never trained',
           'wells opened (log scale)', 'share of goals with at least one real hit')
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED, loc='lower right')
    fig.tight_layout()
    out = os.path.join(paths.ROOT, 'figs', '06_screening_budget.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--labels', default='numerical')
    ap.add_argument('--definition', default='v3')
    ap.add_argument('--no-figure', action='store_true')
    a = ap.parse_args(argv)

    G, n_seeds = goals(labels=a.labels, definition=a.definition)
    composed = sum(d['composed'] for d in G)
    print(f'{len(G)} held-out goals on unseen-scaffold compounds, {composed} of them '
          f'composed and never trained ({n_seeds}-seed ensemble)')
    print()
    B = budget_table(G)
    print('share of goals with at least one real hit, within a budget of k wells')
    print(B.assign(**{c: B[c].map('{:.0%}'.format)
                      for c in ('model', 'toxicity_only', 'dose_only', 'random')})
           .to_string(index=False))
    print()
    print(first_hit_table(G).round(1).to_string(index=False))
    print()
    for k in (HEADLINE_BUDGET, 16):
        p = paired(G, k)
        print(f'budget {k}: model {p["model"]:.0%} vs toxicity-only {p["toxicity_only"]:.0%}, '
              f'{p["difference"]:+.0%} [95% CI {p["ci_low"]:+.0%}, {p["ci_high"]:+.0%}], '
              f'McNemar p = {p["mcnemar_p"]:.4f}')
        print(f'   solved only by the model: {p["solved_only_by_model"]} goals; '
              f'only by the baseline: {p["solved_only_by_toxicity"]}')
        if p['goals_for_80pc_power']:
            print(f'   a prospective repeat at 80% power: {p["goals_for_80pc_power"]} goals, '
                  f'{p["wells_for_80pc_power"]} wells')
        print()

    paths.ensure()
    B.to_csv(os.path.join(paths.RESULTS, 'screening_budget.csv'), index=False)
    pd.DataFrame([paired(G, k) for k in BUDGETS]).to_csv(
        os.path.join(paths.RESULTS, 'screening_paired.csv'), index=False)
    if not a.no_figure:
        print('figure ->', figure(G, B))


if __name__ == '__main__':
    main()
