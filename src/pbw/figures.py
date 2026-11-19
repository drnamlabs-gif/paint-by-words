# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""The five figures in README.md, all from the archive and the saved results.

    01_what_is_a_dot        real images: six channels, and how a cell is counted
    02_dose_series          one compound across its eight doses, one brightness scale
    03_the_definition_problem   why the per-cell predicates had to be redefined
    04_dose_window          why the output is non-monotone in dose
    05_against_the_baselines    the model against two baselines that ignore the question

Figures 1 and 2 need the image shards. Figures 3 and 4 need work/groups.parquet.
Figure 5 needs trained results for both label definitions (make train baselines).
"""
import glob
import os

import numpy as np
import pandas as pd

from . import paths
from .features import load_groups
from .images import CHANNEL_NAMES, band, hcat, nuclei_mask, read_shard_images, scale_from, to_grey
from .labels import ATOMS, HUMAN, PER_CELL, atom_prob, formula_prob

FIGS = os.path.join(paths.ROOT, 'figs')

INK, MUTED, FAINT = '#0b0b0b', '#52514e', '#898781'
GRID, SURFACE = '#e1e0d9', '#fcfcfb'
BLUE, ORANGE, AQUA, VIOLET, RED = '#2a78d6', '#eb6834', '#1baf7a', '#6250d6', '#e34948'

EXAMPLE_COMPOUND = 'ABT-263'
ZOOM = (slice(128, 384), slice(128, 384))


def _style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc='left')
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def _figure(*a, **kw):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(*a, **kw)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes


# ---------------------------------------------------------------- real images

def _wells_for(compound):
    """One well per dose for `compound`, plus a low-dose reference on the same plate."""
    m = pd.read_csv(paths.require(paths.METADATA, 'see README'), low_memory=False)
    m = m[(m.perturbation_type == 'COMPOUND')
          & (m.well_type_label == 'Query Compounds + Intron control')].copy()
    from .features import _dose
    m['plate_id'] = m.experiment_name + '_' + m.plate.astype(str)
    m['dose'] = _dose(m.concentration)
    series = m[m.treatment == compound].sort_values('dose').groupby('dose').first().reset_index()
    plate = series.plate_id.iloc[0]
    reference = m[(m.plate_id == plate) & (m.concentration <= 0.01)].iloc[0]
    return series, reference


def _load_images(rows):
    want = {}
    for r in rows:
        for ch in range(1, 7):
            want[f'{r.experiment_name}/Plate{r.plate}/{r.address}_s1_{ch}'] = True
    found = {}
    for shard in sorted(glob.glob(os.path.join(paths.SHARDS, '*.parquet'))):
        found.update(read_shard_images(shard, want))
        if len(found) == len(want):
            break
    missing = set(want) - set(found)
    if missing:
        raise FileNotFoundError(f'{len(missing)} images not in the downloaded shards, '
                                f'e.g. {sorted(missing)[0]}')
    return found


def _stack(found, r):
    return np.stack([found[f'{r.experiment_name}/Plate{r.plate}/{r.address}_s1_{c}']
                     for c in range(1, 7)])


def fig_what_is_a_dot():
    """What the data physically is, and what one measurement counts."""
    from PIL import Image
    series, reference = _wells_for(EXAMPLE_COMPOUND)
    found = _load_images([reference])
    x = _stack(found, reference)

    tiles = []
    for c in range(6):
        a, b = scale_from(x[c])
        tiles.append(band(Image.fromarray(to_grey(x[c], a, b)).resize((300, 300)),
                          CHANNEL_NAMES[c]))
    channels = band(hcat(tiles), 'ONE WELL, SIX STAINS (an untreated well)')

    from scipy import ndimage as ndi

    from .images import SMOOTH_SIGMA, otsu, vcat
    dna = x[0][ZOOM].astype(float)
    smoothed = ndi.gaussian_filter(dna, SMOOTH_SIGMA)
    mask = smoothed > otsu(smoothed)
    lab, sizes = nuclei_mask(x[0][ZOOM])
    keep = np.isin(lab, np.where((sizes > 30) & (sizes < 2000))[0] + 1)
    outline = keep & ~ndi.binary_erosion(keep, iterations=2)

    a, b = scale_from(x[0])
    grey = to_grey(dna, a, b)
    rgb = np.stack([grey] * 3, -1)
    rgb[outline] = [235, 104, 52]
    n = int(ndi.label(keep)[1])
    w = (6 * 300 + 5 * 8 - 3 * 8) // 4        # match the width of the six-channel row
    steps = [
        band(Image.fromarray(grey).resize((w, w)), '1. the nucleus channel, zoomed'),
        band(Image.fromarray(to_grey(smoothed, a, b)).resize((w, w)),
             f'2. smoothed (Gaussian, sigma {SMOOTH_SIGMA})'),
        band(Image.fromarray((mask * 255).astype(np.uint8)).resize((w, w)),
             '3. above the Otsu threshold'),
        band(Image.fromarray(rgb).resize((w, w)),
             f'4. kept after the size filter: {n} nuclei'),
    ]
    caption = ('EACH BRIGHT DOT IN CHANNEL 1 IS ONE CELL NUCLEUS, SO COUNTING DOTS IS '
               'COUNTING CELLS')
    vcat([channels, band(hcat(steps), caption)]).save(
        os.path.join(FIGS, '01_what_is_a_dot.png'))


def fig_dose_series():
    """Eight doses of one compound, on ONE brightness scale, so the change is real."""
    from PIL import Image

    from .images import vcat
    series, reference = _wells_for(EXAMPLE_COMPOUND)
    found = _load_images(list(series.itertuples()) + [reference])
    ref = _stack(found, reference)

    rows = []
    for channel, name in ((0, 'nuclei (Hoechst)'), (2, 'actin (phalloidin)')):
        a, b = scale_from(ref[channel])
        cells = [band(Image.fromarray(to_grey(_stack(found, r)[channel], a, b)).resize((190, 190)),
                      f'{r.dose:g} uM') for r in series.itertuples()]
        rows.append(band(hcat(cells), name))
    vcat(rows).save(os.path.join(FIGS, '02_dose_series.png'))


# ---------------------------------------------------------------- charts

def fig_definition_problem():
    """The finding: a per-cell 'change' is usually a well that lost its cells."""
    g = load_groups()
    measured = g[g.cfrac.notna()]
    base = float((atom_prob(measured, 'fewer_cells') > 0.5).mean())

    rows = []
    for atom in PER_CELL:
        if atom.startswith(('aedge', 'gpatch')):
            continue
        hit = atom_prob(measured, atom) > 0.5
        if hit.sum() < 10:
            continue
        dying = (atom_prob(measured, 'fewer_cells') > 0.5)[hit]
        rows.append(dict(atom=HUMAN[atom], n=int(hit.sum()), frac=float(dying.mean())))
    r = pd.DataFrame(rows).sort_values('frac')

    fig, axes = _figure(1, 2, figsize=(11.5, 4.4), dpi=150,
                        gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax = axes[0]
    y = np.arange(len(r))
    ax.barh(y, r.frac, color=[RED if f > 0.3 else BLUE for f in r.frac], height=0.62)
    ax.axvline(base, color=INK, lw=1.4, ls='--')
    ax.text(base + 0.012, len(r) - 0.4, f'base rate {base:.1%}', color=INK, fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels([f'{a}  (n={n})' for a, n in zip(r.atom, r.n)], fontsize=8.5,
                       color=MUTED)
    ax.set_xlim(0, max(0.8, r.frac.max() * 1.15))
    ax.xaxis.set_major_formatter(lambda v, _: f'{v:.0%}')
    _style(ax, 'Of the groups that satisfy a per-cell change,\nhow many also lost their cells?',
           'share of those groups that lost cells')

    # right panel: measured per-cell intensity against surviving fraction
    ax = axes[1]
    sub = measured.sample(4000, random_state=0) if len(measured) > 4000 else measured
    ax.scatter(sub.cfrac, sub.mito, s=6, c=FAINT, alpha=0.35, linewidths=0, label='all groups')
    hot = measured[atom_prob(measured, 'mito_up') > 0.5]
    ax.scatter(hot.cfrac, hot.mito, s=14, c=RED, alpha=0.8, linewidths=0,
               label='"brighter mitochondria" under v2')
    ax.axvline(0.6, color=INK, lw=1.2, ls='--')
    ax.set_xlim(0, 1.6)
    ax.set_ylim(-4, float(np.nanpercentile(measured.mito, 99.9)))
    ax.text(0.55, ax.get_ylim()[1] * 0.96, 'cells killed  <', color=INK, fontsize=8.5,
            ha='right', va='top')
    _style(ax, 'A dying cell rounds up, so it reads brighter',
           'surviving cell fraction', 'mitochondrial intensity per cell (z)')
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED, loc='upper right')

    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, '03_the_definition_problem.png'))
    return r, base


def fig_dose_window():
    """Why the output is non-monotone: the effect is usually gone at the top dose."""
    g = load_groups()
    p3 = formula_prob(g, ['mito_up'], 'v3')
    g = g.assign(p3=p3)
    doses = sorted(g.dose.unique())

    peaks, n_off_top, middle = [], 0, []
    for c, sub in g[g.cfrac.notna()].groupby('treatment'):
        s = sub.sort_values('dose')
        if not (s.p3 > 0.5).any():
            continue
        rank = doses.index(s.dose.iloc[int(np.argmax(s.p3.to_numpy()))]) + 1
        peaks.append(rank)
        if rank < len(doses):
            n_off_top += 1
            middle.append((float(s.p3.max()), c))
    peaks = np.array(peaks)

    fig, axes = _figure(1, 2, figsize=(11.5, 4.2), dpi=150)
    ax = axes[0]
    # a compound whose effect peaks at a MIDDLE dose: the case the window output exists for
    pick = max(middle)[1]
    s = g[g.treatment == pick].sort_values('dose')
    ax.plot(s.dose, s.p3, color=BLUE, lw=2, marker='o', ms=6,
            label='measured: brighter mitochondria AND cells alive')
    ax.plot(s.dose, formula_prob(s, ['mito_up'], 'v2'), color=VIOLET, lw=1.6, ls='--',
            marker='o', ms=5, label='the same, old definition (v2)')
    ax.plot(s.dose, s.cfrac.clip(0, 1.2), color=ORANGE, lw=2, marker='o', ms=6,
            label='surviving cell fraction')
    ax.set_xscale('log')
    ax.set_ylim(-0.05, 1.25)
    _style(ax, f'{pick}: the effect exists only in a dose window',
           'dose (micromolar, log scale)', 'probability / fraction')
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED, loc='upper left')

    ax = axes[1]
    counts = [int((peaks == i).sum()) for i in range(1, len(doses) + 1)]
    colors = [BLUE] * (len(doses) - 1) + [RED]
    ax.bar(range(1, len(doses) + 1), counts, color=colors, width=0.66)
    ax.set_xticks(range(1, len(doses) + 1))
    ax.set_xticklabels([f'{d:g}' for d in doses], fontsize=8, rotation=45)
    _style(ax, f'Where the effect peaks, over {len(peaks)} compounds\n'
               f'{n_off_top} of {len(peaks)} peak BELOW the top dose',
           'dose at which the measured effect is largest (uM)', 'compounds')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, '04_dose_window.png'))
    return len(peaks), n_off_top


def fig_baselines():
    """The model against two baselines that never read the question."""
    from .evaluate import evaluate, summarise
    panels = []
    for definition in ('v2', 'v3'):
        try:
            panels.append((definition, summarise(evaluate(definition=definition))))
        except FileNotFoundError:
            continue
    if not panels:
        raise FileNotFoundError('no results; run make train (and make baselines for v2)')

    fig, axes = _figure(1, len(panels), figsize=(5.8 * len(panels), 4.8), dpi=150,
                        squeeze=False)
    handles = None
    for ax, (definition, s) in zip(axes[0], panels):
        s = s[s.compounds == 'unseen scaffold']
        x = np.arange(len(s))
        w = 0.26
        bars = [ax.bar(x - w, s.auc, w, color=BLUE, label='paint-by-words'),
                ax.bar(x, s.auc_toxicity_only, w, color=RED, label='toxicity-only baseline'),
                ax.bar(x + w, s.auc_dose_only, w, color=FAINT, label='dose-only baseline')]
        handles = bars
        for xi, (m, t, d, gap) in enumerate(zip(s.auc, s.auc_toxicity_only,
                                                s.auc_dose_only, s.margin)):
            for dx, v in ((-w, m), (0, t), (w, d)):
                ax.text(xi + dx, v + 0.008, f'{v:.3f}', ha='center', fontsize=8, color=INK)
            ax.annotate('', xy=(xi - w + 0.09, m), xytext=(xi - w + 0.09, t),
                        arrowprops=dict(arrowstyle='<->', color=INK, lw=1.1))
            ax.text(xi - w, m + 0.04, f'+{gap:.3f} over toxicity', ha='center',
                    fontsize=9, color=INK, weight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(s.instructions, fontsize=9.5, color=MUTED)
        ax.set_ylim(0.5, 1.0)       # 0.5 is chance: the honest floor for an AUC bar
        label = ('v2, the literal definition: the model looks better,\n'
                 'and is barely above a baseline that ignores the question'
                 if definition == 'v2' else
                 'v3, "and the cells survive": the model looks worse,\n'
                 'and the gap over that baseline is ten times larger')
        _style(ax, label, None, 'AUC on unseen-scaffold compounds (0.5 = chance)')
    fig.legend(handles=[b for b in handles], frameon=False, fontsize=9,
               labelcolor=MUTED, ncol=3, loc='lower center', bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(FIGS, '05_against_the_baselines.png'))
    return panels


def main():
    os.makedirs(FIGS, exist_ok=True)
    made = []
    for name, fn in (('01 what is a dot', fig_what_is_a_dot),
                     ('02 dose series', fig_dose_series),
                     ('03 definition problem', fig_definition_problem),
                     ('04 dose window', fig_dose_window),
                     ('05 baselines', fig_baselines)):
        try:
            out = fn()
            made.append(name)
            print(f'ok   {name}' + (f'   {out}' if isinstance(out, tuple) else ''))
        except FileNotFoundError as e:
            print(f'skip {name}: {e}')
    print(f'{len(made)} figures -> {FIGS}')


if __name__ == '__main__':
    main()
