# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""The demo. One sentence in; a ranking, a dose curve, and the evidence card out.

Run with:  python -m pbw.app

The layout is deliberate. The toxicity-only column sits inside the ranking table
rather than in a footnote, and the language warning sits above the table rather than
below it, because both exist to stop a user from reading a confident ranking as a
confident answer.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt   # noqa: E402
import pandas as pd               # noqa: E402

from .predict import Screen       # noqa: E402

INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e1e0d9', '#fcfcfb'
BLUE, ORANGE = '#2a78d6', '#eb6834'

EXAMPLES = [
    'make the mitochondria dimmer without killing the cells',
    'make the cells elongated',
    'give me bigger nuclei and dimmer DNA staining',
    'make the Golgi staining smoother while keeping its brightness normal',
    'push the actin out to the cell edge',
]


def dose_figure(curve, compound, wish):
    fig, ax = plt.subplots(figsize=(6.0, 3.4), dpi=140)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot(curve.dose_uM, curve.p, color=BLUE, lw=2, marker='o', ms=6,
            label='p(wish satisfied)')
    if curve.surviving_fraction.notna().any():
        ax.plot(curve.dose_uM, curve.surviving_fraction, color=ORANGE, lw=2,
                marker='o', ms=6, label='surviving cell fraction')
    ax.set_xscale('log')
    ax.set_xlabel('dose (micromolar, log scale)', color=MUTED, fontsize=9)
    ax.set_ylabel('probability', color=MUTED, fontsize=9)
    ax.set_title(f'{compound}: predicted dose response', color=INK, fontsize=11, loc='left')
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, color=GRID, lw=0.8)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED)
    fig.tight_layout()
    return fig


def answer(screen, wish, top_k=10, pool='unseen scaffolds only'):
    """The whole demo except the widgets: notes, table, figure. No gradio needed,
    so this part is tested rather than only clicked."""
    ans = screen.ask(wish, int(top_k), None if pool == 'all compounds' else 'test')
    lines = []
    for w in ans['language']['warnings']:
        lines.append(f'**Warning.** {w}')
    if ans['refused']:
        lines.append(f'**Refused.** {ans["note"]}')
    near = ', '.join(f'"{t}" ({c:.2f})' for t, _, c in ans['language']['nearest_trained'][:3])
    lines.append(f'Nearest trained phrasings: {near}')
    lines.append(f'Groups in the archive that satisfy the nearest trained wish: '
                 f'**{ans["n_archive_groups_satisfying"]}**. '
                 f'A small number here means the ranking is a lottery, whatever its AUC.')
    lines.append('`p_toxicity_only` is the model\'s own score for "kill most of the cells". '
                 'A compound high on both columns was probably found for being toxic.')

    table = pd.DataFrame([{k: v for k, v in r.items() if k != 'measured_z'}
                          for r in ans['ranking']])
    top = ans['ranking'][0]['compound'] if ans['ranking'] else None
    fig = dose_figure(screen.dose_curve(wish, top), top, wish) if top else None
    return '\n\n'.join(lines), table, fig


def build(screen):
    import gradio as gr

    def run(wish, top_k, pool):
        return answer(screen, wish, top_k, pool)

    with gr.Blocks(title='paint-by-words', theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            '# paint-by-words\n'
            'Ask a cell-image archive for a phenotype in plain English. '
            'You get a compound, a dose, and the evidence, or a refusal.\n\n'
            'Compounds are held out by Bemis-Murcko scaffold. On unseen scaffolds the '
            'model reaches AUC .79 on unseen phrasings and .86 on unseen combinations, '
            'against .73 / .82 for a baseline that ignores the question entirely and just '
            'ranks by toxicity. Read the two side by side.\n\n'
            'It selects from an existing library; it does not design molecules, and no '
            'prediction here has been tested in a laboratory. The full list of limits is '
            'in MODEL_CARD.md.')
        with gr.Row():
            wish = gr.Textbox(label='What should the cells look like?', lines=2,
                              value=EXAMPLES[0], scale=4)
            with gr.Column(scale=1):
                top_k = gr.Slider(3, 25, value=10, step=1, label='how many to rank')
                pool = gr.Radio(['unseen scaffolds only', 'all compounds'],
                                value='unseen scaffolds only', label='compound pool')
        go = gr.Button('Ask', variant='primary')
        notes = gr.Markdown()
        table = gr.Dataframe(label='ranked (compound, dose)', wrap=True)
        plot = gr.Plot(label='dose response of the top hit')
        gr.Examples(EXAMPLES, inputs=wish)
        go.click(run, [wish, top_k, pool], [notes, table, plot])
        wish.submit(run, [wish, top_k, pool], [notes, table, plot])
    return demo


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--share', action='store_true')
    ap.add_argument('--port', type=int, default=7860)
    a = ap.parse_args()
    build(Screen(device=a.device)).launch(server_port=a.port, share=a.share)


if __name__ == '__main__':
    main()
