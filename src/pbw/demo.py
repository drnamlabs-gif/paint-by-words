# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""The README animation: a wish typed, answered, and then checked against the images.

It is drawn frame by frame rather than screen-recorded, so `make gif` reproduces it
and the numbers in it are whatever the current checkpoints actually predict.

The three acts are the argument of the whole project:

  1. a wish is typed and the ranking appears, with the toxicity column beside the
     model's own score;
  2. the top-ranked well is shown next to its reference, and it is a well whose
     cells are dying, which is exactly what its toxicity score said;
  3. a lower-ranked well is shown, scoring far lower on toxicity, with its cells
     intact.

So the animation ends on the point the evidence card exists for: the ranking alone
would have sent you to the toxic compound, and the column beside it is what sends
you past it.
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import paths
from .images import scale_from, to_grey

W, H = 1000, 600
BG, FG, DIM = (14, 15, 17), (232, 232, 230), (138, 140, 145)
GREEN, RED, BLUE, AMBER = (60, 200, 130), (235, 85, 80), (90, 150, 230), (235, 170, 60)
PAD = 26
WISH = 'make the mitochondria dimmer without killing the cells'

MONO = ['C:/Windows/Fonts/consola.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',
        '/System/Library/Fonts/Menlo.ttc']
MONO_BOLD = ['C:/Windows/Fonts/consolab.ttf',
             '/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf']


def _font(size, bold=False):
    for p in (MONO_BOLD if bold else MONO):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size)


def _canvas():
    im = Image.new('RGB', (W, H), BG)
    return im, ImageDraw.Draw(im)


def _title(d, text, sub=None):
    d.text((PAD, 20), text, font=_font(20, True), fill=FG)
    if sub:
        d.text((PAD, 48), sub, font=_font(14), fill=DIM)


def _terminal_frames(rows, wish):
    """Act 1: the command types itself, then the table lands row by row."""
    f, fb = _font(17), _font(17, True)
    prompt = f'$ pbw "{wish}"'
    frames = []

    for n in range(0, len(prompt) + 1, 3):
        im, d = _canvas()
        _title(d, 'paint-by-words',
               'ask a cell-image archive for a phenotype in plain English')
        d.text((PAD, 110), prompt[:n], font=f, fill=GREEN)
        if n < len(prompt):
            d.rectangle([PAD + 10 * n, 110, PAD + 10 * n + 9, 130], fill=GREEN)
        frames.append((im, 55))

    header = f'{"compound":<18}{"dose":>7}   {"p(wish)":>8}   {"p(toxicity)":>12}   {"cells left":>11}'
    lines = []
    for r in rows:
        left = '-' if r['surviving_fraction'] is None else f'{r["surviving_fraction"]:.2f}'
        lines.append((f'{r["compound"][:17]:<18}{r["dose_uM"]:>6g}u   {r["p"]:>8.3f}   '
                      f'{r["p_toxicity_only"]:>12.3f}   {left:>11}',
                      r['p_toxicity_only'] > r['p']))

    for k in range(len(lines) + 1):
        im, d = _canvas()
        _title(d, 'paint-by-words',
               'ask a cell-image archive for a phenotype in plain English')
        d.text((PAD, 110), prompt, font=f, fill=GREEN)
        d.text((PAD, 156), header, font=fb, fill=DIM)
        d.line([PAD, 180, W - PAD, 180], fill=(45, 47, 52))
        for i, (line, toxic) in enumerate(lines[:k]):
            y = 192 + i * 26
            d.text((PAD, y), line, font=f, fill=RED if toxic else FG)
        frames.append((im, 70 if k < len(lines) else 900))

    im, d = _canvas()
    _title(d, 'paint-by-words',
           'ask a cell-image archive for a phenotype in plain English')
    d.text((PAD, 110), prompt, font=f, fill=GREEN)
    d.text((PAD, 156), header, font=fb, fill=DIM)
    d.line([PAD, 180, W - PAD, 180], fill=(45, 47, 52))
    for i, (line, toxic) in enumerate(lines):
        d.text((PAD, 192 + i * 26), line, font=f, fill=RED if toxic else FG)
    y = 192 + len(lines) * 26 + 22
    d.text((PAD, y), 'red = scores higher on "kill the cells" than on the wish itself.',
           font=_font(16, True), fill=RED)
    d.text((PAD, y + 26), 'That column is printed with every answer. Here is why.',
           font=_font(16), fill=DIM)
    d.text((PAD, H - 46), 'ranked over 1,671 compounds x 8 doses, scaffolds the model '
                          'never saw, 5-seed ensemble', font=_font(14), fill=DIM)
    frames.append((im, 2200))
    return frames


TILE, GAP = 200, 14
COL = PAD + 2 * (TILE + GAP) + 28          # the right-hand column of labels and numbers
ROW_Y = (100, 326)                          # top of the treated row, top of the reference row


def _well_panel(treated, reference, channels=(0, 2), size=TILE):
    """Treated above reference, one brightness scale per channel from the reference."""
    tiles = []
    for row in (treated, reference):
        strip = []
        for c in channels:
            a, b = scale_from(reference[c])
            strip.append(Image.fromarray(to_grey(row[c], a, b)).resize((size, size)))
        tiles.append(strip)
    return tiles


def _image_frames(case, hold=2600):
    """Acts 2 and 3: the real wells behind one row of the table."""
    tiles = _well_panel(case['treated'], case['reference'])
    im, d = _canvas()
    _title(d, case['title'], case['subtitle'])

    for j, name in enumerate(('nuclei (Hoechst)', 'actin (phalloidin)')):
        d.text((PAD + j * (TILE + GAP), ROW_Y[0] - 20), name, font=_font(13), fill=DIM)
    for i, y in enumerate(ROW_Y):
        for j in range(2):
            im.paste(tiles[i][j], (PAD + j * (TILE + GAP), y))

    d.text((COL, ROW_Y[0]), 'TREATED', font=_font(18, True), fill=case['colour'])
    for k, (label, value, colour) in enumerate(case['stats']):
        y = ROW_Y[0] + 36 + k * 54
        d.text((COL, y), label, font=_font(13), fill=DIM)
        d.text((COL, y + 18), value, font=_font(22, True), fill=colour)

    d.text((COL, ROW_Y[1]), 'REFERENCE', font=_font(18, True), fill=DIM)
    d.text((COL, ROW_Y[1] + 28), 'a typical well from the same', font=_font(13), fill=DIM)
    d.text((COL, ROW_Y[1] + 46), 'plate, on the same brightness', font=_font(13), fill=DIM)
    d.text((COL, ROW_Y[1] + 64), 'scale: what you see is real', font=_font(13), fill=DIM)

    d.text((PAD, ROW_Y[1] + TILE + 24), case['verdict'], font=_font(17, True),
           fill=case['colour'])
    return [(im, hold)]


def dose_gif(compound='ABT-263', channels=(0, 2), size=300, hold=750):
    """The second animation: one compound climbing its own dose ladder.

    Nothing is modelled here. It is the archive playing itself, on a single
    brightness scale taken from the lowest dose, which is why the cells visibly
    thin out rather than the picture merely getting darker.
    """
    from .features import load_groups
    from .figures import _load_images, _stack, _wells_for
    series, reference = _wells_for(compound)
    found = _load_images(list(series.itertuples()) + [reference])
    ref = _stack(found, reference)
    groups = load_groups()
    cfrac = (groups[groups.treatment == compound].set_index('dose').cfrac.to_dict())

    doses = series.dose.tolist()
    lo, hi = min(doses), max(doses)
    names = ('nuclei (Hoechst)', 'actin (phalloidin)')
    frames = []
    for row in series.itertuples():
        x = _stack(found, row)
        im, d = _canvas()
        _title(d, f'{compound}, the same well type at eight doses',
               'one brightness scale, taken from the lowest dose, so the change is real')
        for j, c in enumerate(channels):
            a, b = scale_from(ref[c])
            im.paste(Image.fromarray(to_grey(x[c], a, b)).resize((size, size)),
                     (PAD + j * (size + GAP), 110))
            d.text((PAD + j * (size + GAP), 90), names[j], font=_font(13), fill=DIM)

        cx = PAD + 2 * (size + GAP) + 28
        d.text((cx, 110), 'dose', font=_font(13), fill=DIM)
        d.text((cx, 128), f'{row.dose:g} uM', font=_font(26, True), fill=AMBER)
        frac = cfrac.get(row.dose)
        d.text((cx, 186), 'cells left, against a typical', font=_font(13), fill=DIM)
        d.text((cx, 204), 'well on the same plate', font=_font(13), fill=DIM)
        value = '-' if frac is None or frac != frac else f'{frac:.2f}'
        colour = GREEN if (frac or 1) > 0.6 else RED
        d.text((cx, 224), value, font=_font(26, True), fill=colour)

        # the ladder, so the viewer knows where in the series this frame sits
        y = 300
        d.text((cx, y - 22), 'where this frame sits', font=_font(13), fill=DIM)
        for k, dv in enumerate(doses):
            yy = y + k * 22
            on = dv == row.dose
            d.rectangle([cx, yy, cx + (160 if on else 90), yy + 12],
                        fill=AMBER if on else (48, 50, 55))
            d.text((cx + 176, yy - 2), f'{dv:g}', font=_font(12),
                   fill=FG if on else DIM)
        frames.append((im, hold * (2 if row.dose in (lo, hi) else 1)))

    out = os.path.join(paths.ROOT, 'figs', '02_dose_series.gif')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    images = [i.convert('P', palette=Image.ADAPTIVE, colors=128) for i, _ in frames]
    images[0].save(out, save_all=True, append_images=images[1:],
                   duration=[ms for _, ms in frames], loop=0, optimize=True)
    return out, len(frames), os.path.getsize(out)


def _fetch(compound, dose):
    from .figures import _load_images, _stack, _wells_for
    series, reference = _wells_for(compound)
    row = series.iloc[int(np.abs(series.dose - dose).argmin())]
    found = _load_images([row, reference])
    return _stack(found, row), _stack(found, reference), float(row.dose)


def build(device='cuda'):
    from .predict import Screen
    screen = Screen(device=device)
    answer = screen.ask(WISH, top_k=5, scaffold='test')
    rows = answer['ranking']

    toxic = next((r for r in rows if r['p_toxicity_only'] > r['p']
                  and (r['surviving_fraction'] or 1) < 0.6), rows[0])
    clean = next((r for r in rows if r['p_toxicity_only'] < r['p'] / 1.5
                  and (r['surviving_fraction'] or 0) > 0.85), rows[-1])

    frames = _terminal_frames(rows, WISH)
    cases = [(toxic, RED, 'The ranking alone would have sent you here.'),
             (clean, GREEN, 'Same wish, same ranking, one column further right.')]
    for r, colour, verdict in cases:
        title = f'#{rows.index(r) + 1} in the ranking: {r["compound"]}'
        t, ref, dose = _fetch(r['compound'], r['dose_uM'])
        surviving = '-' if r['surviving_fraction'] is None else f'{r["surviving_fraction"]:.2f}'
        frames += _image_frames(dict(
            treated=t, reference=ref, title=title,
            subtitle=f'{dose:g} micromolar, beside a typical well from the same plate, '
                     f'on one brightness scale',
            colour=colour, verdict=verdict,
            stats=[('p(wish)', f'{r["p"]:.3f}', BLUE),
                   ('p(toxicity)', f'{r["p_toxicity_only"]:.3f}',
                    RED if r['p_toxicity_only'] > r['p'] else GREEN),
                   ('cells left', surviving,
                    RED if float(surviving) < 0.6 else GREEN)]))

    out = os.path.join(paths.ROOT, 'figs', '00_demo.gif')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    images = [im.convert('P', palette=Image.ADAPTIVE, colors=128) for im, _ in frames]
    images[0].save(out, save_all=True, append_images=images[1:],
                   duration=[ms for _, ms in frames], loop=0, optimize=True)
    return out, len(frames), os.path.getsize(out)


if __name__ == '__main__':
    for fn in (build, dose_gif):
        path, n, size = fn()
        print(f'{n} frames, {size / 1e6:.1f} MB -> {path}')
