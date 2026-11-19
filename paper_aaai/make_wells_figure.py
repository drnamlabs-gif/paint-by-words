# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""One column-width figure showing the confound as a photograph rather than a claim.

A reader of an evaluation workshop paper may never have seen a Cell Painting well. The
argument here is that a dying cell reads brighter per unit area, which is a sentence
until you put the treated well next to its reference on one brightness scale, and then
it is obvious: fewer nuclei, and the ones left are bright and rounded.

Two channels, two rows, native resolution, a single brightness scale per channel taken
from the reference, exactly as everywhere else in this project.
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'src'))
from pbw.demo import _fetch                      # noqa: E402
from pbw.images import scale_from, to_grey       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'figures', '01_wells.png')

COMPOUND, DOSE = 'Perphenazine', 10.0
TILE, GAP, LABEL = 300, 10, 26
CHANNELS = ((0, 'nuclei (Hoechst)'), (2, 'actin (phalloidin)'))


def main():
    treated, reference, dose = _fetch(COMPOUND, DOSE)
    w = 2 * TILE + GAP
    h = 2 * (TILE + LABEL) + GAP
    canvas = Image.new('RGB', (w, h), 'black')
    draw = ImageDraw.Draw(canvas)

    for row, (img, tag) in enumerate(((treated, f'TREATED  {COMPOUND}, {dose:g} uM'),
                                      (reference, 'REFERENCE  typical well, same plate'))):
        y = row * (TILE + LABEL + GAP)
        draw.text((2, y + 7), tag, fill='white')
        for col, (c, name) in enumerate(CHANNELS):
            a, b = scale_from(reference[c])
            tile = Image.fromarray(to_grey(img[c], a, b)).resize((TILE, TILE))
            canvas.paste(tile, (col * (TILE + GAP), y + LABEL))
            if row == 0:
                draw.text((col * (TILE + GAP) + 2, y + LABEL + 2), name, fill='white')

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    canvas.save(OUT)
    print(f'{canvas.size} -> {OUT} ({os.path.getsize(OUT) / 1e3:.0f} KB)')


if __name__ == '__main__':
    main()
