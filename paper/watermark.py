# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Stamp the compiled paper for self-hosted release on drnamlabs.com.

The mark has one job: say where this copy came from and that it is a preprint, without
getting in the way of reading it. So it is a footer line on every page plus one very
faint diagonal, not a bar across the text. A heavy watermark on a research preprint
reads as insecurity, and this document is going to researchers and hiring managers.
"""
import datetime
import os
import sys

import pymupdf

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'paper.pdf')
OUT = os.path.join(HERE, 'paper_drnamlabs.pdf')

GREY = (0.58, 0.58, 0.58)
FAINT = (0.935, 0.935, 0.935)
LEFT = 'Nam H. Le  ·  drnamlabs.com'
RIGHT = 'Preprint · {date} · not peer reviewed'


def main():
    date = datetime.date.today().strftime('%d %B %Y')
    right = RIGHT.format(date=date)
    doc = pymupdf.open(SRC)
    for page in doc:
        w, h = page.rect.width, page.rect.height

        # one faint diagonal, well below the text in contrast. insert_text only rotates
        # by multiples of 90, so the diagonal goes through a TextWriter with a morph.
        pivot = pymupdf.Point(w * 0.16, h * 0.72)
        writer = pymupdf.TextWriter(page.rect, color=FAINT)
        writer.append(pivot, 'drnamlabs.com', fontsize=52,
                      font=pymupdf.Font('helv'))
        writer.write_text(page, morph=(pivot, pymupdf.Matrix(45)), overlay=False)

        y = h - 26
        page.insert_text(pymupdf.Point(58, y), LEFT,
                         fontsize=7.2, fontname='helv', color=GREY)
        tw = pymupdf.get_text_length(right, fontname='helv', fontsize=7.2)
        page.insert_text(pymupdf.Point(w - 58 - tw, y), right,
                         fontsize=7.2, fontname='helv', color=GREY)

    doc.save(OUT, garbage=4, deflate=True)
    print(f'{doc.page_count} pages -> {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB)')
    if len(sys.argv) > 1:          # render a page to eyeball the result
        pymupdf.open(OUT)[int(sys.argv[1])].get_pixmap(dpi=100).save(
            os.path.join(HERE, 'watermark_check.png'))
        print('rendered watermark_check.png')


if __name__ == '__main__':
    main()
