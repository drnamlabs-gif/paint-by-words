# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""paint-by-words: ask a cell-image archive for a phenotype in plain English.

Layers, each with its own module and its own measured check:

    images.py        raw six-channel wells -> nuclei counts, readable panels
    features.py      wells -> eight measurements, normalised per plate
    labels.py        measurements -> soft probability of a stated behaviour
    instructions.py  English sentences <-> formulas, and the four splits
    language.py      sentence -> embedding, plus the direction-failure audit
    model.py         (sentence, compound) -> dose-response curve
    train.py         the fit
    evaluate.py      AUC, top-1, refusal, and the two trivial baselines
    predict.py       a wish in, a ranked answer and an evidence card out
"""
__version__ = '0.1.0'
