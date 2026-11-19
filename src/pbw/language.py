# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""The language channel, and an honest warning about where it breaks.

Sentences are embedded by a frozen general-purpose sentence encoder
(nli-mpnet-base-v2) and never fine-tuned by default. Nothing about cells is
learned here; the encoder only has to place two phrasings of the same wish near
each other, and the small network learns the rest.

That works for paraphrase and fails for direction. Measured, in this archive:

    "push the actin out to the edge"  vs  "pull the actin away from the edge"  0.76
    "make the Golgi smoother"         vs  "patchy Golgi"                       0.65

Those pairs are opposites, and the encoder places them as near-synonyms, because
the words they share outnumber the one word that reverses the meaning. When the
instruction is one of these, the model answers the OPPOSITE instruction with high
confidence, which is the worst failure mode a screening tool can have: wrong and
sure of itself.

So this module does not just embed. `audit` checks every query against that failure
and returns a warning the caller is expected to show the user. A silent wrong answer
is a bug; a flagged wrong answer is a limitation.

`audit` reliably catches two things: a wish whose behaviour was never trained (the
archive has too few examples of it), and a direction the encoder cannot separate
from its opposite. It does NOT reliably catch an out-of-scope wish. Measured on this
instruction set, the cosine to the nearest trained phrasing runs 0.41 to 0.92 for
genuine held-out paraphrases and 0.12 to 0.53 for deliberately absurd requests, so
the two overlap and no threshold separates them. The FAR_FROM_TRAINED check below is
therefore a hint, not a filter; the real guard against an unanswerable wish is the
model's own refusal score in evaluate.py, which is calibrated and measured.
"""
import numpy as np

from .instructions import OPPOSITES, PHRASINGS, TRAINED_ATOMS

ENCODER = 'sentence-transformers/nli-mpnet-base-v2'
AMBIGUOUS_MARGIN = 0.05   # opposite this close to the best match is not separated
FAR_FROM_TRAINED = 0.45   # a hint only: genuine paraphrases reach down to 0.41

_model = {}


def encoder(device='cuda'):
    if device not in _model:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(ENCODER, device=device)
        for p in m.parameters():
            p.requires_grad = False
        m.eval()
        _model[device] = m
    return _model[device]


def encode(texts, device='cuda'):
    """-> (n, 768) float32, L2-normalised, so a dot product is a cosine."""
    e = encoder(device).encode(list(texts), convert_to_numpy=True,
                               normalize_embeddings=True, show_progress_bar=False)
    return e.astype(np.float32)


def audit(sentence, device='cuda'):
    """Compare one query against the phrasings, and say what could go wrong.

    Two comparisons, not one. The query is matched against every behaviour, to see
    what it is really asking for, and separately against only the behaviours the
    model was actually TRAINED on, which is what it will answer with. The gap
    between those two is where confident wrong answers come from.

    -> dict(nearest=[(phrasing, atom, cosine), ...], best_atom=..., warnings=[...])
    """
    atoms = [a for a, ps in PHRASINGS.items() for _ in ps[:3]]
    texts = [p for ps in PHRASINGS.values() for p in ps[:3]]
    emb = encode([sentence] + texts, device)
    sims = emb[1:] @ emb[0]
    trained = np.array([a in TRAINED_ATOMS for a in atoms])

    def best(mask):
        idx = np.where(mask)[0]
        j = idx[int(np.argmax(sims[idx]))]
        return atoms[j], texts[j], float(sims[j])

    any_atom, any_text, any_sim = best(np.ones(len(atoms), bool))
    best_atom, best_text, best_sim = best(trained)
    order = np.argsort(-np.where(trained, sims, -np.inf))
    nearest = [(texts[i], atoms[i], float(sims[i])) for i in order[:5]]

    warnings = []
    if any_sim < FAR_FROM_TRAINED:
        warnings.append(
            f'This wish resembles nothing the model has seen (best cosine {any_sim:.2f}, '
            f'to "{any_text}"). Treat the ranking as untested.')
    if any_atom not in TRAINED_ATOMS:
        extra = (' That trained concept is its OPPOSITE.'
                 if OPPOSITES.get(any_atom) == best_atom else '')
        warnings.append(
            f'The behaviour this wish is closest to ("{any_text}") was never trained: the '
            f'archive has too few examples of it. The model will answer with the nearest '
            f'trained concept instead, "{best_text}" (cosine {best_sim:.2f}).{extra}')
    opposite = OPPOSITES.get(best_atom)
    if opposite is not None and opposite in TRAINED_ATOMS:
        opp = max(float(sims[i]) for i, a in enumerate(atoms) if a == opposite)
        if opp >= best_sim - AMBIGUOUS_MARGIN:
            warnings.append(
                f'The sentence encoder does not separate this direction from its opposite '
                f'({best_atom} {best_sim:.2f} vs {opposite} {opp:.2f}). The ranking below '
                f'may answer the opposite wish. This is a known, unfixed failure.')
    return dict(nearest=nearest, best_atom=best_atom, best_sim=best_sim,
                closest_any=any_atom, warnings=warnings)
