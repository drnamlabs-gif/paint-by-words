# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Ask in English; get compounds, doses, and an evidence card.

    from pbw.predict import Screen
    s = Screen()
    answer = s.ask('make the mitochondria dimmer without killing the cells')

The ranking is the easy part. The evidence card is the point, and it exists because
a bare ranking is indistinguishable from a confident wrong answer. Every reply
carries, next to the model's own score:

  the toxicity-only score for the SAME wish
      the model's prediction for "kill most of the cells", used as a score. If a
      compound ranks high on both, the honest reading is that it was found because
      it is toxic, not because it does what was asked.
  a language warning
      whether the sentence encoder separates this wish from its opposite, and
      whether anything in the training set resembles it at all (see language.py).
  the measured breakdown
      for each ranked group, the actual z-score of each feature named in the wish,
      and the surviving cell fraction. For a compound inside the archive these are
      measurements, so a user can check the model against them directly.
  support
      how many groups in the archive satisfy this wish at all. Three positives
      means the ranking is a lottery with three tickets, whatever the AUC says.
  a refusal
      if no compound passes the calibrated threshold, the answer is that the archive
      contains nothing that does this, and the table is returned as context only.

Doses are the archive's eight concentrations, 0.0025 to 10 micromolar. For a
compound the archive contains, prediction is interpolation in dose; for a molecule
given as SMILES it is extrapolation in chemistry, which is the harder direction and
the one the reported numbers describe.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import torch

from . import chem, paths
from .features import load_groups
from .instructions import load as load_instructions
from .labels import ATOMS, HUMAN, atom_prob
from .evaluate import SHORTLIST, TOXICITY_PROBE
from .language import audit, encode
from .model import Net

DEFAULT_TOP_K = SHORTLIST


class Screen:
    """A seed ensemble, loaded once, answering arbitrary sentences."""

    def __init__(self, labels='numerical', out='window', chem_kind='rich',
                 definition='v3', device='cuda'):
        pat = os.path.join(paths.RESULTS, f'ckpt_{labels}_{out}_{chem_kind}_{definition}_s*.pt')
        files = sorted(glob.glob(pat))
        if not files:
            raise FileNotFoundError(f'no checkpoints matching {pat}; run python -m pbw.train')
        self.device = device
        self.definition = definition
        self.tag = f'{labels}_{out}_{chem_kind}_{definition}'
        self.nets, self.meta = [], None
        for f in files:
            ck = torch.load(f, map_location=device, weights_only=False)
            net = Net(ck['out'], d_chem=ck['d_chem']).to(device)
            net.load_state_dict(ck['state_dict'])
            net.eval()
            self.nets.append(net)
            self.meta = ck

        meta = self.meta
        self.groups = load_groups()
        self.smiles = self.groups.groupby('treatment').SMILES.first().to_dict()
        fp, _ = chem.featurise(self.smiles, meta['chem_kind'])
        self.compounds = meta['compounds']
        self.C = torch.tensor(np.stack([fp[c] for c in self.compounds]), device=device)
        self.cid = {c: i for i, c in enumerate(self.compounds)}
        ld = np.log10(self.groups.dose.to_numpy(float))
        self.Z = torch.tensor(
            ((ld - meta['dose_log10_mean']) / meta['dose_log10_std']).astype(np.float32),
            device=device)
        self.CID = torch.tensor(self.groups.treatment.map(self.cid).to_numpy(), device=device)

        rp = os.path.join(paths.RESULTS, f'refusal_{self.tag}.json')
        self.threshold = json.load(open(rp))['threshold'] if os.path.exists(rp) else None
        self.instructions = load_instructions(definition)

    @torch.no_grad()
    def _score(self, sentence):
        e = torch.tensor(encode([sentence], self.device), device=self.device)
        e = e.expand(len(self.groups), -1)
        p = [net(e, self.C[self.CID], self.Z).cpu().numpy() for net in self.nets]
        return np.mean(p, 0)

    def _support(self, sentence):
        """How many archive groups satisfy the formula of the closest trained wish."""
        a = audit(sentence, self.device)
        hit = [r for r in self.instructions if r['concept'] == a['best_atom']]
        return (hit[0]['n_pos'] if hit else None), a

    def ask(self, sentence, top_k=DEFAULT_TOP_K, scaffold=None):
        p = self._score(sentence)
        tox = self._score(TOXICITY_PROBE)
        support, lang = self._support(sentence)

        g = self.groups.copy()
        g['score'] = p
        g['toxicity_score'] = tox
        if scaffold is not None:
            split = chem.scaffold_split(self.smiles, self.meta['split_seed'])
            g = g[g.treatment.isin({c for c, s in split.items() if s == scaffold})]

        ranked = g.sort_values('score', ascending=False).head(top_k)
        # the refusal statistic is the mean over the shortlist, not the single best
        # score: a maximum grows with the size of the pool, so one lucky group can
        # answer a wish that nothing in the archive satisfies (see evaluate.py)
        shortlist = float(g.score.sort_values(ascending=False).head(SHORTLIST).mean())
        refused = self.threshold is not None and shortlist < self.threshold

        rows = []
        for i, r in ranked.iterrows():
            one = self.groups.loc[[i]]
            measured = {HUMAN[a]: round(float(one[ATOMS[a][0]].iloc[0]), 2)
                        for a in ('elong_up', 'nuc_up', 'dna_up', 'mito_up', 'actin_up')}
            rows.append(dict(
                compound=r.treatment, dose_uM=float(r.dose), p=round(float(r.score), 3),
                p_toxicity_only=round(float(r.toxicity_score), 3),
                surviving_fraction=(None if np.isnan(r.cfrac) else round(float(r.cfrac), 2)),
                cell_loss_p=round(float(atom_prob(one, 'fewer_cells')[0]), 2),
                n_wells=int(r.n_wells), measured_z=measured))

        return dict(
            wish=sentence, refused=refused, threshold=self.threshold,
            shortlist_score=round(shortlist, 4),
            n_archive_groups_satisfying=support,
            language=dict(nearest_trained=lang['nearest'], warnings=lang['warnings']),
            ranking=rows,
            note=('Nothing in this archive clears the calibrated threshold for this wish.'
                  if refused else
                  'Compare p with p_toxicity_only: a compound high on both was probably '
                  'found for being toxic rather than for the requested phenotype.'))

    def dose_curve(self, sentence, compound):
        """The full eight-dose curve, which is where the window shape shows."""
        p = self._score(sentence)
        g = self.groups.assign(score=p)
        x = g[g.treatment == compound].sort_values('dose')
        return pd.DataFrame(dict(dose_uM=x.dose.to_numpy(), p=x.score.to_numpy(),
                                 surviving_fraction=x.cfrac.to_numpy()))


def _print(ans):
    print(f'wish: {ans["wish"]}')
    for w in ans['language']['warnings']:
        print(f'  WARNING  {w}')
    print(f'  archive groups satisfying the nearest trained wish: '
          f'{ans["n_archive_groups_satisfying"]}')
    if ans['refused']:
        print(f'  REFUSED  {ans["note"]}')
    print()
    t = pd.DataFrame([{k: v for k, v in r.items() if k != 'measured_z'} for r in ans['ranking']])
    print(t.to_string(index=False))
    print()
    print(ans['note'] if not ans['refused'] else '')


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('wish', nargs='+')
    ap.add_argument('--top-k', type=int, default=DEFAULT_TOP_K)
    ap.add_argument('--scaffold', default=None, choices=['train', 'val', 'test'],
                    help='restrict to one side of the scaffold split')
    ap.add_argument('--definition', default='v3', choices=['v3', 'v2'])
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args(argv)
    s = Screen(definition=a.definition, device=a.device)
    _print(s.ask(' '.join(a.wish), a.top_k, a.scaffold))


if __name__ == '__main__':
    main()
