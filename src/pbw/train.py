# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Fit the predictor, and save enough to answer a sentence it has never seen.

How the fit works, step by step.

  1. Every (instruction, compound, dose) cell of the grid gets a SOFT target in
     [0, 1] from labels.py. Cells whose measurement is missing are dropped, not
     imputed.
  2. Each behaviour ("concept") is sampled as its own task, with equal weight, so a
     common behaviour such as cell loss cannot drown a rare one such as elongation
     just by appearing more often.
  3. The loss is cross-entropy against those soft targets.
  4. AdamW, one step per batch; validation loss on held-out VAL compounds after each
     epoch; the parameters from the best epoch are kept. No test compound is ever
     seen, by the model or by the early-stopping rule.

Interference between behaviours was measured rather than assumed: the cosine between
per-behaviour gradients on the shared trunk is about 0 with 41-56% of pairs negative,
yet one joint model still beats separate single-behaviour models on unseen compounds.
PCGrad, a mixture of experts and explicit logic heads were each tried and each
changed AUC by at most .01, so the plain joint model is the default. Pass
--grad-conflict to recompute that matrix.

Labels:
  numerical   the CellProfiler measurements (the default; see labels.py)
  critic      the vision-model scores (see critic/README.md)
  numsub      the measurements, but masked down to exactly the groups the vision
              model scored. This is the control that makes the critic comparison
              fair: same groups, different label source.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from . import chem, paths
from .features import dose_z, load_groups
from .instructions import load as load_instructions
from .labels import combine, formula_prob
from .language import ENCODER, encode
from .model import Net, bce

BATCH_PER_TASK = 512


def critic_labels(groups, instructions, definition):
    """Soft labels from the vision model's calibrated scores, same formula logic."""
    paths.require(paths.CRITIC_ATOMS, 'see critic/README.md')
    ca = pd.read_parquet(paths.CRITIC_ATOMS)
    gc = groups[['treatment', 'dose']].merge(ca, on=['treatment', 'dose'], how='left')
    return np.stack([combine(lambda n: gc[n].to_numpy(float), r['lits'], definition)
                     for r in instructions])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default='window', choices=['window', 'beta'])
    ap.add_argument('--labels', default='numerical', choices=['numerical', 'critic', 'numsub'])
    ap.add_argument('--definition', default='v3', choices=['v3', 'v2'])
    ap.add_argument('--chem', default='rich', choices=['rich', 'morgan'])
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--split-seed', type=int, default=0,
                    help='scaffold split seed; keep fixed across model seeds')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--grad-conflict', action='store_true')
    a = ap.parse_args(argv)
    paths.ensure()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = a.device
    tag = (f'{a.labels}_{a.out}_{a.chem}_{a.definition}_s{a.seed}')

    groups = load_groups()
    instructions = load_instructions(a.definition)
    smiles = groups.groupby('treatment').SMILES.first().to_dict()
    compounds = sorted(smiles)
    cid = {c: i for i, c in enumerate(compounds)}
    split = chem.scaffold_split(smiles, seed=a.split_seed)
    groups['csplit'] = groups.treatment.map(split)
    groups['cid'] = groups.treatment.map(cid)
    zd, zd_mean, zd_std = dose_z(groups)
    groups['zd'] = zd

    # truth = the measurements, always, because that is what we score against
    truth = np.stack([formula_prob(groups, r['lits'], a.definition) for r in instructions])
    target = truth.copy()
    if a.labels != 'numerical':
        seen = np.zeros(len(groups), bool)
        cg = pd.read_csv(paths.require(paths.CRITIC_GROUPS, 'see critic/README.md'),
                         keep_default_na=False, na_values=[''])
        seen[cg.gidx.to_numpy()] = True
        if a.labels == 'numsub':
            target[:, ~seen] = np.nan
        else:
            target = critic_labels(groups, instructions, a.definition)

    E = torch.tensor(encode([r['text'] for r in instructions], dev), device=dev)
    fp, chem_stats = chem.featurise(smiles, a.chem)
    C = torch.tensor(np.stack([fp[c] for c in compounds]), device=dev)
    Z = torch.tensor(groups.zd.to_numpy(np.float32), device=dev)
    CID = torch.tensor(groups.cid.to_numpy(), device=dev)
    T = torch.tensor(np.nan_to_num(target), dtype=torch.float32, device=dev)

    train_i = [i for i, r in enumerate(instructions) if r['split'] == 'train']
    train_g = np.where(groups.csplit == 'train')[0]
    val_g = np.where(groups.csplit == 'val')[0]
    test_g = np.where(groups.csplit == 'test')[0]

    rows_of = {}
    for concept in sorted({instructions[i]['concept'] for i in train_i}):
        ii = [i for i in train_i if instructions[i]['concept'] == concept]
        I, G = (x.ravel() for x in np.meshgrid(ii, train_g, indexing='ij'))
        ok = ~np.isnan(target[I, G])
        if ok.any():
            rows_of[concept] = (torch.tensor(I[ok], device=dev), torch.tensor(G[ok], device=dev))
    tasks = sorted(rows_of)
    atoms = [t for t in tasks if '+' not in t]

    model = Net(a.out, d_chem=C.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def task_loss(concept, n):
        I, G = rows_of[concept]
        b = torch.randint(0, len(I), (n,), device=dev)
        i, g = I[b], G[b]
        return bce(model(E[i], C[CID[g]], Z[g]), T[i, g])

    @torch.no_grad()
    def predict(gidx, idx):
        model.eval()
        g = torch.tensor(gidx, device=dev)
        out = np.stack([model(E[i].expand(len(gidx), -1), C[CID[g]], Z[g]).cpu().numpy()
                        for i in idx])
        model.train()
        return out

    def val_loss():
        p = predict(val_g, train_i)
        y = target[np.ix_(train_i, val_g)]
        m = ~np.isnan(y)
        p, y = np.clip(p[m], 1e-6, 1 - 1e-6), y[m]
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    def grad_conflict():
        shared = (list(model.text.parameters()) + list(model.chem.parameters())
                  + list(model.trunk.parameters()))
        gs = []
        for concept in atoms:
            model.zero_grad()
            task_loss(concept, 4096).backward()
            gs.append(torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
                                 for p in shared]).clone())
        G = torch.nn.functional.normalize(torch.stack(gs), dim=1)
        return (G @ G.T).cpu().numpy()

    total = sum(len(v[0]) for v in rows_of.values())
    per_epoch = max(1, total // (BATCH_PER_TASK * len(tasks)))
    best, best_state, t0, conflict = 1e9, None, time.time(), []
    for epoch in range(1, a.epochs + 1):
        for _ in range(per_epoch):
            loss = sum(task_loss(t, BATCH_PER_TASK) for t in tasks) / len(tasks)
            opt.zero_grad()
            loss.backward()
            opt.step()
        v = val_loss()
        if v < best:
            best = v
            best_state = {k: x.detach().clone() for k, x in model.state_dict().items()}
        if a.grad_conflict and epoch in (1, a.epochs // 2, a.epochs):
            cm = grad_conflict()
            off = cm[~np.eye(len(atoms), dtype=bool)]
            conflict.append(dict(epoch=epoch, mean_cos=float(off.mean()),
                                 frac_neg=float((off < 0).mean())))
    model.load_state_dict(best_state)

    hard = truth > 0.5
    preds, res = {}, []
    # val_comp is kept so the refusal threshold can be calibrated on compounds that
    # are neither trained on nor tested on. Calibrating it on TRAIN compounds does
    # not transfer: the model fits them closely, so its confidence there is higher
    # than it will ever be on new chemistry, and the threshold comes out far too
    # strict (measured: it answered 16% of answerable wishes).
    for pool, gidx in (('seen_comp', train_g), ('val_comp', val_g), ('unseen_comp', test_g)):
        P = predict(gidx, range(len(instructions)))
        preds[pool] = P.astype(np.float16)
        preds[pool + '_idx'] = gidx
        for i, r in enumerate(instructions):
            known = ~np.isnan(truth[i, gidx])
            y, p = hard[i, gidx][known], P[i][known]
            n_pos = int(y.sum())
            res.append(dict(pool=pool, split=r['split'], concept=r['concept'], text=r['text'],
                            n_pos=n_pos, base_rate=n_pos / max(len(y), 1),
                            auc=roc_auc_score(y, p) if 0 < n_pos < len(y) else np.nan,
                            top1_hit=bool(y[int(np.argmax(p))]) if len(y) else False,
                            max_p=float(p.max()) if len(y) else np.nan))

    torch.save(dict(state_dict=best_state, out=a.out, d_chem=int(C.shape[1]),
                    chem_kind=a.chem, chem_stats=chem_stats, encoder=ENCODER,
                    definition=a.definition, labels=a.labels, seed=a.seed,
                    split_seed=a.split_seed, dose_log10_mean=zd_mean, dose_log10_std=zd_std,
                    compounds=compounds, val_loss=best),
               os.path.join(paths.RESULTS, f'ckpt_{tag}.pt'))
    np.savez_compressed(os.path.join(paths.RESULTS, f'pred_{tag}.npz'), **preds)
    R = pd.DataFrame(res)
    R['tag'] = tag
    R.to_csv(os.path.join(paths.RESULTS, f'eval_{tag}.csv'), index=False)
    if conflict:
        pd.DataFrame(conflict).to_csv(os.path.join(paths.RESULTS, f'gradcos_{tag}.csv'), index=False)

    s = (R[(R.n_pos > 0) & R.split.isin(['para_test', 'comp_test'])]
         .groupby(['split', 'pool'])[['auc', 'top1_hit']].mean().round(3))
    print(f'{tag}  {time.time() - t0:.0f}s  val {best:.4f}')
    print(s.to_string())
    if conflict:
        print('gradient conflict', conflict[-1])
    return tag


if __name__ == '__main__':
    main()
