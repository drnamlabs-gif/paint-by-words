# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Compound representation and the split that decides what "unseen" means.

Splitting compounds at random would leak: two salts of the same molecule, or two
members of one chemical series, would sit on both sides and the reported number
would describe interpolation. We split on the Bemis-Murcko scaffold, the ring
skeleton left after side chains are stripped, so a test compound shares no skeleton
with any training compound.

Representation 'rich' = Morgan count fingerprint (radius 2, 2048 bits, log1p) +
166 MACCS keys + the RDKit 2D descriptor block, winsorised at the 1st/99th
percentile and standardised. Plain 'morgan' is kept for the ablation.
"""
import numpy as np
import pandas as pd


def scaffold_split(smiles, seed=0, fracs=(0.7, 0.8)):
    """{compound: 'train'|'val'|'test'}, whole scaffolds kept together."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog('rdApp.*')
    sc = {}
    for c, s in smiles.items():
        try:
            sc[c] = MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s)) or c
        except Exception:
            sc[c] = c
    groups = pd.Series(sc).groupby(lambda c: sc[c]).groups
    keys = list(groups)
    np.random.default_rng(seed).shuffle(keys)
    n, cum, split = len(smiles), 0, {}
    for k in keys:
        part = 'train' if cum < fracs[0] * n else ('val' if cum < fracs[1] * n else 'test')
        for c in groups[k]:
            split[c] = part
        cum += len(groups[k])
    return split


def morgan(smiles):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog('rdApp.*')
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    out = {}
    for c, s in smiles.items():
        m = Chem.MolFromSmiles(s)
        out[c] = gen.GetFingerprintAsNumPy(m).astype(np.float32) if m is not None else np.zeros(2048, np.float32)
    return out


def rich(smiles):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import MACCSkeys, Descriptors, rdFingerprintGenerator
    RDLogger.DisableLog('rdApp.*')
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    parts = {}
    for c, s in smiles.items():
        m = Chem.MolFromSmiles(s)
        if m is None:
            parts[c] = None
            continue
        cnt = np.log1p(gen.GetCountFingerprintAsNumPy(m).astype(np.float32))
        mac = np.array(list(MACCSkeys.GenMACCSKeys(m)), np.float32)
        desc = []
        for _, fn in Descriptors.descList:
            try:
                desc.append(float(fn(m)))
            except Exception:
                desc.append(np.nan)
        parts[c] = (cnt, mac, np.array(desc, np.float64))
    D = np.stack([v[2] for v in parts.values() if v is not None])
    D = np.where(np.isfinite(D), D, np.nan)
    lo, hi = np.nanpercentile(D, 1, 0), np.nanpercentile(D, 99, 0)
    mu = np.nanmean(np.clip(D, lo, hi), 0)
    sd = np.nanstd(np.clip(D, lo, hi), 0) + 1e-6
    out, width = {}, 2048 + 167 + len(mu)
    for c, v in parts.items():
        if v is None:
            out[c] = np.zeros(width, np.float32)
            continue
        d = np.nan_to_num((np.clip(v[2], lo, hi) - mu) / sd).astype(np.float32)
        out[c] = np.concatenate([v[0], v[1], d])
    return out, dict(lo=lo, hi=hi, mu=mu, sd=sd)


def featurise(smiles, kind='rich'):
    """-> ({compound: vector}, normalisation stats or None). Stats are saved in the
    checkpoint so predict.py can featurise a molecule the model never saw."""
    return (rich(smiles) if kind == 'rich' else (morgan(smiles), None))
