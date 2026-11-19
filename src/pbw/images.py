# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Raw images: counting the cells, and drawing a panel a human or a model can read.

Each well is photographed in six channels at 512x512, 8 bit, stored as JP2 inside
parquet shards. Two jobs happen here.

COUNTING. No CellProfiler column states "the cells died", so it is measured from the
nucleus channel directly: smooth, threshold, label, filter by size. Two estimates:

    n_obj   connected components of 30-2000 px
    n_area  total nuclear foreground divided by the median area of one nucleus

n_area is the one used, because at high density nuclei touch and merge into one
component, so n_obj silently undercounts exactly in the wells that matter. The
well's count is then divided by the median count of the query wells on its own
plate, giving a surviving fraction near 1.0 for an inert treatment.

Validated before use: split-half reproducibility over replicate wells, and a
false-positive rate of 0.5% at the two lowest doses, where almost nothing should be
happening. The 0.6 cut in labels.py was fixed from that measurement.

RENDERING. This is not cosmetic. In an earlier round each image was contrast
stretched against its own percentiles before being shown to the vision model, which
destroyed the comparison: a dim treated well and a bright reference well came out
looking identical, and the model scored at chance on four of eight behaviours. The
fix, and the rule here, is that the brightness scale is computed ONCE from the
reference well and applied unchanged to both, and crops are native resolution with
no resizing. After that fix the same model read four more behaviours at AUC .81 to
1.0. Any change to this function invalidates every vision-model score.
"""
import io
import os

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

CHANNEL_NAMES = ['1 nucleus (Hoechst)', '2 ER (concanavalin A)', '3 actin (phalloidin)',
                 '4 RNA / nucleoli (SYTO14)', '5 mitochondria (MitoTracker)',
                 '6 Golgi + membrane (WGA)']
MIN_NUCLEUS_PX, MAX_NUCLEUS_PX = 30, 2000
SMOOTH_SIGMA = 1.5


def otsu(img):
    """Otsu's threshold: the grey level that best splits the histogram in two."""
    h, _ = np.histogram(img, 256, (0, 256))
    p = h / h.sum()
    w = np.cumsum(p)
    m = np.cumsum(p * np.arange(256))
    between = (m[-1] * w - m) ** 2 / (w * (1 - w) + 1e-9)
    return int(np.argmax(between))


def nuclei_mask(dna):
    smoothed = ndi.gaussian_filter(dna.astype(float), SMOOTH_SIGMA)
    lab, n = ndi.label(smoothed > otsu(smoothed))
    if n == 0:
        return lab, np.zeros(0)
    sizes = ndi.sum(np.ones_like(smoothed), lab, range(1, n + 1))
    return lab, sizes


def count_nuclei(dna):
    """-> (n_obj, n_area). n_area is the estimate used downstream."""
    lab, sizes = nuclei_mask(dna)
    if not len(sizes):
        return 0, 0.0
    ok = (sizes > MIN_NUCLEUS_PX) & (sizes < MAX_NUCLEUS_PX)
    single = np.median(sizes[ok]) if ok.sum() >= 5 else np.nan
    n_area = float(sizes[sizes > MIN_NUCLEUS_PX].sum() / single) if single == single else np.nan
    return int(ok.sum()), n_area


def scale_from(reference_channel, lo=1, hi=99.5):
    """The one brightness scale, taken from the reference well. Never per image."""
    a, b = np.percentile(reference_channel, [lo, hi])
    return float(a), float(b)


def to_grey(channel, a, b):
    return (np.clip((channel - a) / (b - a + 1e-6), 0, 1) * 255).astype(np.uint8)


def band(img, text, h=22):
    out = Image.new('RGB', (img.size[0], img.size[1] + h), 'black')
    out.paste(img, (0, h))
    ImageDraw.Draw(out).text((4, 5), text, fill='white')
    return out


def hcat(imgs, gap=8):
    w = sum(i.size[0] for i in imgs) + gap * (len(imgs) - 1)
    out = Image.new('RGB', (w, max(i.size[1] for i in imgs)), 'black')
    x = 0
    for i in imgs:
        out.paste(i, (x, 0))
        x += i.size[0] + gap
    return out


def vcat(imgs, gap=8):
    h = sum(i.size[1] for i in imgs) + gap * (len(imgs) - 1)
    out = Image.new('RGB', (max(i.size[0] for i in imgs), h), 'black')
    y = 0
    for i in imgs:
        out.paste(i, (0, y))
        y += i.size[1] + gap
    return out


def panel_pair(treated, reference, crop=None, labels=('TREATED', 'REFERENCE')):
    """Six channels of the treated well above the same six of the reference well,
    on one shared brightness scale per channel, taken from the reference."""
    sl = (slice(None), slice(None)) if crop is None else crop
    rows = []
    for x, tag in ((treated, labels[0]), (reference, labels[1])):
        tiles = []
        for c in range(6):
            a, b = scale_from(reference[c])
            tiles.append(band(Image.fromarray(to_grey(x[c][sl], a, b)), CHANNEL_NAMES[c]))
        rows.append(band(hcat(tiles), tag))
    return vcat(rows)


def read_shard_images(shard_path, keys):
    """{key: (6, H, W) uint8} for the requested keys in one parquet shard."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(shard_path)
    present = pf.read(columns=['__key__']).column(0).to_pylist()
    want = {k: i for i, k in enumerate(present) if k in keys}
    if not want:
        return {}
    blobs = pf.read(columns=['jp2']).column(0).take(list(want.values()))
    return {k: np.array(Image.open(io.BytesIO(blobs[j].as_py()['bytes'])))
            for j, k in enumerate(want)}


def _worker(shard):
    out = []
    for key, img in read_shard_images(shard, _KEYS).items():
        out.append((_KEYS[key],) + count_nuclei(img))
    return out


def _init(keys):
    global _KEYS
    _KEYS = keys


def build_cellcount(workers=12):
    """Count nuclei in every compound well of every downloaded shard -> work/cellcount.csv."""
    import glob
    from multiprocessing import Pool

    import pandas as pd

    from . import paths
    from .features import QUERY
    paths.ensure()
    meta = pd.read_csv(paths.require(paths.METADATA, 'see README'), low_memory=False)
    meta = meta[meta.perturbation_type == 'COMPOUND'].copy()
    meta['key'] = (meta.experiment_name + '/Plate' + meta.plate.astype(str)
                   + '/' + meta.address + '_s1_1')          # site 1, nucleus channel
    keys = dict(zip(meta.key, meta.well_id))
    shards = sorted(glob.glob(os.path.join(paths.SHARDS, 'train-*.parquet')))
    if not shards:
        raise FileNotFoundError(f'no image shards in {paths.SHARDS}')
    with Pool(workers, initializer=_init, initargs=(keys,)) as pool:
        rows = [r for part in pool.imap_unordered(_worker, shards) for r in part]

    c = pd.DataFrame(rows, columns=['well_id', 'n_obj', 'n_area'])
    m = meta.merge(c, on='well_id')
    m['plate_id'] = m.experiment_name + '_' + m.plate.astype(str)
    isq = m.well_type_label == QUERY
    for k in ('n_obj', 'n_area'):
        med = m[isq].groupby('plate_id')[k].median()
        m['frac_' + k] = m[k] / m.plate_id.map(med)
    m['frac'] = m.frac_n_area
    cols = ['well_id', 'plate_id', 'treatment', 'concentration',
            'n_obj', 'n_area', 'frac_n_obj', 'frac_n_area', 'frac']
    m[cols].to_csv(paths.CELLCOUNT, index=False)
    return m


if __name__ == '__main__':
    from . import paths as _paths
    _m = build_cellcount()
    print(f'counted {len(_m)} wells -> {_paths.CELLCOUNT}')
