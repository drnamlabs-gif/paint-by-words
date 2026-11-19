# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Where the archive lives, and where we write.

The image archive is NOT redistributed with this repository (see ATTRIBUTION.md).
Point P2I_DATA at a local copy of RxRx3-core, which must contain:

    metadata_rxrx3_core.csv
    CellProfiler_features_rxrx3_core.parquet
    data/                         (optional: the JP2 image shards)
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.environ.get('P2I_DATA', os.path.join(ROOT, 'data', 'rxrx3_core'))
WORK = os.environ.get('P2I_WORK', os.path.join(ROOT, 'work'))
RESULTS = os.path.join(ROOT, 'results')

METADATA = os.path.join(DATA, 'metadata_rxrx3_core.csv')
FEATURES = os.path.join(DATA, 'CellProfiler_features_rxrx3_core.parquet')
SHARDS = os.path.join(DATA, 'data')

GROUPS = os.path.join(WORK, 'groups.parquet')
CELLCOUNT = os.path.join(WORK, 'cellcount.csv')
CRITIC_ATOMS = os.path.join(WORK, 'critic_atoms.parquet')
CRITIC_GROUPS = os.path.join(WORK, 'critic_groups.csv')


def ensure():
    for d in (WORK, RESULTS):
        os.makedirs(d, exist_ok=True)


def require(path, how):
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} is missing. Build it with: {how}')
    return path
