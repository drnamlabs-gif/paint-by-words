# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""From the archive to one table: a row per (compound, dose) with eight measurements.

Eight features, each chosen to be the simplest CellProfiler column that states one
of the behaviours in plain language:

    elong   cell eccentricity                how stretched the cell outline is
    nuc     nucleus area                     how big the nucleus is
    dna     mean Hoechst in the nucleus       how brightly the DNA stains
    mito    mean MitoTracker in the cell      how brightly the mitochondria stain
    actin   mean phalloidin in the cell       how brightly the actin stains
    aedge   outer-quartile fraction of actin  how far actin sits from the centre
    gpatch  MAD of the WGA channel            how uneven the Golgi/membrane stain is
    gmean   mean WGA                          overall Golgi brightness (nuisance only)

Normalisation. A plate is a batch: its own staining, its own exposure, its own day.
Each well is scored against the plate it sits on, using the median and the MAD of
the *query-compound* wells on that plate:

    z = (x - median) / (1.4826 * MAD)

Two decisions here were made against measurement, not taste.
  1. The reference is the query wells, not the EMPTY control wells. Benchmarked on
     the pilot, EMPTY as reference produced 12-20% false effects: there is no solvent
     control in this archive, so EMPTY differs from a treated well by more than the
     treatment.
  2. MAD, not standard deviation. The thing we are looking for is the tail, and a
     standard deviation is inflated by the very outliers we want to detect, which
     would shrink their own z-score.

Cell loss is measured separately, from the images, because no single CellProfiler
column states it: counted nuclei in the well over the typical count for the plate.
Groups with no image available get NaN and are masked, never counted as survivors.
"""
import glob
import os

import numpy as np
import pandas as pd

from . import paths

FEATURES = {
    'elong':  'AreaShape_Eccentricity_cell',
    'nuc':    'AreaShape_Area_nucleus',
    'dna':    'Intensity_MeanIntensity_OrigBlue_nucleus',
    'mito':   'Intensity_MeanIntensity_OrigFRed_cell',
    'actin':  'Intensity_MeanIntensity_OrigTexRed_cell',
    'aedge':  'RadialDistribution_MeanFrac_OrigTexRed_4of4_cell',
    'gpatch': 'Intensity_MADIntensity_Ch6_cell',
    'gmean':  'Intensity_MeanIntensity_Ch6_cell',
}

# the six stains, as named in this archive's CellProfiler columns
CHANNELS = {
    'nucleus (Hoechst)': 'OrigBlue',
    'endoplasmic reticulum (concanavalin A)': 'OrigGreen',
    'actin (phalloidin)': 'OrigTexRed',
    'RNA / nucleoli (SYTO14)': 'OrigRed',
    'mitochondria (MitoTracker)': 'OrigFRed',
    'Golgi and membrane (WGA)': 'Ch6',
}

QUERY = 'Query Compounds + Intron control'
MIN_PLATE_WELLS = 30

# the eight nominal concentrations, in micromolar
DOSES = np.array([0.0025, 0.01, 0.025, 0.1, 0.25, 1.0, 2.5, 10.0])


def _dose(concentration):
    """Snap a recorded concentration to the nominal ladder.

    The archive records a handful of wells at 0.002, 0.026, 0.26 and 2.6, which are
    the same eight intended doses with recording noise. Snapping is done by nearest
    neighbour in log space, because the ladder is logarithmic, and because rounding
    to a fixed number of decimals silently relabels 0.025 as 0.02 and invents a
    ninth dose out of the stragglers.
    """
    x = np.log10(np.asarray(concentration, float))
    idx = np.abs(x[:, None] - np.log10(DOSES)[None, :]).argmin(1)
    return pd.Series(DOSES[idx], index=getattr(concentration, 'index', None))


def plate_z(wells, isquery, column):
    """Robust z of one feature against each plate's own query-well distribution."""
    z = pd.Series(np.nan, index=wells.index)
    for _, plate in wells.groupby('plate_id'):
        ref = plate.loc[isquery[plate.index], column].dropna()
        if len(ref) >= MIN_PLATE_WELLS:
            med = ref.median()
            mad = 1.4826 * np.median(np.abs(ref - med)) + 1e-9
            z[plate.index] = (plate[column] - med) / mad
    return z


def build_groups():
    paths.ensure()
    paths.require(paths.METADATA, 'download RxRx3-core and set P2I_DATA (see README)')
    m = pd.read_csv(paths.METADATA, low_memory=False)
    m = m[m.perturbation_type == 'COMPOUND'].copy()
    m['plate_id'] = m.experiment_name + '_' + m.plate.astype(str)
    cp = pd.read_parquet(paths.FEATURES, columns=['well_id'] + list(FEATURES.values()))
    m = m.join(cp.set_index('well_id'), on='well_id')

    isquery = m.well_type_label == QUERY
    for short, column in FEATURES.items():
        m[short] = plate_z(m, isquery, column)

    q = m[isquery & m.SMILES.notna()].copy()
    q['dose'] = _dose(q.concentration)

    counts = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(paths.WORK, 'cellcount*.csv')))]
    if counts:
        frac = pd.concat(counts).drop_duplicates('well_id').set_index('well_id').frac
        q['cfrac'] = q.well_id.map(frac)
    else:
        q['cfrac'] = np.nan

    how = {k: 'median' for k in FEATURES}
    how.update(cfrac='median', well_id='count', SMILES='first', experiment_name='first')
    g = (q.groupby(['treatment', 'dose']).agg(how)
          .rename(columns={'well_id': 'n_wells'}).reset_index())
    g.to_parquet(paths.GROUPS)
    return g


def load_groups():
    paths.require(paths.GROUPS, 'python -m pbw.features')
    return pd.read_parquet(paths.GROUPS).reset_index(drop=True)


def dose_z(groups):
    """log10 dose, standardised. The model sees dose only through this one number."""
    ld = np.log10(groups.dose.to_numpy(float))
    return (ld - ld.mean()) / ld.std(), float(ld.mean()), float(ld.std())


if __name__ == '__main__':
    g = build_groups()
    print(f'{len(g)} groups, {g.treatment.nunique()} compounds, '
          f'{g.cfrac.notna().sum()} with a cell count')
    for k in FEATURES:
        print(f'  {k:7s} z>=2: {(g[k] >= 2).mean():.3f}   z<=-2: {(g[k] <= -2).mean():.3f}')
    print(f'  cell loss (surviving fraction <= 0.6): {(g.cfrac <= 0.6).mean():.3f}')
