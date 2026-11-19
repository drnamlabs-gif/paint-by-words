# Model card

Limitations first, because they are the part that decides whether this is usable.

## What it cannot do

**It does not design molecules.** It ranks the 1,671 compounds that are already in
the archive, at the eight doses they were already run at. A wish it can satisfy is
one some compound in that library happens to satisfy.

**There is no wet-lab validation.** Every number here is retrospective, computed on
data collected before this work existed. Nothing has been ordered, dosed or imaged
to test a prediction.

**It does not encode direction.** The frozen sentence encoder places opposites as
near-synonyms (`"push the actin out to the edge"` vs `"pull the actin away from the
edge"`, cosine 0.76), so for those wishes the model answers the opposite question
with high confidence. `pbw.language.audit` detects the condition and warns; it does
not fix it. Fine-tuning the encoder's last layers raised top-1 (.53 / .39) but
degraded refusal, so it is not the default. The untried fix is training the encoder
on antonym pairs.

**The language audit cannot detect an out-of-scope wish.** It catches an untrained
behaviour and an unseparated direction, both reliably. It does not catch nonsense:
cosine to the nearest trained phrasing runs 0.41-0.92 for genuine held-out
paraphrases and 0.12-0.53 for deliberately absurd requests, so the distributions
overlap and no threshold separates them. The guard that does work is the model's own
refusal score.

**One behaviour is evaluated but never trained.** "Push the actin out to the cell
edge" reaches only 8 groups in the whole archive, below the training floor, so it is
reported as `rare_test` and the audit warns on it. It is a measure of what the
archive contains, not of what the method can learn.

**"Unseen" means unseen chemistry, not an unseen experiment.** Compounds are held
out by scaffold, but train and test share plates and staining batches. In this
archive 1,661 of 1,671 compounds appear in only one of the four experiments, so
holding out an experiment would hold out its compounds too and the two effects
could not be separated. Batch transfer is therefore **unmeasured**. Measuring it
needs an archive where the same compound is run across sources, such as JUMP
cpg0016.

**The behaviours are eight image features.** "Brighter mitochondria" means the mean
MitoTracker intensity per cell is high. It is not a claim about mitochondrial
biology, mechanism, or anything a clinician would recognise.

**One cell type.** All wells are HUVEC. Nothing here says what transfers to another
cell line.

**Positives are rare and sometimes very rare.** Some held-out wishes have 3-10
satisfying groups in the whole archive. A good AUC on 3 positives is close to a
coin flip dressed as a number; `predict` reports the support count for this reason.

**Absolute probabilities are not calibrated** for compounds outside the archive.
The ranking is the output; the number beside it is a score, not a frequency.

## What it does

Given an English sentence describing a desired appearance of cultured cells, rank
(compound, dose) pairs from RxRx3-core by the probability that they produce it, over
compounds whose Bemis-Murcko scaffold the model never saw, and report that nothing
matches when nothing does.

| on unseen-scaffold compounds | unseen phrasing | unseen combination |
|---|---|---|
| model, AUC | .799 | .861 |
| toxicity-only baseline, AUC | .696 | .789 |
| dose-only baseline, AUC | .722 | .761 |
| model, top-1 | .077 | .105 |
| toxicity-only baseline, top-1 | .077 | **.158** |

Refusal AUC .85 (64 answerable wishes against 37 impossible ones). At a threshold
calibrated on validation compounds: answers 89% of the answerable, wrongly answers
38% of the impossible.

**Top-1 does not yet beat the trivial baseline.** On unseen combinations the
toxicity-only baseline picks a correct single compound more often than the model
(.158 against .105), even though the model orders the library far better (AUC .861
against .789). Ranking skill and single-pick skill are different things here, and
only the first is established.

**Always read the model row against the toxicity row.** Under the earlier `v2`
label definition the model looked much stronger and its toxicity-only baseline
looked almost as strong, i.e. the behaviour-specific contribution nearly vanished.
See README.md for why, and `--definition v2` to reproduce it.

**The refusal statistic is the mean of the top-10 scores, not the best score.** A
maximum is an extreme-value statistic that grows with the size of the compound pool,
so a threshold learned on one pool does not transfer to a larger one and one lucky
group can answer a wish nothing satisfies. Measured: going from 1,256 to 2,670
groups, the median best score for impossible wishes rose .038 to .083 while the
achievable ones did not move; refusal AUC is .85 with the top-10 mean and .79 with
the maximum. 10 is the demo's shortlist length, fixed in advance, not tuned.

## Data

[RxRx3-core](https://huggingface.co/datasets/recursionpharma/rxrx3-core), Recursion.
56,700 query-compound wells, 1,671 compounds, 8 doses (0.0025-10 µM), 6 channels,
HUVEC. Licence and attribution: [ATTRIBUTION.md](ATTRIBUTION.md). The archive is not
redistributed in this repository.

## How it is built

- Labels are **soft**: `p = sigmoid((sign * z - 2.0) / 0.5)`, thresholds fixed
  before any training run and never tuned. Missing measurements are NaN and masked,
  never treated as negatives.
- Per-cell predicates use definition **v3**: `X and not fewer_cells`.
- Compound representation: Morgan count fingerprint + MACCS + RDKit 2D descriptors.
- Sentence encoder: `sentence-transformers/nli-mpnet-base-v2`, frozen.
- Output: `sigmoid(b0 + b1 z) * sigmoid(c0 - softplus(b3) z)`, effect times
  survival, non-monotone in dose.
- Early stopping on validation **compounds**; no test compound touches training or
  model selection. The refusal threshold is calibrated on seen compounds only.

## Quirks in the archive, and how they are handled

- **Dose labels.** A handful of wells are recorded at 0.002, 0.026, 0.26 and 2.6:
  the same eight intended doses with recording noise. Rounding them to two decimals
  relabels 0.025 as 0.02 and invents a ninth dose holding a single group. They are
  snapped to the nearest nominal dose in log space instead.
- **No solvent control.** There is no DMSO-only well, so normalisation is against
  the median of the *query* wells on each plate. Using the EMPTY control wells
  instead produced 12-20% false effects in the pilot.
- **One group per compound-dose, 4 replicate wells.** Group values are medians over
  replicates, so a single bad well does not carry a group.
- **Cell counts exist only where image shards were downloaded.** Missing counts are
  NaN and masked in training; they are never read as "the cells survived".
- **Four experiments, and 1,661 of 1,671 compounds appear in only one of them.**
  This is why batch transfer cannot be measured here, as noted above.

## Approaches tried and rejected

Reported so they are not retried in the dark. Each changed held-out AUC by at most
.01 unless noted.

| tried | outcome |
|---|---|
| PCGrad on per-behaviour gradients | no gain (conflict exists but is near zero: mean cosine ~0, 41-56% of pairs negative) |
| mixture of experts over behaviours | no gain |
| explicit per-behaviour logic heads | no gain |
| Hill-shaped output with a ceiling | no gain |
| auxiliary loss onto OpenPhenom embeddings | no gain |
| ChemBERTa embeddings instead of / with fingerprints | no gain |
| fine-tuning the encoder's last layers | top-1 up, refusal down; rejected |
| separate single-behaviour models | worse than one joint model on unseen compounds |
| factorised shared survival head (`fwin`) | .74 / .81 vs .80 / .85; dropped |
| vision model as the label source | below its own toxicity baseline (−.02 / −.08) |
| single-cell montages for the vision model | alone not significant (+.016, 95% CI [−.03, +.06]); with field crops +.036, CI [+.011, +.057], at ~6M tokens to scale; not adopted |
| best-score (max) as the refusal statistic | pool-size dependent; refusal AUC .79 against .85 for the top-10 mean |
| calibrating the refusal threshold on training compounds | does not transfer: the model fits them closely, so the threshold came out far too strict and answered 16% of answerable wishes. Calibrated on validation compounds instead |

The bottleneck is chemistry to phenotype, not language or capacity: the same model
scores .96 on compounds it has seen and .83 on unseen scaffolds.

## Intended use

Research and method development. Prioritising which compounds from an existing
library to look at first, with the toxicity baseline in view. **Not** for clinical,
diagnostic or therapeutic decisions, and not a source of biological claims.

## Credit

Postdoctoral work at the University of Vermont in Prof. Josh Bongard's lab, funded
by the Army Research Office. Dr. Nam H. Le, drnamlabs.com.
