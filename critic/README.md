# Layer 3: a vision model as a second measuring instrument

The measurement branch reads eight CellProfiler columns. This branch reads the
images, with a vision-language model in the place of the feature extractor, and
produces labels for exactly the same formulas. Two label sources, one formula logic,
so the comparison is about the instrument and nothing else.

**The headline is negative, and that is the point.** Trained as a label source this
branch ranks unseen compounds *below its own toxicity-only score* (−.02 / −.08 AUC).
A vision model reading these panels delivers toxicity, cell number and gross shape.
It does not deliver subtle per-cell changes in the cells that survive.

## Protocol

**Groups.** 2,000 (compound, dose) groups, from train and validation compounds only,
so the held-out scaffolds stay held out. Stratified: up to 60 per behaviour in round
one, up to 100 per behaviour in round two, plus a random fill, because a uniform
sample of the archive is 95% inert wells and would spend the budget on nothing
happening.

**The well shown.** The replicate well nearest the group median across all eight
features, so the image is representative of the group being labelled rather than its
most extreme well.

**The reference.** A typical low-dose well from the *same plate*, so staining batch,
exposure and day are matched. References whose own cell fraction fell outside
0.85-1.15 were re-picked, since a damaged reference makes every comparison against
it wrong in the same direction.

**The image.** Two 256x256 native-resolution crops from different regions of the
field, each rendered as a TREATED row of six channels above a REFERENCE row of the
same six, with **one brightness scale per channel, taken from the reference**. No
resizing, no per-image contrast stretch. See `pbw.images.panel_pair`.

**The ask.** Seven scores from 0 to 100 with 50 meaning "no different from the
reference", plus a free-text note. Blind identifiers; the agent is told not to open
anything but the batch list and the images. Exact prompt: [PROMPT.md](PROMPT.md).

**Calibration, without labels.** `z = (score - 50) / (1.4826 * MAD(score))`, then the
same `sigmoid((sign * z - 2) / 0.5)` rule and the same fixed threshold as the
measurement branch. The CellProfiler numbers are never used to calibrate, or this
would stop being an independent instrument. See `pbw.critic`.

**Reliability.** Test-retest on a re-scored batch: 0.80-0.93 by axis. Averaging
several runs does not help, so the residual error is systematic rather than noise,
and it is input-limited: the model sees two 256-pixel crops where CellProfiler
measured the whole field.

## Agreement with the measurements

Evaluation only; these numbers never enter training.

| behaviour | AUC |
|---|---|
| cell loss | .93 |
| elongated cells | .87 |
| dimmer DNA | .81 |
| dimmer mitochondria | .75 |
| fainter actin | .74 |
| brighter actin | .73 |
| brighter mitochondria | .69 |
| bigger nuclei | .66 |
| brighter DNA | .59 |
| smaller nuclei | .58 |

The pattern is not random. The model is good at *counting* and at *shape*, mediocre
at *dimmer*, and worst at *brighter* and at *small size changes*. Brightness
comparisons across two panels are exactly where a human eye is also unreliable, and
a few pixels of nuclear diameter are below what a 256-pixel crop resolves.

Three behaviours were deliberately not labelled at all: actin at the cell edge, Golgi
patchiness, and overall Golgi brightness. In the pilot the model was at chance on the
first and read the second through brightness rather than texture, so labelling them
would have manufactured noise with a budget attached.

## What was tried to fix it

**More labels.** Doubling from 1,000 to 2,000 groups gained +.04 and stopped. The
2,000 groups cover **100% of the positives that exist in the archive** for these
behaviours among train and validation compounds; some behaviours have only 31
positive groups in total. The ceiling is the archive, not the labelling budget.

**A better view.** Single-cell montages: 24 nucleus-centred 64x64 tiles per well,
treated grid beside reference grid, shared scale. On 158 pilot groups, montages alone
scored .781 against .765 for field crops, which a bootstrap over groups does not
distinguish (95% CI [−.03, +.06]). Both views together scored .801, which it does
distinguish (+.036, CI [+.011, +.057]). Scaling that to all 2,000 groups would cost
roughly 6M tokens for an estimated +.02 at the system level, so it was not adopted.

**Wording and rendering.** The large win came earlier and was not about the model at
all: fixing the per-image contrast stretch moved four of eight behaviours from chance
to AUC .81-1.0.

## Files

```
critic/PROMPT.md              the exact scoring prompt
pbw/critic.py                 label-free calibration and the agreement report
pbw/images.py:panel_pair      the rendering, which must not be changed casually
work/critic_groups.csv        blind id -> (compound, dose, well shown, reference well)
work/critic_scores/*.json     the raw scores, one file per batch of 50
work/critic_atoms.parquet     calibrated atom probabilities, the training input
```

To train on this label source instead of the measurements:

```bash
python -m pbw.critic                       # calibrate and report agreement
python -m pbw.train --labels critic
python -m pbw.train --labels numsub        # the control: same groups, measured labels
python -m pbw.evaluate --labels critic
```

`numsub` is the comparison that makes the result interpretable. It trains on the
measurements but masked down to exactly the groups the vision model saw, so the
difference between `critic` and `numsub` is the instrument, not the sample size.
