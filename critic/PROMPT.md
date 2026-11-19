# The scoring prompt, verbatim

This is the exact text each vision-model agent was given, one agent per batch of 50
groups. `{batch}` and `{out}` are the only substitutions. It is reproduced here
unchanged, including the parts that turned out to matter: the statement that both
rows share one brightness scale, the instruction to use the full range in either
direction, and the ban on inferring anything from file names.

The agents were Claude agents with file-read access and nothing else. They never saw
the compound, the dose, the plate, or any measurement: group identifiers are blind
(`G0000`...) and the mapping lives in `critic_groups.csv`, which the agents were
told not to open.

---

```
You are scoring microscopy images. Work only from what you see. You do not know what
treatment, if any, each well received, and you must not guess or look anything up.

Read the item list at {batch}. Each item has an `id` and an `image` path. Use the Read
tool on each image path to view it. Do not open any other file.

Each image has FOUR rows of six greyscale panels, each a 256x256 crop at full
resolution. Rows 1-2 are one region of the well, rows 3-4 a second region:
- rows labelled TREATED: one well of cultured human cells (HUVEC).
- rows labelled REFERENCE: a typical well from the same plate, imaged identically,
  same region. This is your baseline.

The six panels are one per fluorescence channel, labelled above each panel:
1 nuclei (Hoechst), 2 ER (ConA), 3 actin (phalloidin), 4 RNA/nucleoli (SYTO14),
5 mitochondria (MitoTracker), 6 Golgi/membrane (WGA).

Important: within each channel, TREATED and REFERENCE panels are displayed on the SAME
brightness scale. So if a treated panel looks brighter or dimmer than the reference
panel for the same channel, that difference is real. Judge from both regions together.

For EVERY item compare TREATED against REFERENCE and give seven scores from 0 to 100,
where 50 means "no different from the reference":

- `fewer_cells`: there are fewer cells (nuclei in panel 1) than in the reference
  (100 = far fewer, 0 = far more)
- `dna_bright`: nuclei in panel 1 are brighter (more intensely stained) than in the
  reference (100 = much brighter, 0 = much dimmer)
- `bigger_nuclei`: individual nuclei in panel 1 are larger than in the reference
  (100 = much larger, 0 = much smaller)
- `mito_bright`: the signal in panel 5 is brighter than in the reference
  (100 = much brighter, 0 = much dimmer)
- `actin_bright`: the signal in panel 3 is brighter than in the reference
  (100 = much brighter, 0 = much dimmer)
- `elongated`: cell bodies look more stretched or spindle shaped than in the reference,
  judged from panels 3 and 6 (100 = much more elongated, 0 = much rounder)
- `any_change`: how different the treated well looks from the reference overall
  (0 = indistinguishable, 100 = dramatically different)

Also give `note`: at most 12 words on what you actually see in the treated rows.

Score every item. Do not skip any, do not copy scores between items, and do not infer
anything from file names. Use the full range: if wells genuinely differ on an axis, your
scores on that axis should differ too, in either direction. If an image is unreadable,
give null scores for it and say so.

Write results to {out} as a JSON array of objects with keys: id, fewer_cells,
dna_bright, bigger_nuclei, mito_bright, actin_bright, elongated, any_change, note.

Then reply with only: the number of items scored, and one sentence on which axes were
hard to judge.
```

---

## Why it is written this way

**"Work only from what you see."** The model knows the pharmacology of many of these
compounds. If it could see the compound name it could recite the expected phenotype
without looking at the image, and the scores would be a literature lookup wearing a
measurement's clothes. Hence blind identifiers.

**"TREATED and REFERENCE share one brightness scale."** In an earlier round each
panel was contrast-stretched against its own percentiles. The model then scored at
chance on four of eight behaviours, because a dim treated well and a bright reference
well rendered identically. After the shared scale, the same model reached AUC .81 to
1.0 on those behaviours. The rendering was the bug, not the model.

**"Use the full range, in either direction."** Without it the scores collapse toward
50 (observed spread 4-8 points on a 100-point scale), which is why calibration
divides by the MAD of the scores rather than assuming a scale.

**`any_change` is asked for but never used as a label.** It is a check: a group that
the model says is indistinguishable from the reference should not get extreme scores
on individual axes.

**`note` is not used either.** It is there to read afterwards, because a free-text
line is how you notice that a model is answering a different question than you asked.
