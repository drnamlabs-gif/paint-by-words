# Attribution and data licence

## The archive

This repository contains **no image or feature data**. It reads a local copy of
**RxRx3-core**, published by Recursion Pharmaceuticals:

- dataset: https://huggingface.co/datasets/recursionpharma/rxrx3-core
- paper: Fay et al., *RxRx3-core: Benchmarking drug-target interactions in
  high-content microscopy*, 2024.

RxRx3-core is distributed under Recursion's **Commercial End User Licence
Agreement (CEULA)**. Anyone running this code must obtain the data from Recursion
and comply with that licence themselves, including its attribution requirement and
its restriction on neuroscience applications. Read the licence as published; the
summary here is not a substitute for it.

Nothing derived from the archive, including trained weights, is published in this
repository, so that the share-alike provisions of the CEULA are not triggered by it.
To release weights or derived tables, check that licence text first.

A fully permissive alternative archive for this method is **JUMP cpg0016**
(CC0), which additionally allows batch transfer to be measured because the same
compounds are run across multiple sources. That is the natural next step and is
noted as such in MODEL_CARD.md.

## The method

Background for the language-to-intervention approach used here:

- Le, Erickson, Zhang, Levin, Bongard. *Prompt-conditioned control of simulated
  cells.* arXiv:2509.10660.
- Le, Blackiston, Levin, Bongard. *Toward Controlling Biology with Language:
  Offline Learning of Prompt-Conditioned Interventions for Cells, Organoids, and
  Biobots.* arXiv:2610.02247.

## This work

Independent work. It received **no specific funding** and was carried out while the
author held a postdoctoral position at the **University of Vermont**.

Author: Dr. Nam H. Le, drnamlabs.com.

## Software

sentence-transformers (`nli-mpnet-base-v2`), RDKit, PyTorch, scikit-learn, SciPy,
Pillow, pandas, PyArrow.
