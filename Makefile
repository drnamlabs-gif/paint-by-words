# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
# Every number and figure in README.md comes from `make all`.
# Requires P2I_DATA to point at a local copy of RxRx3-core (see ATTRIBUTION.md).

PY ?= python
SEEDS ?= 0 1 2 3 4

.PHONY: all counts data train eval figs gif screening baselines critic test app clean

all: data train eval figs gif screening

## counts nuclei in every well; needs the image shards, skipped if absent
counts:
	$(PY) -m pbw.images

## the groups table and the instruction sets
data:
	$(PY) -m pbw.features
	$(PY) -m pbw.instructions v3
	$(PY) -m pbw.instructions v2

train:
	for s in $(SEEDS); do $(PY) -m pbw.train --seed $$s; done

eval:
	$(PY) -m pbw.evaluate --definition v3

## the five figures in README.md; 1 and 2 need the image shards, 5 needs `make baselines`
figs:
	$(PY) -m pbw.figures

## the two animations in README.md; both need the image shards and trained checkpoints
gif:
	$(PY) -m pbw.demo

## what the ranking saves, counted in wells rather than in AUC
screening:
	$(PY) -m pbw.screening

## the finding: the same work under the old label definition, where the model
## looked much better and its toxicity-only baseline looked almost as good
baselines:
	for s in $(SEEDS); do $(PY) -m pbw.train --definition v2 --seed $$s; done
	$(PY) -m pbw.evaluate --definition v2

## layer 3: the vision model as a label source, and its control
critic:
	$(PY) -m pbw.critic
	for s in $(SEEDS); do $(PY) -m pbw.train --labels critic --seed $$s; done
	for s in $(SEEDS); do $(PY) -m pbw.train --labels numsub --seed $$s; done
	$(PY) -m pbw.evaluate --labels critic
	$(PY) -m pbw.evaluate --labels numsub

test:
	$(PY) -m pytest tests -q

app:
	$(PY) -m pbw.app

clean:
	rm -rf results/*.pt results/*.npz __pycache__ .pytest_cache
