# py_multiclust

This folder contains my in-progress Python reimplementation of the MULTICLUST C code.  
The goal is not to build a version I can use to extend an algorithm for my dissertation.

---

## Module overview

- **`io.py`**  
  - Reads STRUCTURE-format genotype files (`*.str`), including the wolves example.  
  - Handles messy headers and missing data.  
  - Returns a `StruData` object with:
    - `genotypes` (shape: `n_ind × n_loci × ploidy`)
    - `ids` (individual IDs)
    - `pops` (population labels)

- **`model.py`**  
  - Defines `MulticlustModel`, a container for:
    - `X`: genotype tensor
    - `K`: number of clusters
    - `Q`: admixture proportions (individual × cluster)
    - `P`: allele frequencies (cluster × locus × allele)
    - `pi`: cluster weights  
  - `from_data(...)` builds an initial model from `StruData` with random Q and P.

- **`em.py`**  
  - Serial EM implementation (single core).  
  - Uses a mixture-of-multinomials model over allele counts per individual and locus.  
  - Main functions:
    - `em_run(model, max_iter, tol, verbose)` – runs EM in-place.
    - `log_likelihood(model)` – computes the current log-likelihood.

- **`em_parallel.py`**  
  - Prototype “embarrassingly parallel” EM for the M-step.  
  - Splits loci across processes and aggregates expected allele counts.  
  - Uses the same underlying model as `em.py`, and is mainly for exploring parallelization ideas.

- **`cli.py`**  
  - Command-line interface that mimics the C `multiclust` program.  
  - For a given K:
    - runs multiple random EM initializations,
    - keeps the run with the highest log-likelihood,
    - writes a `*.indivq` file with Q (admixture proportions) per individual.  

---

## How to run (example)

From the project root (not inside `py_multiclust/`), with the `multiclust` conda environment active:

```bash
python -m py_multiclust.cli \
  -k 2 \
  -f tests/data/wolves.str \
  --format stru \
  -p 2 \
  -n 40 \
  -C 100 \
  -E 1e-4 \
  -r 42 \
  -d out_py_wolves
