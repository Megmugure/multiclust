"""
py_multiclust.cli

Command-line interface for the Python reimplementation of MULTICLUST.

Overview
--------
This module plays the same role as the original C `multiclust` executable:

  - Read a genotype file (here: STRUCTURE format, `.str`).
  - For a given K (number of clusters), run multiple EM initializations
    with different random seeds.
  - Keep the run with the highest log-likelihood.
  - Write an "indivq" file with admixture proportions (Q) per individual.

This makes it easy to see how data move from disk → EM algorithm → output,
and to compare runs between the C and Python implementations.
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np

from .io import load_stru
from .model import MulticlustModel
from .em import em_run


def _run_single_em(
    data,
    K: int,
    max_iter: int,
    tol: float,
    seed: int,
) -> Tuple[float, int, MulticlustModel]:
    """
    Run one EM optimization starting from a random initialization.

    Steps:
      1. Build a MulticlustModel from the input data.
      2. Run the serial EM algorithm until convergence (or max_iter).
      3. Return the final log-likelihood, number of iterations, and model.

    Parameters
    ----------
    data
        StruData object returned by load_stru().
    K : int
        Number of clusters.
    max_iter : int
        Maximum number of EM iterations.
    tol : float
        Convergence tolerance on change in log-likelihood.
    seed : int
        Random seed used for initialization of Q and P.

    Returns
    -------
    final_ll : float
        Final log-likelihood for this run.
    n_iter : int
        Number of EM iterations performed.
    model : MulticlustModel
        Model containing fitted Q, P, and pi.
    """
    # Initialize model parameters (Q, P, pi) from data with given seed
    model = MulticlustModel.from_data(data, K=K, random_state=seed)

    # Run standard serial EM (no parallelism here)
    final_ll, n_iter = em_run(model, max_iter=max_iter, tol=tol, verbose=False)

    return final_ll, n_iter, model


def _write_indivq(
    out_path: str,
    model: MulticlustModel,
    ids,
    pops,
) -> None:
    """
    Write Q (admixture proportions) to a .indivq-style file.

    This is the Python analogue of the C program’s `*.indivq` output.

    Format per line (when both ids and pops are available):

        [ID] [POP] q1 q2 ... qK

    Example
    -------
    CAN001024 1 0.000000 1.000000

    If ids or pops are None, the corresponding column is omitted.
    """
    # Make sure the output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as f:
        n_ind, K = model.Q.shape
        for i in range(n_ind):
            fields = []

            # Optional ID column
            if ids is not None:
                fields.append(str(ids[i]))

            # Optional population label column
            if pops is not None:
                fields.append(str(pops[i]))

            # K admixture proportions with 6 decimal places
            fields.extend(f"{q:.6f}" for q in model.Q[i])

            f.write(" ".join(fields) + "\n")


def main(argv=None) -> None:
    """
    Entry point for the Python MULTICLUST CLI.

    This function:
      1. Parses command-line arguments.
      2. Loads genotype data (currently only STRUCTURE format).
      3. Runs multiple EM initializations for a fixed K.
      4. Selects the best run by log-likelihood.
      5. Writes a .indivq file and prints a short summary.

    It is wired to `python -m py_multiclust.cli ...` via the
    `if __name__ == "__main__": main()` block at the bottom.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Python reimplementation of the multiclust C program "
            "(admixture model with EM)."
        )
    )

    # Required core arguments 

    parser.add_argument(
        "-k", "--K",
        type=int,
        required=True,
        help="Number of clusters (K).",
    )
    parser.add_argument(
        "-f", "--file",
        type=str,
        required=True,
        help="Input genotype file (e.g. STRUCTURE format).",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="stru",
        help="Input format (currently only 'stru' is supported).",
    )

    # Genotype / model options 

    parser.add_argument(
        "-p", "--ploidy",
        type=int,
        default=2,
        help="Ploidy (default: 2).",
    )

    # EM control parameters 

    parser.add_argument(
        "-n", "--n_init",
        type=int,
        default=40,
        help="Number of random initializations (default: 40).",
    )
    parser.add_argument(
        "-C", "--max_iter",
        type=int,
        default=1000,
        help="Maximum EM iterations per initialization (default: 1000).",
    )
    parser.add_argument(
        "-E", "--tol",
        type=float,
        default=1e-4,
        help="Convergence tolerance on log-likelihood (default: 1e-4).",
    )
    parser.add_argument(
        "-r", "--seed",
        type=int,
        default=42,
        help=(
            "Base random seed. Initialization i uses seed + i "
            "(default base: 42)."
        ),
    )

    # Output options 

    parser.add_argument(
        "-d", "--outdir",
        type=str,
        default=".",
        help="Output directory (default: current directory).",
    )

    # Parse CLI args (or provided argv list, useful for testing)
    args = parser.parse_args(argv)

    
    # 1. Load data
    
    if args.format != "stru":
        raise ValueError(
            f"Only STRUCTURE format ('stru') is supported for now, "
            f"got --format={args.format!r}"
        )

    data = load_stru(
        args.file,
        ploidy=args.ploidy,
        has_ids=True,
        has_pops=True,
    )

    
    # 2. Run multiple EM initializations for fixed K
   
    K = args.K
    max_iter = args.max_iter
    tol = args.tol
    base_seed = args.seed
    n_init = args.n_init

    # Track the best run across all random starts
    best_ll = -np.inf
    best_model: MulticlustModel | None = None
    best_init_idx: int | None = None
    best_seed: int | None = None
    best_n_iter: int | None = None

    for init_idx in range(n_init):
        seed = base_seed + init_idx

        final_ll, n_iter, model = _run_single_em(
            data=data,
            K=K,
            max_iter=max_iter,
            tol=tol,
            seed=seed,
        )

        # Roughly mimic the C run.log line style
        print(
            f"K = {K}, initialization = {init_idx}: "
            f"{final_ll:.6f} (converged) in {n_iter:4d} iterations, seed: {seed}"
        )

        # Keep the best run according to log-likelihood
        if final_ll > best_ll:
            best_ll = final_ll
            best_model = model
            best_init_idx = init_idx
            best_seed = seed
            best_n_iter = n_iter

    assert best_model is not None, "Internal error: best_model never set"

    
    # 3. Prepare output paths
    
    in_basename = os.path.basename(args.file)  # e.g. "wolves.str"

    # Naming scheme close to the C implementation:
    #   <basename>.mix.K=<K>.indivq
    out_prefix = f"{in_basename}.mix.K={K}"
    indivq_path = os.path.join(args.outdir, f"{out_prefix}.indivq")

    
    # 4. Write Q (indivq) file
    
    _write_indivq(indivq_path, best_model, data.ids, data.pops)

    
    # 5. Print final summary
    
    print()
    print("Best run summary")
    print("----------------")
    print(f"  K              : {K}")
    print(f"  best init idx  : {best_init_idx}")
    print(f"  best seed      : {best_seed}")
    print(f"  best n_iter    : {best_n_iter}")
    print(f"  best log-like  : {best_ll:.6f}")
    print(f"  indivq output  : {indivq_path}")


if __name__ == "__main__":
    # Allow `python -m py_multiclust.cli ...` or direct execution
    main()
