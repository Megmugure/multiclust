"""
py_multiclust.em_parallel

Parallel EM for the MULTICLUST-style admixture model, using embarrassingly
parallel updates over loci.

Overview
--------
This module implements a parallel variant of the EM algorithm where:

  - The underlying model is the same as in py_multiclust.em:
        X[i, l, m]   : genotype of individual i, locus l, allele copy m
        P[k, l, a]   : frequency of allele a at locus l in cluster k
        Q[i, k]      : responsibility / admixture proportion of cluster k
        pi[k]        : mixture weight for cluster k

  - The expensive part of the M-step is computing expected allele counts

        c[k, l, a] = sum_i Q[i, k] * (# copies of allele a for individual i at locus l)

    Loci are conditionally independent given the cluster, so we can split
    the locus axis across processes and compute c on each chunk in parallel.

  - The E-step is kept serial and explicit on purpose, so it is easy to
    explain and reason about in a methods section.

This code is mainly intended as a clear example of locus-wise parallel EM,
rather than as a fully optimized drop-in replacement for the C code.
"""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple, List

import numpy as np

# We deliberately do not import MulticlustModel here to avoid circular imports.
# em_run_parallel() only assumes that `model` has:
#   - X : (I, L, M)
#   - Q : (I, K)
#   - P : (K, L, A)


# ---------------------------------------------------------------------------
# Helper: split locus indices into chunks
# ---------------------------------------------------------------------------

def _chunk_indices(n_loci: int, n_jobs: int) -> List[slice]:
    """
    Split the locus axis [0, n_loci) into approximately equal contiguous slices.

    Parameters
    ----------
    n_loci : int
        Total number of loci L.
    n_jobs : int
        Desired number of chunks / processes.

    Returns
    -------
    slices : list of slice
        Each slice selects a contiguous block of loci; together they cover
        [0, n_loci) without overlap.
    """
    n_jobs = max(1, int(n_jobs))
    if n_jobs == 1 or n_loci <= n_jobs:
        return [slice(0, n_loci)]

    base = n_loci // n_jobs
    rem = n_loci % n_jobs
    slices: List[slice] = []
    start = 0

    # Distribute the remainder one-by-one to the first `rem` chunks
    for j in range(n_jobs):
        length = base + (1 if j < rem else 0)
        if length <= 0:
            continue
        end = start + length
        slices.append(slice(start, end))
        start = end

    return slices


# ---------------------------------------------------------------------------
# Chunk-level expected counts (M-step) for one process
# ---------------------------------------------------------------------------

def _compute_counts_chunk(
    X_chunk: np.ndarray,  # shape (I, Lc, M)
    Q: np.ndarray,        # shape (I, K)
    A: int,               # number of alleles
) -> np.ndarray:
    """
    Compute expected allele counts c[k, l, a] on a locus chunk.

    This is the M-step contribution for a subset of loci:

        c[k, l, a] = sum_i Q[i, k] * (# copies of allele a at locus l in individual i)

    Parameters
    ----------
    X_chunk : np.ndarray, shape (I, Lc, M)
        Genotypes for all individuals but only a subset of loci.
    Q : np.ndarray, shape (I, K)
        Responsibilities / admixture proportions for each individual and cluster.
    A : int
        Number of alleles (indexed 0..A-1).

    Returns
    -------
    counts : np.ndarray, shape (K, Lc, A)
        Expected counts for this chunk only.
    """
    I, Lc, M = X_chunk.shape
    Iq, K = Q.shape
    assert I == Iq

    counts = np.zeros((K, Lc, A), dtype=float)

    # For each allele value a:
    #   1. Count how many times it appears per (i, l).
    #   2. Weight that count by Q[i, k] and sum over i.
    for a in range(A):
        mask = (X_chunk == a)              # (I, Lc, M)
        allele_counts = mask.sum(axis=2)   # (I, Lc)
        if not allele_counts.any():
            continue

        for k in range(K):
            counts[k, :, a] = (Q[:, k][:, None] * allele_counts).sum(axis=0)

    return counts


# ---------------------------------------------------------------------------
# Parallel and serial expected counts over all loci
# ---------------------------------------------------------------------------

def _compute_counts_parallel(
    X: np.ndarray,  # (I, L, M)
    Q: np.ndarray,  # (I, K)
    P: np.ndarray,  # (K, L, A) just to get A
    n_jobs: int,
) -> np.ndarray:
    """
    Compute expected allele counts using locus-parallelism.

    The locus axis is split into slices, each slice is processed in a separate
    process, and the resulting chunks are combined.

    Parameters
    ----------
    X : np.ndarray, shape (I, L, M)
        Full genotype array.
    Q : np.ndarray, shape (I, K)
        Responsibilities / admixture proportions per individual and cluster.
    P : np.ndarray, shape (K, L, A)
        Current allele frequencies (used only to infer the number of alleles A).
    n_jobs : int
        Number of processes to use.

    Returns
    -------
    counts : np.ndarray, shape (K, L, A)
        Expected allele counts across all loci.
    """
    I, L, M = X.shape
    K, Lp, A = P.shape
    assert L == Lp

    slices = _chunk_indices(L, n_jobs)
    if len(slices) == 1:
        return _compute_counts_chunk(X, Q, A)

    results = []

    # On shared systems this should be run inside a batch job, with a sensible
    # max_workers value. Here we match the number of workers to the number of
    # chunks for simplicity.
    with ProcessPoolExecutor(max_workers=len(slices)) as ex:
        futures = []
        for sl in slices:
            X_chunk = X[:, sl, :]
            futures.append(ex.submit(_compute_counts_chunk, X_chunk, Q, A))
        for fut in futures:
            results.append(fut.result())

    counts = np.zeros((K, L, A), dtype=float)
    for sl, chunk_counts in zip(slices, results):
        counts[:, sl, :] = chunk_counts

    return counts


def _compute_counts_serial(
    X: np.ndarray,
    Q: np.ndarray,
    P: np.ndarray,
) -> np.ndarray:
    """
    Serial version of expected counts; used when n_jobs == 1 or for testing.

    Parameters
    ----------
    X : np.ndarray, shape (I, L, M)
        Genotypes.
    Q : np.ndarray, shape (I, K)
        Responsibilities.
    P : np.ndarray, shape (K, L, A)
        Allele frequencies (used to infer L and A).

    Returns
    -------
    counts : np.ndarray, shape (K, L, A)
        Expected allele counts over all loci.
    """
    I, L, M = X.shape
    K, Lp, A = P.shape
    assert L == Lp

    counts = np.zeros((K, L, A), dtype=float)

    for a in range(A):
        mask = (X == a)                  # (I, L, M)
        allele_counts = mask.sum(axis=2)  # (I, L)
        if not allele_counts.any():
            continue
        for k in range(K):
            counts[k, :, a] = (Q[:, k][:, None] * allele_counts).sum(axis=0)

    return counts


# ---------------------------------------------------------------------------
# E-step (serial, explicit)
# ---------------------------------------------------------------------------

def _e_step(
    X: np.ndarray,   # (I, L, M)
    P: np.ndarray,   # (K, L, A)
    pi: np.ndarray,  # (K,)
) -> Tuple[np.ndarray, float]:
    """
    E-step: update responsibilities Q and return (Q, log-likelihood).

    Given P and pi, we compute

        log p(x_i | k)  = sum_{l,m} log P[k, l, X[i,l,m]]
        log p(x_i, k)   = log pi[k] + log p(x_i | k)

    and normalize over k to get responsibilities:

        Q[i,k] = p(z_i = k | x_i)
               = softmax_k( log pi[k] + sum_{l,m} log P[k,l, X[i,l,m]] ).

    Parameters
    ----------
    X : np.ndarray, shape (I, L, M)
        Genotype array.
    P : np.ndarray, shape (K, L, A)
        Allele frequencies per cluster, locus, and allele.
    pi : np.ndarray, shape (K,)
        Mixture weights per cluster.

    Returns
    -------
    Q : np.ndarray, shape (I, K)
        Updated responsibilities for each individual.
    ll : float
        Total log-likelihood sum_i log p(x_i).
    """
    I, L, M = X.shape
    K, Lp, A = P.shape
    assert L == Lp

    # log P[k, l, a] with small floor to avoid log(0)
    logP = np.log(P + 1e-300)
    log_pi = np.log(pi + 1e-300)

    log_prob = np.zeros((I, K), dtype=float)

    # For each allele value a, count its occurrences in each (i, l),
    # then for each cluster k, accumulate the contribution of that allele.
    for a in range(A):
        mask = (X == a)                  # (I, L, M)
        allele_counts = mask.sum(axis=2)  # (I, L)
        if not allele_counts.any():
            continue

        for k in range(K):
            # logP[k, :, a] has shape (L,)
            # allele_counts @ logP[k, :, a] -> (I,)
            log_prob[:, k] += allele_counts @ logP[k, :, a]

    # Add log pi_k term
    log_prob += log_pi[None, :]

    # Numerically stable softmax over clusters
    max_log = log_prob.max(axis=1, keepdims=True)  # (I, 1)
    stabilized = log_prob - max_log
    np.exp(stabilized, out=stabilized)
    sum_exp = stabilized.sum(axis=1, keepdims=True)
    Q = stabilized / sum_exp

    # log-likelihood: sum_i log sum_k pi_k p(x_i | k)
    ll = float((max_log + np.log(sum_exp)).sum())

    return Q, ll


# ---------------------------------------------------------------------------
# Main parallel EM driver
# ---------------------------------------------------------------------------

def em_run_parallel(
    model,
    max_iter: int = 100,
    tol: float = 1e-4,
    verbose: bool = False,
    n_jobs: int = 1,
    alpha: float = 1.0,
) -> Tuple[float, int]:
    """
    Run EM with optional locus-parallel M-step on a MULTICLUST-style model.

    E-step is done serially; the M-step (expected allele counts and P update)
    can be split across loci using multiple processes.

    Parameters
    ----------
    model : object
        Must have attributes:
          - X : np.ndarray, shape (I, L, M)
          - Q : np.ndarray, shape (I, K)
          - P : np.ndarray, shape (K, L, A)
        Q and P are updated in-place at the end of the run.
    max_iter : int, default=100
        Maximum number of EM iterations.
    tol : float, default=1e-4
        Convergence tolerance on change in log-likelihood between iterations.
    verbose : bool, default=False
        If True, print log-likelihood and delta each iteration.
    n_jobs : int, default=1
        Number of processes to use to split loci. If n_jobs <= 1, runs serially.
    alpha : float, default=1.0
        Symmetric Dirichlet prior on allele frequencies per (k, l):
            P[k,l,:] ∝ counts[k,l,:] + alpha

    Returns
    -------
    final_ll : float
        Final log-likelihood after the last EM iteration.
    n_iter : int
        Number of iterations performed.
    """
    X = model.X
    Q = model.Q.copy()  # working copy; write back at the end
    P = model.P.copy()

    I, L, M = X.shape
    K, Lp, A = P.shape
    assert L == Lp

    # Initialize mixture weights from initial Q
    pi = Q.mean(axis=0)
    pi /= pi.sum()

    ll_old = -math.inf
    final_ll = ll_old
    n_iter_done = 0

    for it in range(1, max_iter + 1):
        # ----- E-step -----
        Q, ll = _e_step(X, P, pi)

        # ----- M-step -----
        # Compute expected counts, either serially or in parallel across loci
        if n_jobs is None or n_jobs <= 1:
            counts = _compute_counts_serial(X, Q, P)
        else:
            counts = _compute_counts_parallel(X, Q, P, n_jobs=n_jobs)

        # Update P with symmetric Dirichlet prior alpha
        P = counts + alpha
        P /= P.sum(axis=2, keepdims=True)

        # Update mixing proportions pi
        pi = Q.mean(axis=0)
        pi /= pi.sum()

        # ----- convergence check -----
        delta = ll - ll_old
        final_ll = ll
        n_iter_done = it

        if verbose:
            if it == 1:
                print(f"iter {it:3d}: ll={ll:.6f}")
            else:
                print(f"iter {it:3d}: ll={ll:.6f}, delta={delta:.6f}")

        if it > 1 and abs(delta) < tol:
            break

        ll_old = ll

    # Write final parameters back into the model
    model.Q[:] = Q
    model.P[:] = P
    # If the model tracks pi or loglike, they could also be stored here:
    #   model.pi = pi
    #   model.loglike = final_ll

    return final_ll, n_iter_done
