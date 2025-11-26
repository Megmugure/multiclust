"""
py_multiclust.em

Serial (single-core) EM algorithm for the Python MULTICLUST reimplementation.

This module implements a simplified EM scheme that mirrors the structure of the
C MULTICLUST code, but in a more explicit and numpy-friendly way.

Model sketch
------------
  • Each individual i has allele counts at each locus:
        counts[i, l, a]  = number of copies of allele a at locus l.

  • Given a cluster k, these counts are modeled as multinomial with parameters
        P[k, l, :]  at each locus l.

  • The full model is a mixture of K such components with mixture weights
        pi[k].

The state of the model is stored in `MulticlustModel` (see model.py):

    X   : genotype tensor, shape (n_ind, n_loci, ploidy)
    Q   : admixture / responsibilities per individual, shape (n_ind, K)
    P   : allele frequencies per cluster and locus, shape (K, n_loci, n_alleles)
    pi  : mixture weights per cluster, shape (K,)

This file provides:

    - em_run(model, max_iter, tol, verbose)  : main serial EM loop
    - log_likelihood(model)                 : convenience function
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .model import MulticlustModel

# Small numerical constants:
#   _EPS    : avoid log(0) and division by zero
#   _SMOOTH : add to allele counts so no P entry is exactly 0
_EPS = 1e-12
_SMOOTH = 1e-3


def _compute_counts(model: MulticlustModel) -> np.ndarray:
    """
    Convert genotype tensor X (n_ind, n_loci, ploidy) into allele counts
    (n_ind, n_loci, n_alleles).

    Instead of working directly with allele codes in X, this function builds
    a per-individual count table over alleles at each locus.

    Missing alleles
    ---------------
    Missing allele copies are assumed to have negative integer codes
    (e.g. -9) and are ignored when accumulating counts.

    Parameters
    ----------
    model : MulticlustModel
        Requires:
          - X          : (n_ind, n_loci, ploidy) integer allele codes
          - n_alleles  : number of distinct allele codes

    Returns
    -------
    counts : np.ndarray
        Array of shape (n_ind, n_loci, n_alleles) where
        counts[i, l, a] is the number of copies of allele a observed
        at locus l in individual i.
    """
    X = model.X  # alias for genotypes
    n_ind, n_loci, ploidy = X.shape
    n_alleles = model.n_alleles

    # Initialize all counts to zero
    counts = np.zeros((n_ind, n_loci, n_alleles), dtype=float)

    # Loop over each allele copy position (0..ploidy-1)
    for c in range(ploidy):
        # alleles[i, l] is the integer code of copy c for individual i at locus l
        alleles = X[:, :, c]  # (n_ind, n_loci)

        # Valid (non-missing) alleles have non-negative codes
        mask = alleles >= 0
        if not np.any(mask):
            continue

        # Indices of observed alleles
        idx_i, idx_l = np.where(mask)  # positions where there is a real allele
        idx_a = alleles[mask]          # allele code at each observed position

        # Increment counts[i, l, a] by 1 for each observed allele copy
        np.add.at(counts, (idx_i, idx_l, idx_a), 1.0)

    return counts


def _e_step(
    model: MulticlustModel,
    counts: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    E-step: compute responsibilities R[i, k] for each individual i and cluster k.

    Model
    -----
    For each individual i and cluster k:

        log p(x_i | k) = sum_{l,a} counts[i,l,a] * log P[k,l,a]

    Combining with the mixture weights pi[k] gives:

        log p(x_i, z_i = k) = log pi[k] + log p(x_i | k)

    Responsibilities are then:

        R[i,k] = p(z_i = k | x_i)
               = softmax_k ( log pi[k] + sum_{l,a} counts[i,l,a] * log P[k,l,a] )

    Parameters
    ----------
    model : MulticlustModel
        Uses:
          - K   : number of clusters
          - pi  : mixture weights, shape (K,)
          - P   : allele freqs, shape (K, n_loci, n_alleles)
    counts : np.ndarray
        Allele counts per individual and locus, shape (n_ind, n_loci, n_alleles).

    Returns
    -------
    R : np.ndarray
        Responsibilities, shape (n_ind, K). Each row sums to 1.
    ll : float
        Total log-likelihood sum_i log p(x_i).
    """
    K = model.K
    pi = model.pi  # (K,)
    P = model.P    # (K, n_loci, n_alleles)

    # Precompute log P with a small epsilon for numerical safety
    log_P = np.log(P + _EPS)  # (K, n_loci, n_alleles)

    # log pi as a row vector for broadcasting: (1, K)
    log_pi = np.log(pi + _EPS)[None, :]

    # log_prob[i, k] = log pi_k + sum_{l,a} counts[i,l,a] * log P[k,l,a]
    #
    # shapes:
    #   counts[:, None, :, :] -> (n_ind, 1, n_loci, n_alleles)
    #   log_P[None, :, :, :]  -> (1, K,   n_loci, n_alleles)
    # elementwise product then summed over loci and alleles -> (n_ind, K)
    log_prob = log_pi + (counts[:, None, :, :] * log_P[None, :, :, :]).sum(axis=(2, 3))

    # Convert log_prob to responsibilities using a numerically stable softmax.
    max_log = log_prob.max(axis=1, keepdims=True)  # (n_ind, 1)
    stable = np.exp(log_prob - max_log)
    denom = stable.sum(axis=1, keepdims=True)
    R = stable / denom  # (n_ind, K), rows sum to 1

    # Total log-likelihood:
    #   log p(x_i) = max_log[i] + log(sum_k exp(log_prob[i,k] - max_log[i]))
    # summed over individuals
    ll = float((max_log + np.log(denom)).sum())

    return R, ll


def _m_step(
    model: MulticlustModel,
    counts: np.ndarray,
    R: np.ndarray,
) -> None:
    """
    M-step: update pi, P, and Q given responsibilities R and allele counts.

    Using responsibilities R[i,k] and allele counts counts[i,l,a]:

      1. Effective cluster sizes:
            N_k = sum_i R[i,k]
         and mixture weights:
            pi_k = N_k / sum_j N_j

      2. Cluster- and locus-specific allele counts:
            allele_counts[k,l,a] = sum_i R[i,k] * counts[i,l,a]

         then frequencies with smoothing:
            P[k,l,:] ∝ allele_counts[k,l,:] + _SMOOTH

      3. Admixture proportions:
            Q[i,k] = R[i,k]
    """
    # ----------------------
    # 1. Update mixture weights pi
    # ----------------------
    Nk = R.sum(axis=0) + _EPS  # (K,)
    pi = Nk / Nk.sum()
    model.pi = pi

    # ----------------------
    # 2. Update allele frequencies P
    # ----------------------
    # weighted_counts[i, k, l, a] = R[i,k] * counts[i,l,a]
    weighted_counts = R[:, :, None, None] * counts[:, None, :, :]  # (n_ind, K, n_loci, n_alleles)

    # Sum over individuals -> cluster/locus/allele counts
    allele_counts = weighted_counts.sum(axis=0)  # (K, n_loci, n_alleles)

    # Add smoothing so no entry is exactly zero; this also acts like
    # a weak symmetric Dirichlet prior and improves numerical stability.
    P = allele_counts + _SMOOTH
    P /= P.sum(axis=2, keepdims=True)  # normalize along allele axis
    model.P = P

    # ----------------------
    # 3. Update Q (admixture proportions)
    # ----------------------
    model.Q = R.copy()


def log_likelihood(model: MulticlustModel) -> float:
    """
    Compute the current log-likelihood of the model under the
    mixture-of-multinomials formulation.

    This is a thin wrapper around `_compute_counts` and `_e_step`.

    Parameters
    ----------
    model : MulticlustModel
        Current model whose (X, P, pi) define the distribution.

    Returns
    -------
    ll : float
        Total log-likelihood over all individuals.
    """
    counts = _compute_counts(model)
    _, ll = _e_step(model, counts)
    return ll


def em_run(
    model: MulticlustModel,
    max_iter: int = 100,
    tol: float = 1e-4,
    verbose: bool = False,
) -> Tuple[float, int]:
    """
    Run EM on the given model *in-place* (serial version).

    Per iteration:
      1. E-step: compute responsibilities R[i,k] and log-likelihood ll.
      2. M-step: update pi, P, and Q using R and counts.
      3. Check convergence based on change in log-likelihood.

    Parameters
    ----------
    model : MulticlustModel
        Initialized model; its fields provide the starting values:
          - X  : genotype tensor
          - Q  : initial admixture proportions
          - P  : initial allele frequencies
          - pi : initial cluster weights

    max_iter : int, default=100
        Maximum number of EM iterations.

    tol : float, default=1e-4
        Convergence tolerance on the absolute change in log-likelihood
        between successive iterations:

            |ll_t - ll_{t-1}| < tol

        triggers a stop.

    verbose : bool, default=False
        If True, print the log-likelihood and delta at each iteration.

    Returns
    -------
    final_ll : float
        Final log-likelihood after the last EM iteration.

    n_iter : int
        Number of iterations performed before stopping (either due to
        convergence or hitting `max_iter`).
    """
    # Precompute allele counts once; they do not change during EM.
    counts = _compute_counts(model)

    prev_ll: float | None = None
    final_ll: float = float("nan")

    for it in range(1, max_iter + 1):
        # --- E-step ---
        R, ll = _e_step(model, counts)

        # --- M-step ---
        _m_step(model, counts, R)

        # --- Logging / convergence check ---
        if verbose:
            if prev_ll is None:
                print(f"iter {it:3d}: ll={ll:.6f}")
            else:
                print(f"iter {it:3d}: ll={ll:.6f}, delta={ll - prev_ll:.6f}")

        if prev_ll is not None and abs(ll - prev_ll) < tol:
            final_ll = ll
            return final_ll, it

        prev_ll = ll
        final_ll = ll

    # Reached max_iter without satisfying the tolerance
    return final_ll, max_iter


__all__ = ["em_run", "log_likelihood"]
