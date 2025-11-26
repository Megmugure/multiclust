"""
py_multiclust.model

Core data structure for the Python reimplementation of MULTICLUST.

This module defines a single class, `MulticlustModel`, which collects:

    - X   : genotype matrix (I × L × M)
    - Q   : admixture proportions per individual (I × K)
    - P   : allele frequencies per cluster and locus (K × L × A)
    - pi  : mixture weights for clusters (K,)

and optional metadata (IDs and population labels) from the original
STRUCTURE file.

Typical usage:

    from py_multiclust.io import load_stru
    from py_multiclust.model import MulticlustModel

    data = load_stru("tests/data/wolves.str", ploidy=2, has_ids=True, has_pops=True)
    model = MulticlustModel.from_data(data, K=2, random_state=42)

    # `model` can then be passed to EM routines (em.py / em_parallel.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .io import StruData


@dataclass
class MulticlustModel:
    """
    Python version of the MULTICLUST admixture model.

    This class stores both:
      * the observed data (genotypes X), and
      * the EM parameters (Q, P, pi) for a given number of clusters K.

    Attributes
    ----------
    X : np.ndarray
        Genotype tensor of shape (n_ind, n_loci, ploidy).
        - n_ind  : number of individuals (I)
        - n_loci : number of loci (L)
        - ploidy : number of allele copies per locus (M, e.g. 2 for diploid)
        Values are integer allele codes; negative values (e.g. -9) can be used
        for missing alleles.

    K : int
        Number of clusters (components) in the admixture model.

    n_alleles : int
        Number of distinct allele codes, assumed to be 0 .. n_alleles-1.
        Inferred from X, ignoring negative / missing values.

    Q : np.ndarray
        Admixture proportions per individual, shape (n_ind, K).
        Each row is a probability vector over clusters (sums to 1).

    P : np.ndarray
        Allele frequencies, shape (K, n_loci, n_alleles).
        P[k, l, a] ≈ probability of allele a at locus l in cluster k.
        Each (k, l, :) slice is a probability vector (sums to 1).

    pi : np.ndarray
        Mixture weights for clusters, shape (K,).
        Computed initially as the mean of the Q rows.

    ids : Optional[np.ndarray]
        Individual IDs from the STRUCTURE file, shape (n_ind,), or None.

    pops : Optional[np.ndarray]
        Population labels from the STRUCTURE file, shape (n_ind,), or None.
    """

    X: np.ndarray
    K: int
    n_alleles: int
    Q: np.ndarray
    P: np.ndarray
    pi: np.ndarray
    ids: Optional[np.ndarray] = None
    pops: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------
    @property
    def n_ind(self) -> int:
        """Number of individuals."""
        return self.X.shape[0]

    @property
    def n_loci(self) -> int:
        """Number of loci."""
        return self.X.shape[1]

    @property
    def ploidy(self) -> int:
        """Number of allele copies per locus."""
        return self.X.shape[2]

    # ------------------------------------------------------------------
    # Constructor from StruData
    # ------------------------------------------------------------------
    @classmethod
    def from_data(
        cls,
        data: StruData,
        K: int,
        random_state: Optional[int] = None,
    ) -> "MulticlustModel":
        """
        Build an initial MulticlustModel from STRUCTURE-format data.

        This method performs the initialization step for EM:
          1. Copies the genotype tensor from StruData into X.
          2. Infers the number of alleles (n_alleles) from X.
          3. Randomly initializes admixture proportions Q.
          4. Sets mixture weights pi as the mean of Q across individuals.
          5. Randomly initializes allele frequencies P using Dirichlet draws.

        Parameters
        ----------
        data : StruData
            Output of py_multiclust.io.load_stru(), containing:
                - genotypes : (n_ind, n_loci, ploidy)
                - ids, pops : optional metadata

        K : int
            Number of clusters for the admixture model.

        random_state : Optional[int], default=None
            Seed for NumPy's RandomState, for reproducible initialization.

        Returns
        -------
        MulticlustModel
            A new MulticlustModel instance ready to be passed into EM.
        """
        # Copy genotype tensor to avoid mutating the original StruData.
        X = data.genotypes.copy()  # shape (n_ind, n_loci, ploidy)
        n_ind, n_loci, ploidy = X.shape

        # Infer the number of alleles, ignoring missing values (< 0).
        if np.any(X >= 0):
            max_allele = int(X[X >= 0].max())
        else:
            max_allele = 0
        n_alleles = max_allele + 1  # alleles are 0..max_allele inclusive

        rng = np.random.RandomState(random_state)

        # Initialize Q (admixture proportions) and normalize rows.
        Q = rng.rand(n_ind, K)
        Q /= Q.sum(axis=1, keepdims=True)

        # Initialize pi (mixture weights) as the average of Q.
        pi = Q.mean(axis=0)
        pi /= pi.sum()

        # Initialize P (allele frequencies) with symmetric Dirichlet(1,...,1).
        P = np.empty((K, n_loci, n_alleles), dtype=float)
        alpha = np.ones(n_alleles, dtype=float)

        for k in range(K):
            for l in range(n_loci):
                g = rng.gamma(alpha, 1.0)
                P[k, l, :] = g / g.sum()

        return cls(
            X=X,
            K=K,
            n_alleles=n_alleles,
            Q=Q,
            P=P,
            pi=pi,
            ids=data.ids,
            pops=data.pops,
        )


# ---------------------------------------------------------------------------
# Minimal sanity-check when running this module directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Example:
    #
    #   python -m py_multiclust.model
    #
    # from the project root (assuming tests/data/wolves.str exists).
    # This just checks that model construction works end-to-end.
    import os
    from .io import load_stru

    example_path = os.path.join("tests", "data", "wolves.str")
    if os.path.exists(example_path):
        print(f"Loading STRUCTURE data from: {example_path}")
        data = load_stru(example_path, ploidy=2, has_ids=True, has_pops=True)
        model = MulticlustModel.from_data(data, K=2, random_state=42)
        print("  n_ind     :", model.n_ind)
        print("  n_loci    :", model.n_loci)
        print("  ploidy    :", model.ploidy)
        print("  K         :", model.K)
        print("  n_alleles :", model.n_alleles)
        print("  Q.shape   :", model.Q.shape)
        print("  P.shape   :", model.P.shape)
        print("  pi        :", model.pi)
    else:
        print("Example file tests/data/wolves.str not found; nothing to test.")
