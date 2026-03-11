"""
py_multiclust.io

I/O utilities for reading STRUCTURE-format genotype data into a
NumPy-friendly container.

Main task:

    STRUCTURE text file  ->  StruData(genotypes, ids, pops, ploidy)

where

    genotypes : (n_ind, n_loci, ploidy) integer array
    ids       : (n_ind,) array of sample IDs or None
    pops      : (n_ind,) array of population labels or None

This is the “dynamic genotype matrix” (I × L × M) used by the Python
implementation of the MULTICLUST model and the EM routines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class StruData:
    """
    Container for STRUCTURE-format genotype data.

    Attributes
    ----------
    genotypes : np.ndarray
        Integer genotype array of shape (n_ind, n_loci, ploidy).
        For diploids (ploidy = 2):

            genotypes[i, ℓ, 0] = allele 1 for individual i at locus ℓ
            genotypes[i, ℓ, 1] = allele 2 for individual i at locus ℓ

    ids : Optional[np.ndarray]
        Sample IDs (e.g. "CAN001024"), shape (n_ind,), or None.

    pops : Optional[np.ndarray]
        Population labels (e.g. "1", "2"), shape (n_ind,), or None.

    ploidy : int
        Number of allele copies per locus (2 for diploids).
    """

    genotypes: np.ndarray  # shape (n_ind, n_loci, ploidy)
    ids: Optional[np.ndarray] = None  # shape (n_ind,)
    pops: Optional[np.ndarray] = None  # shape (n_ind,)
    ploidy: int = 2

    @property
    def n_ind(self) -> int:
        """Number of individuals."""
        return self.genotypes.shape[0]

    @property
    def n_loci(self) -> int:
        """Number of loci."""
        return self.genotypes.shape[1]


def load_stru(
    path: str,
    ploidy: int = 2,
    has_ids: bool = True,
    has_pops: bool = True,
    comment_chars: str = "#",
    missing_value: int = -9,
) -> StruData:
    """
    Load a STRUCTURE-format file into a StruData object.

    Parameters
    ----------
    path : str
        Path to the STRUCTURE-format file (e.g. tests/data/wolves.str).

    ploidy : int, default=2
        Number of alleles per locus per individual.

    has_ids : bool, default=True
        Whether the first column is an individual ID.

    has_pops : bool, default=True
        Whether the second column is a population label.

    comment_chars : str, default="#"
        Lines whose first character is in this set are skipped.

    missing_value : int, default=-9
        Integer code for missing alleles. Common non-integer encodings
        (".", "NA", "NaN", "nan") are mapped to this value.

    Expected data layout
    --------------------
    For data lines (not headers), the function assumes rows of the form:

        [ID] [POP] allele_11 allele_12 ... allele_L1 allele_L2

    depending on `has_ids` and `has_pops`:

        has_ids=True, has_pops=True:
            ID  POP  a_11  a_12  a_21  a_22 ... a_L1  a_L2

        has_ids=True, has_pops=False:
            ID  a_11  a_12  a_21  a_22 ... a_L1  a_L2

        has_ids=False, has_pops=False:
            a_11  a_12  a_21  a_22 ... a_L1  a_L2

    The total number of allele entries per individual is n_loci * ploidy.

    Handling of irregular files
    ---------------------------
    This function is written to handle the structure of the example
    `wolves.str` file and similar “noisy” STRUCTURE-like files. It:

      1. Reads all non-blank, non-comment lines.
      2. Skips lines made entirely of '.' tokens (visual separators).
      3. Identifies the most frequent column count that is compatible with
         the specified (ploidy, has_ids, has_pops).
      4. Treats lines whose first token (when has_ids=True) is numeric
         (e.g. "-1") as header/meta.
      5. Parses only lines with the chosen column count and a non-numeric
         first token (if has_ids=True).

    Returns
    -------
    StruData
        Object containing genotype array, IDs, populations and ploidy.

    Raises
    ------
    ValueError
        If no usable lines are found, no compatible column count exists,
        or if a data line has an incompatible number of allele tokens
        or a non-integer allele value.
    """
    ids: List[str] = []
    pops: List[str] = []
    geno_rows: List[np.ndarray] = []

    # Number of non-genotype columns at the start of each data line
    meta_cols = (1 if has_ids else 0) + (1 if has_pops else 0)

    # First pass: collect candidate lines 
    lines_info = []  # (line_number, tokens_list, num_columns)

    with open(path, "r") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()

            if not line:
                continue

            if line[0] in comment_chars:
                continue

            tokens = line.split()
            if not tokens:
                continue

            # Skip lines that are entirely '.' tokens
            if all(tok == "." for tok in tokens):
                continue

            n_cols = len(tokens)
            lines_info.append((lineno, tokens, n_cols))

    if not lines_info:
        raise ValueError(f"No usable lines found in {path}")

    # Determine the dominant compatible column count 
    col_freq = {}

    for lineno, tokens, n_cols in lines_info:
        if n_cols <= meta_cols:
            continue

        allele_cols = n_cols - meta_cols
        if allele_cols % ploidy != 0:
            continue

        col_freq[n_cols] = col_freq.get(n_cols, 0) + 1

    if not col_freq:
        raise ValueError(
            f"Could not find any lines in {path} whose column count is compatible "
            f"with ploidy={ploidy} and meta_cols={meta_cols}"
        )

    # Use the most common compatible column count as the data width
    expected_n_cols = max(col_freq.items(), key=lambda kv: kv[1])[0]
    allele_cols = expected_n_cols - meta_cols
    n_loci = allele_cols // ploidy

    def is_header_line(tokens: List[str]) -> bool:
        """
        Heuristic to classify a line as header/meta instead of data.

        Currently:
        - if has_ids=True and the first token is an integer, we treat the line
          as header/meta (e.g. "-1 ...").
        """
        if not has_ids:
            return False

        first = tokens[0]
        try:
            int(first)
            return True
        except ValueError:
            return False

    # Second pass: parse data lines
    for lineno, tokens, n_cols in lines_info:
        if n_cols != expected_n_cols:
            continue

        if is_header_line(tokens):
            continue

        offset = 0
        if has_ids:
            ids.append(tokens[offset])
            offset += 1
        if has_pops:
            pops.append(tokens[offset])
            offset += 1

        allele_tokens = tokens[offset:]

        if len(allele_tokens) != n_loci * ploidy:
            raise ValueError(
                f"Line {lineno} in {path} has {len(allele_tokens)} allele tokens, "
                f"expected {n_loci * ploidy}"
            )

        cleaned: List[int] = []
        for x in allele_tokens:
            if x in {".", "NA", "NaN", "nan"}:
                cleaned.append(missing_value)
            else:
                try:
                    cleaned.append(int(x))
                except ValueError as e:
                    raise ValueError(
                        f"Non-integer allele value on line {lineno} in {path}: {x!r}"
                    ) from e

        allele_array = np.array(cleaned, dtype=int)
        geno_row = allele_array.reshape(n_loci, ploidy)
        geno_rows.append(geno_row)

    if not geno_rows:
        raise ValueError(
            f"No genotype rows parsed in {path} with {expected_n_cols} columns"
        )

    # Stack individuals: (n_ind, n_loci, ploidy)
    genotypes = np.stack(geno_rows, axis=0)

    ids_arr = np.array(ids, dtype=object) if ids else None
    pops_arr = np.array(pops, dtype=object) if pops else None

    return StruData(genotypes=genotypes, ids=ids_arr, pops=pops_arr, ploidy=ploidy)


# Minimal self-test / example usage

if __name__ == "__main__":
    # Example use:
    #
    #   python py_multiclust/io.py
    #
    # from the project root. Tries to load tests/data/wolves.str and
    # prints basic summary information.
    import os

    example_path = os.path.join("tests", "data", "wolves.str")
    if os.path.exists(example_path):
        print(f"Loading example STRUCTURE file: {example_path}")
        data = load_stru(example_path, ploidy=2, has_ids=True, has_pops=True)
        print("  n_ind :", data.n_ind)
        print("  n_loci:", data.n_loci)
        print("  ploidy:", data.ploidy)
        if data.ids is not None:
            print("  first 5 IDs :", data.ids[:5])
        if data.pops is not None:
            print("  first 5 pops:", data.pops[:5])
    else:
        print("Example file tests/data/wolves.str not found; nothing to do.")
