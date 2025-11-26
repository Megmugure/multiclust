"""
compare_indivq.py

Small helper script to compare the C and Python MULTICLUST outputs.

Goal
----
The C `multiclust` binary and the Python CLI both produce "indivq" files,
but the formats are slightly different:

    C example:
        0 CAN001024 (x) 1 : 0.000000 1.000000

    Python example:
        CAN001024 1 0.000000 1.000000

This script does three things:

  1. Parses both formats in a robust way (ignoring decorative tokens like
     indices, "(x)", and ":").
  2. Aligns individuals by ID and compares their Q vectors (admixture
     proportions).
  3. Prints summary diagnostics:
        - number of overlapping individuals
        - mean / max absolute differences in Q
        - agreement in hard cluster assignments (argmax)
        - whether population labels match
        - first few individuals with both Q vectors side by side

Usage
-----
From the project root:

    python compare_indivq.py \
        out_c_wolves/wolves.str.mix.K=2.indivq \
        out_py_wolves/wolves.str.mix.K=2.indivq
"""

import sys
import numpy as np


def is_float_token(tok: str) -> bool:
    """
    Return True if `tok` can be parsed as a float, False otherwise.

    This is used to distinguish:
      - index-like tokens ("0", "1", "2", ...)
      - actual Q values ("0.000000", "1.000000")
    from string IDs like "CAN001024".
    """
    try:
        float(tok)
        return True
    except ValueError:
        return False


def read_indivq(path):
    """
    Robust reader for both C and Python indivq formats.

    We do not assume fixed column positions; instead we scan each line
    and interpret tokens based on simple rules.

    Examples
    --------
    C-style lines:
        0 CAN001024 (x) 1 : 0.000000 1.000000

    Python-style lines:
        CAN001024 1 0.000000 1.000000

    Strategy
    --------
      - Find the first non-numeric token that is not '(x)' or ':' → ID.
      - Next non '(x)' / ':' token after the ID → POP (population label).
      - All remaining tokens that can be parsed as floats → Q values.

    Parameters
    ----------
    path : str
        Path to an indivq-like file from either the C or Python run.

    Returns
    -------
    ids : np.ndarray (dtype=object)
        Individual IDs in the order they appear in the file.
    pops : np.ndarray (dtype=object)
        Population labels (stored as strings).
    Q : np.ndarray (float)
        Q[i, k] = admixture proportion for individual i in cluster k.

    Raises
    ------
    ValueError
        If no valid rows with Q values are found.
    """
    ids = []
    pops = []
    Q = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            toks = line.split()
            if len(toks) < 3:
                continue

            # -------- find ID (first non-numeric, non-decorative token) --------
            id_idx = None
            for i, tok in enumerate(toks):
                if tok in {"(x)", ":"}:
                    continue
                # skip numeric tokens (indices, counts, etc.)
                if is_float_token(tok):
                    continue
                id_idx = i
                break

            if id_idx is None:
                # can't find a plausible ID; skip this line
                continue

            id_tok = toks[id_idx]

            # -------- find POP (first non '(x)' / ':' token after ID) --------
            pop_idx = None
            pop_tok = None
            for j in range(id_idx + 1, len(toks)):
                tok = toks[j]
                if tok in {"(x)", ":"}:
                    continue
                pop_idx = j
                pop_tok = tok
                break

            if pop_idx is None:
                # no population label found; skip
                continue

            # -------- collect Q values (all remaining float-parsable tokens) ---
            q_tokens = toks[pop_idx + 1 :]
            q_vals = []
            for q in q_tokens:
                if is_float_token(q):
                    q_vals.append(float(q))

            if not q_vals:
                # no Q values found; skip
                continue

            ids.append(id_tok)
            pops.append(pop_tok)
            Q.append(q_vals)

    if not Q:
        raise ValueError(f"No valid Q rows parsed from {path}")

    return (
        np.array(ids, dtype=object),
        np.array(pops, dtype=object),
        np.array(Q, dtype=float),
    )


def main(argv=None) -> None:
    """
    Compare C and Python indivq files and print a short report.

    Parameters
    ----------
    argv : list[str] or None
        Optional argument list. If None, uses sys.argv[1:].
    """
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) != 2:
        print(f"Usage: python {sys.argv[0]} C_indivq Python_indivq")
        sys.exit(1)

    c_path, py_path = argv

    # Read both files
    ids_c, pops_c, Q_c = read_indivq(c_path)
    ids_p, pops_p, Q_p = read_indivq(py_path)

    print("C file:   ", c_path)
    print("  n_ind:", len(ids_c), "Q shape:", Q_c.shape)
    print("Py file:  ", py_path)
    print("  n_ind:", len(ids_p), "Q shape:", Q_p.shape)

    # Map IDs -> index for Python file
    idx_p = {id_: i for i, id_ in enumerate(ids_p)}

    # Only compare individuals that appear in both outputs
    common_ids = [id_ for id_ in ids_c if id_ in idx_p]
    if not common_ids:
        print("No overlapping IDs between the two files!")
        sys.exit(1)

    Q_c_aligned = []
    Q_p_aligned = []
    pops_c_aligned = []
    pops_p_aligned = []

    for id_ in common_ids:
        i_c = np.where(ids_c == id_)[0][0]
        i_p = idx_p[id_]
        Q_c_aligned.append(Q_c[i_c])
        Q_p_aligned.append(Q_p[i_p])
        pops_c_aligned.append(pops_c[i_c])
        pops_p_aligned.append(pops_p[i_p])

    Q_c_aligned = np.vstack(Q_c_aligned)
    Q_p_aligned = np.vstack(Q_p_aligned)
    pops_c_aligned = np.array(pops_c_aligned)
    pops_p_aligned = np.array(pops_p_aligned)

    if Q_c_aligned.shape != Q_p_aligned.shape:
        print("Warning: Q shapes differ:", Q_c_aligned.shape, Q_p_aligned.shape)

    diff = np.abs(Q_c_aligned - Q_p_aligned)

    print(f"\nNumber of individuals compared: {len(common_ids)}")
    print(f"Q mean abs diff: {diff.mean():.6g}")
    print(f"Q max abs diff : {diff.max():.6g}")

    # Compare hard cluster assignments (argmax over K)
    assign_c = Q_c_aligned.argmax(axis=1)
    assign_p = Q_p_aligned.argmax(axis=1)
    agree = (assign_c == assign_p)
    print(
        f"Hard cluster assignment agreement: {agree.sum()}/{len(agree)} "
        f"({100.0 * agree.mean():.1f}%)"
    )

    # Check that population labels match
    pop_mismatch = (pops_c_aligned != pops_p_aligned)
    if pop_mismatch.any():
        print("Warning: population labels differ for some IDs between files.")
    else:
        print("Population labels match for all individuals.")

    # Show a few rows for quick visual inspection
    print("\nFirst 5 individuals (ID, C_Q, Py_Q):")
    for i, id_ in enumerate(common_ids[:5]):
        print(
            id_,
            "C:",
            " ".join(f"{x:.3f}" for x in Q_c_aligned[i]),
            "Py:",
            " ".join(f"{x:.3f}" for x in Q_p_aligned[i]),
        )


if __name__ == "__main__":
    main()
