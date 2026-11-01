#!/usr/bin/env python3
"""Block B8 runner: sweep_live_b5.py unchanged, except that distinct-predicate cells (name contains "_dist") append
b8_predicates to the corpus's predicate list after setup_corpus. Prompts of the first 16 predicates, which the harness
uses for its token statistics, stay identical. Usage as sweep_live_b5.py with --freeze, --freeze-sha, --cell, --seed."""
import sys

import b8_predicates as P
import sweep_live_b5 as H


def main():
    if "_dist" in sys.argv[sys.argv.index("--cell") + 1]:
        setup = H.setup_corpus

        def setup_extended(corpus):
            setup(corpus)
            H.PREDICATES = list(H.PREDICATES) + {"arxiv": P.ARXIV_EXTRA, "pubmed": P.PUBMED_EXTRA}[corpus]

        H.setup_corpus = setup_extended
    H.main()


if __name__ == "__main__":
    main()
