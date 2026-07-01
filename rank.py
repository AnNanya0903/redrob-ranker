#!/usr/bin/env python3
"""
rank.py — Produces the top-100 ranked CSV submission for the Redrob
"Intelligent Candidate Discovery & Ranking Challenge".

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Design constraints honored (submission_spec.md Section 3):
    - CPU only, no GPU
    - No network calls
    - Streams the 100K-line JSONL (doesn't require loading gzip separately)
    - Runs in a few seconds to low tens of seconds on a single core,
      well inside the 5-minute / 16GB budget.

Approach: a transparent, rule-based ranker over structured features
(skills-with-trust-scoring, title relevance, company-type, experience
band, location, notice period) combined with a multiplicative behavioral
modifier from redrob_signals, and a honeypot filter. See README.md for
the full rationale — this is deliberately NOT an embeddings/LLM ranker,
both because the compute budget forbids per-candidate LLM calls at 100K
scale, and because a fully-interpretable scorer makes the reasoning
column verifiably grounded in the candidate's actual profile.
"""

import argparse
import csv
import gzip
import json
import sys
import time
from pathlib import Path

from features import score_candidate
from reasoning import build_reasoning

TOP_N = 100


def open_candidates(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def load_and_score(path: Path, verbose=True):
    results = []
    n = 0
    n_honeypot = 0
    with open_candidates(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            candidate = json.loads(line)
            r = score_candidate(candidate)
            n += 1
            if r["is_honeypot"]:
                n_honeypot += 1
            results.append((candidate["candidate_id"], r["score"], candidate, r))
    if verbose:
        print(f"Scored {n} candidates ({n_honeypot} flagged as honeypots).", file=sys.stderr)
    return results


def rank_top_n(results, n=TOP_N):
    # Sort by score desc, tie-break by candidate_id ascending (per spec section 3).
    results.sort(key=lambda x: (-x[1], x[0]))
    return results[:n]


def write_csv(top, out_path: Path):
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (cid, score, candidate, result) in enumerate(top, start=1):
            reasoning = build_reasoning(candidate, result)
            writer.writerow([cid, rank, f"{score:.6f}", reasoning])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, type=Path,
                     help="Path to candidates.jsonl or candidates.jsonl.gz")
    ap.add_argument("--out", required=True, type=Path,
                     help="Output CSV path (top-100 ranking)")
    args = ap.parse_args()

    t0 = time.time()
    results = load_and_score(args.candidates)
    top = rank_top_n(results)
    write_csv(top, args.out)
    elapsed = time.time() - t0
    print(f"Wrote top-{len(top)} ranking to {args.out} in {elapsed:.1f}s.", file=sys.stderr)


if __name__ == "__main__":
    main()
