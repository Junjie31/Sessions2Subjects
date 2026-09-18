"""Verify the run-identity manifest against local checkpoint files.

Reads ``manifests/run_identity.csv`` (the committed artifact), then, for each
run, recomputes the checkpoint SHA-256 and the test-manifest SHA-256 from the
local files and reports whether they match. This lets a reviewer confirm that
the scores in ``data/paired_scores.csv`` correspond to the recorded checkpoints.

The checkpoint files themselves are not distributed; place them under
``PreTrainedModels/`` (or pass ``--checkpoint-dir``) before running.

Usage:
    python scripts/verify_identity.py [--checkpoint-dir /path/to/PreTrainedModels]
"""
import argparse
import hashlib
import os
import sys

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = f"{BASE}/manifests/run_identity.csv"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default=f"{BASE}/PreTrainedModels")
    args = parser.parse_args()

    if not os.path.exists(MANIFEST):
        print(f"missing {MANIFEST}")
        sys.exit(1)

    mf = pd.read_csv(MANIFEST)
    scores = pd.read_csv(f"{BASE}/data/paired_scores.csv")

    n_pass = n_missing = n_mismatch = 0
    for _, row in mf.iterrows():
        direction, mode, seed = row["direction"], row["mode"], row["seed"]
        name = f"{direction}_{mode}_S{seed}"
        ckpt = None
        if row["checkpoint_file"]:
            import glob
            matches = glob.glob(os.path.join(args.checkpoint_dir, "**", f"{name}_Epoch29.pth"), recursive=True)
            ckpt = matches[0] if matches else None

        # test-manifest check (always possible from the scores)
        sess = sorted(set(scores[(scores["direction"] == direction) & (scores["seed"] == seed)]["session_id"]))
        tm_hash = sha256_text("\n".join(sess))
        tm_ok = tm_hash == row["test_manifest_sha256"]

        if ckpt is None:
            n_missing += 1
            print(f"[MISSING] {name}: checkpoint not found")
            continue
        ckpt_hash = sha256_file(ckpt)
        ckpt_ok = ckpt_hash == row["checkpoint_sha256"]
        if ckpt_ok and tm_ok:
            n_pass += 1
            print(f"[PASS] {name}")
        else:
            n_mismatch += 1
            print(f"[MISMATCH] {name}: checkpoint_ok={ckpt_ok} test_manifest_ok={tm_ok}")

    print(f"\n{len(mf)} runs: {n_pass} pass, {n_mismatch} mismatch, {n_missing} missing checkpoint")


if __name__ == "__main__":
    main()
