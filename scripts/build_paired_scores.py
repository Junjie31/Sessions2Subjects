"""Build ``data/paired_scores.csv`` from the per-session MACC cache, with hard
symmetry assertions.

Reads ``data/per_session_macc.pkl`` (``(direction, mode, session, seed) -> macc``)
and writes the de-identified long-form table consumed by the analysis scripts.
The assertions guarantee the paired design is symmetric: for every direction x
seed x session there is exactly one ``none`` and one ``osc_real`` record, the
seed sets match, and there are no duplicate/NaN/Inf MACC values.

Usage:
    python scripts/build_paired_scores.py
"""
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from analysis.statistics import DIRECTION_KIND, DIRECTION_ORDER, SEEDS, subject_of
CACHE = f"{BASE}/data/per_session_macc.pkl"
OUT = f"{BASE}/data/paired_scores.csv"


def main():
    if not os.path.exists(CACHE):
        print(f"missing {CACHE}; run scripts/inference.py first")
        sys.exit(1)
    cache = pickle.load(open(CACHE, "rb"))

    rows = []
    for direction in DIRECTION_ORDER:
        kind = DIRECTION_KIND[direction]
        for mode in ["none", "osc_real"]:
            for seed in SEEDS:
                for (dd, mm, session, sd), v in cache.items():
                    if dd == direction and mm == mode and sd == seed:
                        rows.append({
                            "direction": direction, "mode": mode, "seed": seed,
                            "subject_id": subject_of(session, kind),
                            "session_id": session, "MACC": float(v),
                        })
    df = pd.DataFrame(rows)

    errors = []
    for direction in DIRECTION_ORDER:
        sub = df[df["direction"] == direction]
        none_df = sub[sub["mode"] == "none"]
        osc_df = sub[sub["mode"] == "osc_real"]
        if set(none_df["seed"]) != set(osc_df["seed"]):
            errors.append(f"{direction}: asymmetric seed sets")
        for seed in SEEDS:
            n_sess = set(none_df[none_df["seed"] == seed]["session_id"])
            o_sess = set(osc_df[osc_df["seed"] == seed]["session_id"])
            if n_sess != o_sess:
                errors.append(f"{direction}/S{seed}: asymmetric session sets")
        key_count = sub.groupby(["seed", "session_id"])["mode"].nunique()
        if (key_count != 2).any():
            errors.append(f"{direction}: {(key_count != 2).sum()} (seed,session) not exactly 2 modes")
    if df.duplicated(subset=["direction", "mode", "seed", "session_id"]).any():
        errors.append("duplicate (direction,mode,seed,session)")
    if df["MACC"].isna().any() or np.isinf(df["MACC"]).any():
        errors.append("NaN/Inf MACC")

    if errors:
        print("Assertion failures:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print("All assertions passed")

    df.to_csv(OUT, index=False)
    print(f"Wrote: {OUT} ({len(df)} rows)")
    for direction in DIRECTION_ORDER:
        g = df[df["direction"] == direction]
        print(f"  {direction:12s} n_session={g['session_id'].nunique():4d} "
              f"n_subject={g['subject_id'].nunique():3d}")


if __name__ == "__main__":
    main()
