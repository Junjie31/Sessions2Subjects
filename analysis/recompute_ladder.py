"""Reproduce Table 1: paired MACC differences across aggregation levels.

Reads ``data/paired_scores.csv`` (de-identified per-session MACC records for all
six directed cross-domain pairs) and computes, for each direction:

  * L0  session x seed  (``n = 1836`` for MMPD-target, 177/126 otherwise)
  * L1  session         (median over seeds within a session)
  * L2  subject         (median over seeds, then median over sessions)

For every level it reports n, median/mean paired difference, win rate,
positive/negative/zero counts, the two-sided sign-flip p-value, and (subject
level only) the Hodges-Lehmann estimate + interval, the subject-cluster
bootstrap 95% CI for the mean, the rank-biserial correlation, and the ICC(1) /
DEFF / effective-N diagnostics. The six subject-level p-values are then
Benjamini-Hochberg adjusted together (Table 1's ``q`` column).

Outputs ``results/final_numbers.csv`` and ``results/final_numbers.json``.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from analysis.statistics import (SEEDS, DIRECTION_ORDER, bh_fdr, cluster_bootstrap_mean,
                                 hodges_lehmann, icc1, paired_diff_subject, rank_biserial,
                                 sign_flip_p, _session_diff, _session_seed_diff)
CSV = f"{BASE}/data/paired_scores.csv"
OUTDIR = f"{BASE}/results"
os.makedirs(OUTDIR, exist_ok=True)


def main():
    df = pd.read_csv(CSV)
    rows = []
    # The four earlier directions and the two MMPD-source directions were each
    # analyzed in a separate process with their own generator seeded to
    # 20260826; mirror that here so the Monte Carlo draws reproduce the
    # reported p-values exactly.
    OLD_DIRS = {"PURE_MMPD", "UBFC_MMPD", "PURE_UBFC", "UBFC_PURE"}
    rng_old = np.random.default_rng(20260826)
    rng_new = np.random.default_rng(20260826)

    for direction in DIRECTION_ORDER:
        rng = rng_old if direction in OLD_DIRS else rng_new
        sub = df[df["direction"] == direction]

        l2 = paired_diff_subject(df, direction, SEEDS)
        l2_d = np.array(list(l2.values()))
        l1_d = _session_diff(df, direction, SEEDS)
        l0_d = _session_seed_diff(df, direction, SEEDS)

        for level_name, diffs, subj_group in [
            ("session×seed", l0_d, None),
            ("session", l1_d, None),
            ("subject", l2_d, l2),
        ]:
            d = np.asarray(diffs, float)
            n = len(d)
            med = np.median(d)
            mean = np.mean(d)
            win = np.mean(d > 0)
            pos = int((d > 0).sum())
            neg = int((d < 0).sum())
            zero = int((d == 0).sum())
            p = sign_flip_p(d, rng=rng)
            rb = rank_biserial(d)

            if level_name == "subject":
                hl, hl_lo, hl_hi = hodges_lehmann(d, rng=rng)
                ci_lo, ci_hi = cluster_bootstrap_mean(subj_group, rng=rng)
                subj_macc = defaultdict(list)
                for _, r in sub.iterrows():
                    if r["seed"] in SEEDS:
                        subj_macc[r["subject_id"]].append(r["MACC"])
                icc_res = icc1(subj_macc)
                if icc_res:
                    icc_val, mbar, ng, N = icc_res
                    deff = 1 + (mbar - 1) * icc_val
                    n_eff = N / deff
                else:
                    icc_val = mbar = ng = N = deff = n_eff = np.nan
            else:
                hl = hl_lo = hl_hi = ci_lo = ci_hi = np.nan
                icc_val = mbar = ng = N = deff = n_eff = np.nan

            rows.append({
                "direction": direction, "level": level_name, "n": n,
                "median_diff": med, "mean_diff": mean, "win_rate": win,
                "pos": pos, "neg": neg, "zero": zero,
                "sign_flip_p": p,
                "hl": hl, "hl_ci_lo": hl_lo, "hl_ci_hi": hl_hi,
                "bootstrap_ci_lo": ci_lo, "bootstrap_ci_hi": ci_hi,
                "rank_biserial": rb,
                "icc1": icc_val, "mean_cluster_size": mbar, "deff": deff, "n_eff": n_eff,
            })

    out = pd.DataFrame(rows)

    # BH-FDR across the six subject-level comparisons (Table 1 q column).
    subj_rows = out[out["level"] == "subject"].reset_index(drop=True)
    q = bh_fdr(subj_rows["sign_flip_p"].values)

    # Write bh_q / bh_sig into the output for every row (NaN/False for the
    # non-subject levels, which are not part of the six-test family).
    q_map = dict(zip(subj_rows["direction"], q))
    out["bh_q"] = np.nan
    out["bh_sig"] = False
    mask = out["level"].eq("subject")
    out.loc[mask, "bh_q"] = out.loc[mask, "direction"].map(q_map).astype(float)
    out.loc[mask, "bh_sig"] = out.loc[mask, "bh_q"] <= 0.05
    subj_rows = out[mask].reset_index(drop=True)

    out.to_csv(f"{OUTDIR}/final_numbers.csv", index=False)
    out.to_json(f"{OUTDIR}/final_numbers.json", orient="records", indent=2)

    print("=" * 96)
    print("Three-level MACC statistics (six directions)")
    print("=" * 96)
    for direction in DIRECTION_ORDER:
        dr = out[out["direction"] == direction]
        print(f"\n{direction}:")
        for _, r in dr.iterrows():
            extra = ""
            if r["level"] == "subject":
                extra = (f" mean={r['mean_diff']:+.4f} CI=[{r['bootstrap_ci_lo']:+.4f},"
                         f"{r['bootstrap_ci_hi']:+.4f}] HL={r['hl']:+.4f}")
            print(f"  {r['level']:12s} n={r['n']:5d} med={r['median_diff']:+.5f}"
                  f" win={r['win_rate']:.1%} p={r['sign_flip_p']:.4f} rb={r['rank_biserial']:+.3f}{extra}")
        l0 = dr[dr["level"] == "session×seed"]["sign_flip_p"].iloc[0]
        l2 = dr[dr["level"] == "subject"]["sign_flip_p"].iloc[0]
        collapse = abs(np.log10(max(l0, 1e-16)) - np.log10(max(l2, 1e-16)))
        print(f"  collapse_orders = |log10(L0_p) - log10(L2_p)| = {collapse:.1f}")

    print("\n--- Subject-level with BH-q (six-direction family) ---")
    for _, r in subj_rows.iterrows():
        flag = "SIG" if r["bh_sig"] else ""
        print(f"  {r['direction']:12s} n={r['n']:2d} med={r['median_diff']:+.4f} "
              f"p={r['sign_flip_p']:.4f} q={r['bh_q']:.4f} {flag}")

    print(f"\nWrote: results/final_numbers.csv / .json")


if __name__ == "__main__":
    main()
