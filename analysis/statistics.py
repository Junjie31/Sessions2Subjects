"""Statistical methods for the ICASSP 2027 paper.

Unified, dependency-light implementations of every inference method used in
``From Sessions to Subjects: Statistical Inference in Cross-Domain rPPG
Evaluation``. These functions operate on de-identified paired score records
(``data/paired_scores.csv``) and reproduce Table 1 and Fig. 2.

Methods (Sec. 3 of the paper):
  * subject-level paired differences ``D_i``
  * two-sided sign-flip permutation test (exact for ``n<=20``, Monte Carlo else)
  * Hodges-Lehmann estimator + bootstrap interval
  * subject-cluster bootstrap (percentile, mean statistic -- the primary effect)
  * rank-biserial correlation
  * ICC(1) / design effect (DEFF) / effective sample size
  * Benjamini-Hochberg FDR correction (monotonic from the back)

All stochastic procedures use a fixed seed (20260826), matching the paper.
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

SEED = 20260826
SEEDS = [42, 100, 202]

# directed train->test pairs used in the paper, with the *target* dataset that
# determines how a session id maps to a subject id.
DIRECTION_KIND = {
    "PURE_MMPD": "MMPD", "UBFC_MMPD": "MMPD",
    "MMPD_PURE": "PURE", "MMPD_UBFC": "UBFC",
    "PURE_UBFC": "UBFC", "UBFC_PURE": "PURE",
}
# Grouping: the first four directions and the two MMPD-source directions each
# use their own generator stream in recompute_ladder.py; this order keeps the
# two streams separate so the Monte Carlo draws reproduce the reported values.
DIRECTION_ORDER = ["PURE_MMPD", "UBFC_MMPD", "PURE_UBFC", "UBFC_PURE",
                   "MMPD_PURE", "MMPD_UBFC"]


def subject_of(session_id: str, kind: str) -> str:
    """Map a session id to a subject id.

    PURE encodes subject/trial as ``subject*100 + trial`` (e.g. ``101`` =
    subject 1 trial 1, ``1001`` = subject 10 trial 1), so the subject is
    ``int(session) // 100``. Slicing ``session[:2]`` would merge subject 1 and
    subject 10. MMPD / UBFC session ids start with ``subjectNN``.
    """
    if kind in ("MMPD", "UBFC"):
        m = re.match(r"subject\d+", session_id)
        return m.group(0) if m else session_id.split("_")[0]
    if kind == "PURE":
        return str(int(session_id) // 100)
    return "UNKNOWN"


def paired_diff_subject(df: pd.DataFrame, direction: str,
                        seeds: list[int] | None = None) -> dict[str, float]:
    """Subject-level paired differences ``D_i = tilde(y_osc)_i - tilde(y_none)_i``.

    Aggregation order (Sec. 3): median over seeds within a session, then median
    over sessions within a subject, then oscillatory minus baseline.
    """
    seeds = seeds if seeds is not None else SEEDS
    sub = df[df["direction"] == direction]
    sess_vals = defaultdict(lambda: {"none": [], "osc": []})
    session_subj: dict[str, str] = {}
    for _, r in sub.iterrows():
        if r["seed"] not in seeds:
            continue
        sess_vals[r["session_id"]]["none" if r["mode"] == "none" else "osc"].append(r["MACC"])
        session_subj[r["session_id"]] = r["subject_id"]

    subj_vals = defaultdict(lambda: {"none": [], "osc": []})
    for session, d in sess_vals.items():
        subj = session_subj[session]
        if d["none"]:
            subj_vals[subj]["none"].append(np.median(d["none"]))
        if d["osc"]:
            subj_vals[subj]["osc"].append(np.median(d["osc"]))

    return {subj: np.median(d["osc"]) - np.median(d["none"])
            for subj, d in subj_vals.items() if d["none"] and d["osc"]}


def _session_diff(df: pd.DataFrame, direction: str, seeds: list[int]) -> np.ndarray:
    """Session-level paired differences (median over seeds within a session)."""
    sub = df[df["direction"] == direction]
    sess_vals = defaultdict(lambda: {"none": [], "osc": []})
    for _, r in sub.iterrows():
        if r["seed"] in seeds:
            sess_vals[r["session_id"]]["none" if r["mode"] == "none" else "osc"].append(r["MACC"])
    return np.array([np.median(v["osc"]) - np.median(v["none"])
                     for v in sess_vals.values() if v["none"] and v["osc"]])


def _session_seed_diff(df: pd.DataFrame, direction: str, seeds: list[int]) -> np.ndarray:
    """Session x seed-level paired differences ``d_{isr}``."""
    sub = df[df["direction"] == direction]
    pairs = defaultdict(dict)
    for _, r in sub.iterrows():
        if r["seed"] in seeds:
            pairs[(r["seed"], r["session_id"])][r["mode"]] = r["MACC"]
    return np.array([v["osc_real"] - v["none"] for v in pairs.values()
                     if "none" in v and "osc_real" in v])


def sign_flip_p(diffs, n_perm: int = 100_000, rng: np.random.Generator | None = None,
                batch_size: int = 2_000) -> float:
    """Two-sided sign-flip permutation p-value, mean statistic.

    For ``n <= 20`` all ``2**n`` sign assignments are enumerated exactly;
    otherwise ``n_perm`` random sign draws are used with ``p = (b + 1) / (B + 1)``.
    The sign matrix is processed in batches to keep peak memory bounded
    (``batch_size * n`` instead of ``n_perm * n``).
    """
    rng = rng if rng is not None else np.random.default_rng(SEED)
    # NOTE: the input order is preserved (not sorted) so that the Monte Carlo
    # sign draws reproduce the original analysis stream when a shared rng is
    # passed in by recompute_ladder.py.
    d = np.asarray(diffs, float)
    n = len(d)
    obs = abs(np.mean(d))
    if n <= 20:
        count = 0
        for bits in range(2 ** n):
            signs = np.array([1 if (bits >> i) & 1 else -1 for i in range(n)])
            count += abs(np.mean(d * signs)) >= obs
        return count / (2 ** n)

    count = 0
    completed = 0
    while completed < n_perm:
        b = min(batch_size, n_perm - completed)
        signs = rng.choice([-1.0, 1.0], size=(b, n))
        means = (signs @ d) / n
        count += int(np.count_nonzero(np.abs(means) >= obs))
        completed += b
    return (count + 1) / (n_perm + 1)


def hodges_lehmann(diffs, n_boot: int = 5000, rng: np.random.Generator | None = None) -> tuple[float, float, float]:
    """Hodges-Lehmann point estimate (median of pairwise Walsh averages) + bootstrap interval.

    The pairwise Walsh averages are computed with vectorized upper-triangle
    indexing, and the bootstrap distribution is computed in a single vectorized
    pass (``n_boot x n(n+1)/2``) instead of a Python double loop.
    """
    rng = rng if rng is not None else np.random.default_rng(SEED)
    d = np.asarray(diffs, float)
    n = len(d)
    iu = np.triu_indices(n)
    hl = float(np.median((d[iu[0]] + d[iu[1]]) / 2))

    idx = rng.integers(0, n, size=(n_boot, n))
    db = d[idx]                                  # (n_boot, n)
    pw = (db[:, iu[0]] + db[:, iu[1]]) / 2.0     # (n_boot, n(n+1)/2)
    boots = np.median(pw, axis=1)
    return hl, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def cluster_bootstrap_mean(subject_diffs, n_boot: int = 10_000,
                           rng: np.random.Generator | None = None) -> tuple[float, float]:
    """Subject-cluster bootstrap percentile interval for the *mean* (primary statistic).

    Resamples whole subjects with replacement; training runs are held fixed.
    """
    rng = rng if rng is not None else np.random.default_rng(SEED)
    vals = np.asarray(list(subject_diffs.values()), float)
    n = len(vals)
    boots = np.array([np.mean(vals[rng.integers(0, n, size=n)]) for _ in range(n_boot)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def cluster_bootstrap_median(subject_diffs, n_boot: int = 10_000,
                             rng: np.random.Generator | None = None) -> tuple[float, float]:
    """Subject-cluster bootstrap percentile interval for the median (descriptive)."""
    rng = rng if rng is not None else np.random.default_rng(SEED)
    vals = np.asarray(list(subject_diffs.values()), float)
    n = len(vals)
    boots = np.array([np.median(vals[rng.integers(0, n, size=n)]) for _ in range(n_boot)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def rank_biserial(diffs) -> float:
    """Rank-biserial correlation (Cliff's delta analogue on matched pairs)."""
    d = np.asarray(diffs, float)
    ranks = stats.rankdata(np.abs(d))
    wp = np.sum(ranks[d > 0])
    wn = np.sum(ranks[d < 0])
    return float((wp - wn) / (wp + wn)) if (wp + wn) > 0 else 0.0


def icc1(values_by_group) -> tuple[float, float, int, int] | None:
    """One-way ANOVA ICC(1), equal-cluster approximation.

    Returns ``(icc, mean_cluster_size, n_groups, N)``.
    """
    groups = {g: np.asarray(v, float) for g, v in values_by_group.items() if len(v) > 0}
    ng = len(groups)
    if ng < 2:
        return None
    allv = np.concatenate(list(groups.values()))
    gm = allv.mean()
    N = len(allv)
    ssb = sum(len(v) * (v.mean() - gm) ** 2 for v in groups.values())
    ssw = sum(((v - v.mean()) ** 2).sum() for v in groups.values())
    msb = ssb / (ng - 1)
    msw = ssw / (N - ng)
    ks = np.array([len(v) for v in groups.values()])
    k0 = (1 / (ng - 1)) * (ks.sum() - (ks ** 2).sum() / ks.sum())
    icc = (msb - msw) / (msb + (k0 - 1) * msw)
    return float(icc), float(ks.mean()), ng, N


def bh_fdr(pvalues: list[float] | np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted q-values, enforced monotonic from the back."""
    p = np.asarray(pvalues, float)
    n = len(p)
    if n == 0:
        return p.copy()
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, n + 1)
    q = p * n / ranks
    # enforce monotonicity: q[i] = min(q[i], q[i+1]) walking back from the largest
    for i in range(n - 2, -1, -1):
        q[order[i]] = min(q[order[i]], q[order[i + 1]])
    return q
