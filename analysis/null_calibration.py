"""Null-calibration simulation: Type-I error under zero effect.

Compares two resampling schemes for the sign-flip test at a nominal ``alpha =
0.05`` when the true model difference is zero but sessions are clustered within
subjects:

  (a) naive session-level sign-flip: each session x seed paired difference is an
      independent flip unit (ignores within-subject dependence);
  (b) subject-block sign-flip: the sign is flipped jointly for all records of a
      subject (the whole subject is the resampling unit).

Data-generating process (Sec. 4): ``d_{isr} = b_i + e_{isr}`` with subject random
effect ``b_i ~ N(0, sigma_b^2)`` and session x seed noise ``e_{isr} ~
N(0, sigma_w^2)``, so the within-subject correlation of the paired differences is
``ICC = sigma_b^2 / (sigma_b^2 + sigma_w^2)``. Subject count ``S = 31`` and a
cluster-size distribution mimicking mini-MMPD (12--20 recordings, median 20).

Writes ``data/null_calibration.csv``. Pure CPU; fixed seed 20260826.
"""
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SEED = 20260826
rng = np.random.default_rng(SEED)

S = 31          # subjects
R = 3           # seeds per session
N_REP = 2000    # simulation replications
N_SIGN = 2000   # sign-flip resamples per test
ALPHA = 0.05


def mmpd_cluster_sizes() -> np.ndarray:
    # approx. mini-MMPD: one subject with 12 recordings, 30 subjects with 20.
    return np.array([12] + [20] * 30)


def gen_data(icc, cluster_sizes) -> dict[int, np.ndarray]:
    sigma_w = 1.0
    sigma_b = np.sqrt(icc / (1 - icc)) * sigma_w if icc < 1 else 1e6
    data = {}
    for i, m in enumerate(cluster_sizes):
        b_i = rng.normal(0, sigma_b)
        vals = [b_i + rng.normal(0, sigma_w) for _ in range(m * R)]
        data[i] = np.array(vals)
    return data


def sign_flip_test(diffs, block_ids, n_sign=N_SIGN) -> float:
    obs = np.mean(diffs)
    subjects = np.unique(block_ids)
    subj_idx = np.searchsorted(subjects, block_ids)
    n_subj = len(subjects)
    signs = rng.choice([-1.0, 1.0], size=(n_sign, n_subj))
    flip = signs[:, subj_idx]
    means = (diffs[None, :] * flip).mean(axis=1)
    count = int((np.abs(means) >= abs(obs)).sum())
    return (count + 1) / (n_sign + 1)


def naive_sign_flip_test(diffs, n_sign=N_SIGN) -> float:
    obs = np.mean(diffs)
    n = len(diffs)
    signs = rng.choice([-1.0, 1.0], size=(n_sign, n))
    means = (diffs[None, :] * signs).mean(axis=1)
    count = int((np.abs(means) >= abs(obs)).sum())
    return (count + 1) / (n_sign + 1)


def run_calibration(icc, cluster_sizes) -> tuple[float, float]:
    typeI_block = typeI_naive = 0
    for _ in range(N_REP):
        data = gen_data(icc, cluster_sizes)
        diffs, block_ids = [], []
        for i, vals in data.items():
            for v in vals:
                diffs.append(v)
                block_ids.append(i)
        diffs = np.array(diffs)
        block_ids = np.array(block_ids)
        if sign_flip_test(diffs, block_ids) < ALPHA:
            typeI_block += 1
        if naive_sign_flip_test(diffs) < ALPHA:
            typeI_naive += 1
    return typeI_block / N_REP, typeI_naive / N_REP


def main():
    cluster_sizes = mmpd_cluster_sizes()
    iccs = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]
    results = []
    print("Null-calibration: Type-I error, alpha=0.05, zero effect")
    print(f"S={S} subjects, R={R} seeds, cluster sizes 12-20 (median 20), {N_REP} reps")
    print(f"{'ICC':>5} | {'subject-block':>14} | {'naive session':>14} | {'inflation':>10}")
    for icc in iccs:
        t_block, t_naive = run_calibration(icc, cluster_sizes)
        results.append((icc, t_block, t_naive))
        print(f"{icc:5.2f} | {t_block:14.3f} | {t_naive:14.3f} | {t_naive - t_block:+10.3f}")

    pd.DataFrame({
        'icc': np.array([r[0] for r in results]),
        'typeI_subject_block': np.array([r[1] for r in results]),
        'typeI_naive_session': np.array([r[2] for r in results]),
        'inflation': np.array([r[2] - r[1] for r in results]),
    }).to_csv(f'{BASE}/data/null_calibration.csv', index=False)

    print(f"\nWrote: data/null_calibration.csv")


if __name__ == "__main__":
    main()
