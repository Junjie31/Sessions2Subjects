# From Sessions to Subjects: Statistical Inference in Cross-Domain rPPG Evaluation

Code and de-identified paired score records for the submission
*"From Sessions to Subjects: Statistical Inference in Cross-Domain rPPG
Evaluation"* (under review).

The paper is about **evaluation methodology**, not a new architecture: repeated
recordings (sessions) from the same subject, combined with several training
runs, are often counted as independent samples in cross-domain rPPG comparisons.
That underestimates uncertainty and can make a non-existent model difference
look significant. This repository reproduces the reported effect estimates,
sign-flip p-values, BH-adjusted q-values, and significance decisions exactly;
the subject-cluster bootstrap interval bounds may differ at the 4th decimal
(Monte Carlo noise from the resampling seed).

---

## Layout

```
Sessions2Subjects/
├── data/
│   ├── paired_scores.csv        # de-identified MACC records, 6 directions
│   └── null_calibration.csv     # pre-computed Type-I error simulation
├── analysis/                    # the methodology (CPU only)
│   ├── statistics.py            # sign-flip / HL / cluster bootstrap / ICC / BH-FDR
│   ├── recompute_ladder.py      # → Table 1 (three aggregation levels + BH-q)
│   └── null_calibration.py      # → Type-I inflation vs ICC
├── manifests/                   # run-identity audit (checkpoint SHA-256, etc.)
│   ├── run_identity.csv
│   └── README.md
├── rppg/                        # the model (GPU; the motivating case)
│   ├── main.py, config.py
│   ├── neural_methods/          # RhythmMamba_fusion_scan (none / osc_real)
│   ├── mamba_ssm_scan/          # the oscillatory-A SSM fork
│   ├── dataset/ evaluation/     # PURE / UBFC / MMPD loaders + metrics
│   └── unsupervised_methods/
└── scripts/                     # gen configs / train / infer / build / verify
```

---

## Reproduce the paper's statistics (no GPU)

```bash
pip install -r requirements-analysis.txt

# Table 1: three-level paired differences + six-direction BH-FDR
python analysis/recompute_ladder.py            # → results/final_numbers.{csv,json}

# Re-run the null-calibration simulation (a few minutes)
python analysis/null_calibration.py            # → data/null_calibration.csv
```

`data/paired_scores.csv` already contains the de-identified per-session MACC
records, so the analysis runs offline.

---

## Reproduce the motivating case (GPU)

Requirements: Python 3.8+, CUDA, PyTorch 2.x, and the `mamba-ssm` CUDA kernels.

```bash
pip install -r requirements.txt   # includes mamba-ssm (CUDA kernels)

python scripts/gen_configs.py --data-root /path/to/Data   # 36 configs
bash  scripts/run_cross.sh                                 # train, resumable
python scripts/inference.py                                # per-session MACC
python scripts/build_paired_scores.py                      # symmetry asserts
python scripts/verify_identity.py                          # provenance audit
python analysis/recompute_ladder.py                        # back to the stats
```

`run_cross.sh` trains six directions × two modes (`none`, `osc_real`) × three
matched seeds (42/100/202) = 36 runs, skipping any with an existing checkpoint.
`verify_identity.py` writes `manifests/run_identity.csv` with the checkpoint and
test-manifest SHA-256 hashes and the checkpoint-to-dataset mapping check
described in the paper.

---

## Statistical protocol (Sec. 3)

The paired difference (oscillatory `A_osc` vs baseline `none`) is aggregated
median-over-seeds within a session, then median-over-sessions within a subject:

| Unit | Definition | MMPD-target `n` |
|------|------------|-----------------|
| session×seed | `d_isr` | 1836 |
| session | `d_is` | 612 |
| subject | `D_i` (primary) | 31 |

Methods (all in `analysis/statistics.py`): two-sided sign-flip permutation test
(mean statistic; exact for `n ≤ 20`, else 100,000 draws, seed 20260826),
Hodges–Lehmann estimator + bootstrap interval, subject-cluster bootstrap
percentile interval for the **mean** (10,000 resamples), rank-biserial
correlation, ICC(1)/DEFF/effective-`n`, and Benjamini–Hochberg FDR adjustment
across the six subject-level comparisons.

The null-calibration simulation (`analysis/null_calibration.py`) uses 2,000
replications and 2,000 sign-flip draws per test; the simulated cluster sizes are
one subject with 12 recordings and 30 subjects with 20 (matching mini-MMPD's
31 subjects / 612 sessions).

---

## Data format (`data/paired_scores.csv`)

One row per (direction, mode, seed, session): `direction` (`SRC_TGT`), `mode`
(`none`/`osc_real`), `seed` (42/100/202), `subject_id`, `session_id`, `MACC`.
Six directions: PURE→MMPD (31 subjects), UBFC→MMPD (31), MMPD→PURE (10),
MMPD→UBFC (42), PURE→UBFC (42), UBFC→PURE (10).

Licensing: code under `LICENSE.txt` (with third-party attribution in
`THIRD_PARTY_LICENSES.md`); score records under `DATA_LICENSE.md`.

---

## Citation

Under review.
