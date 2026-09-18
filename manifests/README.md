# Run Identity Manifest

`run_identity.csv` records the identity assertions for the 36 training runs
described in the paper (six directed pairs × two modes × three seeds).

## Columns

| column | meaning |
|--------|---------|
| `direction` | `SRC_TGT` directed pair |
| `mode` | `none` (baseline) or `osc_real` (oscillatory variant) |
| `seed` | training seed 42 / 100 / 202 |
| `checkpoint_file` | checkpoint basename |
| `checkpoint_sha256` | full SHA-256 of the checkpoint file |
| `test_manifest_sha256` | SHA-256 of the sorted unique test-session list |
| `n_sessions` | number of unique test sessions |

All 36 runs are recorded with both a checkpoint hash and a test-manifest hash.
The checkpoint hash identifies the exact model weights analyzed; the
test-manifest hash identifies the exact set of test sessions (derived from
`data/paired_scores.csv`).

## Verification

Recompute the checkpoint hashes from local files and compare against this
manifest with `scripts/verify_identity.py`.
