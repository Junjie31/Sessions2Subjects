# Third-Party Licenses

This repository builds on the following open-source projects. Their original
license terms remain in force for the corresponding portions of the code.

## RhythmMamba

Portions of the model implementation in `rppg/` (in particular
`rppg/neural_methods/model/RhythmMamba_fusion*.py`) are derived from
[RhythmMamba](https://github.com/zizheng-guo/RhythmMamba).

Copyright (c) 2024 Zizheng Guo
Licensed under the MIT License.

## rPPG-Toolbox

The dataset loaders, evaluation metrics, and training scaffold in `rppg/`
(`rppg/dataset/`, `rppg/evaluation/`, `rppg/config.py`, `rppg/main.py`) are
adapted from [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox).

Licensed under the Apache License 2.0. See the upstream project for details.

## mamba-ssm

`rppg/mamba_ssm_scan/` is a local fork of
[mamba-ssm](https://github.com/state-spaces/mamba), modified to add an
oscillatory state-transition matrix. It reuses the compiled
`selective_scan_cuda` kernels provided by the `mamba-ssm` Python package.

Licensed under the Apache License 2.0 (originally by Tri Dao and Albert Gu).
See the upstream project for details.

---

The paired score records in `data/paired_scores.csv` are released separately
under the terms described in `DATA_LICENSE.md`.
