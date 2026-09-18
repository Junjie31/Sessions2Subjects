"""Generate training configs for all six directed cross-domain pairs.

Directions (train -> test) and their splits, matching Sec. 3 of the paper:

  * PURE->MMPD : train PURE 0.0-0.8,        test MMPD 0.0-1.0
  * UBFC->MMPD : train UBFC 0.0-0.72,       test MMPD 0.0-1.0
  * MMPD->PURE : train MMPD 0.0-0.7,        test PURE 0.0-1.0
  * MMPD->UBFC : train MMPD 0.0-0.7,        test UBFC 0.0-1.0
  * PURE->UBFC : train PURE 0.0-0.8,        test UBFC 0.0-1.0
  * UBFC->PURE : train UBFC 0.0-0.72,       test PURE 0.0-1.0

Two modulation modes per direction: ``none`` (baseline) and ``osc_real``
(oscillatory state-transition variant), each with three matched seeds (42/100/202)
= 36 configs. ``USE_LAST_EPOCH: true`` means the valid split is not used for model
selection.

Edit ``DATA_ROOT`` below (or pass ``--data-root``) to point at the downloaded
PURE / UBFC-rPPG / mini-MMPD directories.

Usage:
    python scripts/gen_configs.py [--data-root /path/to/Data]
"""
import argparse
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = f"{BASE}/rppg/configs/train_configs"
os.makedirs(OUTDIR, exist_ok=True)

SEEDS = [42, 100, 202]
MODES = {
    "none":     {"MODULATION_MODE": "none",     "QUALITY_SCALE_INIT": 0.0},
    "osc_real": {"MODULATION_MODE": "osc_real", "QUALITY_SCALE_INIT": 0.1},
}

# train-split END (fraction) per source dataset; valid = END..1.0
SOURCE_SPLIT = {"PURE": 0.8, "UBFC": 0.72, "MMPD": 0.7}

# MMPD requires an INFO filter block (lighting/motion/etc. conditions).
MMPD_INFO = """    INFO:
      LIGHT: [1, 2, 3, 4]
      MOTION: [1, 2, 3, 4]
      EXERCISE: [1, 2]
      SKIN_COLOR: [3, 4, 5, 6]
      GENDER: [1, 2]
      GLASSER: [1, 2]
      HAIR_COVER: [1, 2]
      MAKEUP: [1, 2]
"""

PREPROCESS = """    PREPROCESS:
      DATA_TYPE: ['Standardized']
      LABEL_TYPE: Standardized
      DO_CHUNK: True
      CHUNK_LENGTH: 160
      CROP_FACE:
        DO_CROP_FACE: True
        USE_LARGE_FACE_BOX: True
        LARGE_BOX_COEF: 1.5
        DETECTION:
          DO_DYNAMIC_DETECTION: False
          DYNAMIC_DETECTION_FREQUENCY: 30
          USE_MEDIAN_FACE_BOX: False
      RESIZE:
        H: 128
        W: 128
"""

# the six directed pairs (source, target)
DIRECTIONS = [("PURE", "MMPD"), ("UBFC", "MMPD"), ("MMPD", "PURE"),
              ("MMPD", "UBFC"), ("PURE", "UBFC"), ("UBFC", "PURE")]


def data_block(dataset, begin, end, data_root, with_info):
    path = f"{data_root}/{dataset}/" if dataset != "MMPD" else f"{data_root}/mini_MMPD/"
    info = MMPD_INFO if with_info else ""
    return f"""  DATA:
{info}    FS: 30
    DATASET: {dataset}
    DO_PREPROCESS: False
    DATA_FORMAT: NDCHW
    DATA_PATH: "{path}"
    CACHED_PATH: "{path}"
    EXP_DATA_NAME: ""
    BEGIN: {begin}
    END: {end}
{PREPROCESS}"""


def gen_config(src, tgt, mode, seed, data_root):
    name = f"{src}_{tgt}_{mode}_S{seed}"
    m = MODES[mode]
    src_end = SOURCE_SPLIT[src]
    train_info = (src == "MMPD")
    test_info = (tgt == "MMPD")
    return f"""BASE: ['']
TOOLBOX_MODE: "train_and_test"
TRAIN:
  BATCH_SIZE: 16
  EPOCHS: 30
  LR: 0.0003
  MODEL_FILE_NAME: {name}
  AUG: 0
{data_block(src, '0.0', src_end, data_root, train_info)}
VALID:
{data_block(src, src_end, '1.0', data_root, train_info)}
TEST:
  METRICS: ['MAE','RMSE','MAPE','Pearson','SNR']
  USE_LAST_EPOCH: true
{data_block(tgt, '0.0', '1.0', data_root, test_info)}
DEVICE: cuda:0
NUM_OF_GPU_TRAIN: 1
LOG:
  PATH: {BASE}/logs
MODEL:
  DROP_RATE: 0.2
  NAME: RhythmMamba_fusion_scan
  GRID_SIZE: 3
  MODULATION_MODE: {m['MODULATION_MODE']}
  QUALITY_SCALE_INIT: {m['QUALITY_SCALE_INIT']}
  QUALITY_MODE: fusion
  OSC_LEARN_OMEGA: true
  MODEL_DIR: "{BASE}/PreTrainedModels"
INFERENCE:
  BATCH_SIZE: 2
  EVALUATION_METHOD: "FFT"
  EVALUATION_WINDOW:
    USE_SMALLER_WINDOW: False
    WINDOW_SIZE: 10
  MODEL_PATH: ""
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default=os.environ.get("RPPG_DATA_ROOT", "~/Workspace/Data"),
                        help="Directory containing PURE/, UBFC/, mini_MMPD/ subdirs.")
    args = parser.parse_args()
    data_root = os.path.expanduser(args.data_root)

    n = 0
    for src, tgt in DIRECTIONS:
        for mode in MODES:
            for seed in SEEDS:
                name = f"{src}_{tgt}_{mode}_S{seed}"
                content = gen_config(src, tgt, mode, seed, data_root)
                with open(os.path.join(OUTDIR, f"{name}.yaml"), "w") as f:
                    f.write(content)
                n += 1
    print(f"生成 {n} 个 config（6 方向 × 2 模式 × 3 seed）→ {OUTDIR}")


if __name__ == "__main__":
    main()
