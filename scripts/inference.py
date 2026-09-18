"""Per-session waveform (MACC) inference for all six directions.

Loads each trained checkpoint (``{DIR}_{mode}_S{seed}_Epoch29.pth`` under
``PreTrainedModels/``), runs it on the test split, and records the per-session
MACC into a cache pickle: ``(direction, mode, session, seed) -> macc``. This
cache is the de-identified score record that feeds ``build_paired_scores.py``.

Progress is saved atomically after every (direction, mode, seed) run, and a run
is skipped only when it is marked complete in ``data/inference_done.json`` — a
partial or interrupted run is re-run, never skipped on the basis of a single
cache entry.

Usage:
    python scripts/inference.py [--directions PURE_MMPD,UBFC_MMPD,...]
"""
import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RPPG = f"{BASE}/rppg"
sys.path.insert(0, RPPG)

SEEDS = [42, 100, 202]
MODES = ["none", "osc_real"]
DIRECTIONS = ["PURE_MMPD", "UBFC_MMPD", "MMPD_PURE", "MMPD_UBFC",
              "PURE_UBFC", "UBFC_PURE"]
# target dataset per direction (for subject mapping and test loader)
TEST_DS = {"PURE_MMPD": "MMPD", "UBFC_MMPD": "MMPD", "MMPD_PURE": "PURE",
           "MMPD_UBFC": "UBFC", "PURE_UBFC": "UBFC", "UBFC_PURE": "PURE"}
# expected number of test sessions per direction (unique sessions)
EXPECTED = {"PURE_MMPD": 612, "UBFC_MMPD": 612, "MMPD_PURE": 59,
            "MMPD_UBFC": 42, "PURE_UBFC": 42, "UBFC_PURE": 59}
CACHE = f"{BASE}/data/per_session_macc.pkl"
DONE = f"{BASE}/data/inference_done.json"


def find_ckpt(direction, mode, seed):
    name = f"{direction}_{mode}_S{seed}"
    matches = glob.glob(f"{BASE}/PreTrainedModels/*/{name}_Epoch29.pth")
    return matches[0] if matches else None


def atomic_save(cache):
    tmp = CACHE + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(cache, f)
    os.replace(tmp, CACHE)


def load_done() -> set:
    if os.path.exists(DONE):
        return set(tuple(k) for k in json.load(open(DONE)))
    return set()


def save_done(done: set):
    json.dump([list(k) for k in sorted(done)], open(DONE, "w"), indent=2)


def run_inference(direction, mode, seed, cache) -> bool:
    """Run one (direction, mode, seed); return True if cache changed (save needed)."""
    import torch
    from config import get_config
    from neural_methods.model.RhythmMamba_fusion_scan import RhythmMamba_fusion_scan
    from evaluation.waveform_metrics import compute_waveform_metrics
    from evaluation.evaluate import normalize_prediction
    from dataset import data_loader
    from torch.utils.data import DataLoader

    ckpt = find_ckpt(direction, mode, seed)
    if not ckpt:
        print(f"  [skip] {direction}/{mode}/S{seed}: MISSING ckpt")
        return False

    cfg_path = f"{RPPG}/configs/train_configs/{direction}_{mode}_S{seed}.yaml"
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default=cfg_path)
    args = parser.parse_args([])
    config = get_config(args)

    device = torch.device(config.DEVICE)
    chunk_len = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
    test_ds = TEST_DS[direction]

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random as rnd
    rnd.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    g = torch.Generator()
    g.manual_seed(seed)

    def sw(w):
        ws = torch.initial_seed() % 2 ** 32
        np.random.seed(ws)
        rnd.seed(ws)

    L = {'PURE': data_loader.PURELoader.PURELoader,
         'UBFC': data_loader.UBFCrPPGLoader.UBFCrPPGLoader,
         'MMPD': data_loader.MMPDLoader.MMPDLoader}
    tsd = L[test_ds](name='test', data_path=config.TEST.DATA.DATA_PATH,
                     config_data=config.TEST.DATA)
    dl = DataLoader(tsd, num_workers=1, batch_size=16, shuffle=False,
                    worker_init_fn=sw, generator=g)

    mdl = RhythmMamba_fusion_scan(
        depth=24, embed_dim=96, grid_size=3, modulation_mode=mode,
        quality_scale_init=0.0 if mode == "none" else 0.1).to(device)
    mdl = torch.nn.DataParallel(mdl, device_ids=[0])
    mdl.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    mdl.eval()

    preds, lbls = {}, {}
    with torch.no_grad():
        for batch in dl:
            bs = batch[0].shape[0]
            dt_in, lt_in = batch[0].float().to(device), batch[1].float().to(device)
            pt = mdl(dt_in)
            pt = normalize_prediction(pt)
            lt_in, pt = lt_in.view(-1, 1), pt.view(-1, 1)
            for ib in range(bs):
                sid, si = str(batch[2][ib]), int(batch[3][ib])
                preds.setdefault(sid, {})[si] = pt[ib * chunk_len:(ib + 1) * chunk_len]
                lbls.setdefault(sid, {})[si] = lt_in[ib * chunk_len:(ib + 1) * chunk_len]
    m = compute_waveform_metrics(preds, lbls, fs=config.TEST.DATA.FS)
    n_new = 0
    for session, metrics in m.items():
        if metrics.get('macc') is not None and not np.isnan(metrics['macc']):
            cache[(direction, mode, session, seed)] = float(metrics['macc'])
            n_new += 1
    print(f"  [done] {direction}/{mode}/S{seed}: {n_new} sessions")
    return n_new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directions', default=",".join(DIRECTIONS))
    args = parser.parse_args()
    dirs = args.directions.split(",")

    cache = {}
    if os.path.exists(CACHE):
        cache = pickle.load(open(CACHE, "rb"))
    done = load_done()

    for direction in dirs:
        for mode in MODES:
            for seed in SEEDS:
                key = (direction, mode, seed)
                if key in done:
                    print(f"[skip] {direction}/{mode}/S{seed}: already complete")
                    continue
                n = run_inference(direction, mode, seed, cache)
                if n > 0:
                    atomic_save(cache)
                if n == EXPECTED[direction]:
                    done.add(key)
                    save_done(done)
                elif n > 0:
                    print(f"  [warn] {direction}/{mode}/S{seed}: got {n} sessions, "
                          f"expected {EXPECTED[direction]} (will re-run)")

    print(f"\nSaved: {CACHE} ({len(cache)} keys)")


if __name__ == "__main__":
    main()
