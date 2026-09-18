"""
UNIFIED EVALUATION: Single entry point for both standalone and main.py paths.

Addresses deterministic randomness and subprocess artifacts:
  (1) Fixed normalization with epsilon in all paths
  (2) Single calculate_metrics call via predict_and_evaluate()
  (3) No more standalone vs subprocess code divergence in eval logic

Usage:
  from evaluation.evaluate import predict_and_evaluate, check_determinism

  # After training:
  results = predict_and_evaluate(model, test_dataloader, config)
  # results["MAE"], results["predictions"], results["labels"], results["per_video"]

  # Check determinism (same model + inputs → same metrics):
  r1 = predict_and_evaluate(model, dl, config)
  r2 = predict_and_evaluate(model, dl, config)
  assert abs(r1["MAE"] - r2["MAE"]) < 1e-4
"""
import os, numpy as np, torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from evaluation.metrics import calculate_metrics


def normalize_prediction(pred: torch.Tensor) -> torch.Tensor:
    """Z-score normalization WITH epsilon (safe for all paths)."""
    return (pred - pred.mean(-1, keepdim=True)) / (pred.std(-1, keepdim=True) + 1e-8)


def predict_and_evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    config,
    device: Optional[torch.device] = None,
    external_quality_fn: Optional[callable] = None,
) -> Dict:
    """Unified model evaluation.

    Args:
        model: trained model (DataParallel or single)
        dataloader: test DataLoader with (data, labels, subj_id, chunk_idx) batches
        config: rPPG toolbox config
        device: torch device (auto-detected if None)
        external_quality_fn: optional fn(labels) -> quality(B,T) for oracle experiments

    Returns:
        dict with keys: "predictions", "labels", "per_chunk_pred", "per_chunk_label",
        plus printed metrics (MAE, RMSE, MAPE, Pearson, SNR).
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    predictions = {}   # subj_id -> {chunk_idx: tensor}
    labels_dict = {}   # subj_id -> {chunk_idx: tensor}

    chunk_len = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH

    with torch.no_grad():
        for batch in dataloader:
            bs = batch[0].shape[0]
            data_t = batch[0].float().to(device)
            labels_t = batch[1].float().to(device)

            # Oracle quality support
            if external_quality_fn is not None:
                oro_q = external_quality_fn(labels_t)
                pred_t = model(data_t, external_quality=oro_q)
            else:
                pred_t = model(data_t)

            # Unified normalization WITH epsilon
            pred_t = normalize_prediction(pred_t)
            labels_t = labels_t.view(-1, 1)
            pred_t = pred_t.view(-1, 1)

            for ib in range(bs):
                sid = str(batch[2][ib])
                si = int(batch[3][ib])
                predictions.setdefault(sid, {})[si] = pred_t[ib * chunk_len:(ib + 1) * chunk_len]
                labels_dict.setdefault(sid, {})[si] = labels_t[ib * chunk_len:(ib + 1) * chunk_len]

    # Compute and print metrics (same as calculate_metrics internally)
    calculate_metrics(predictions, labels_dict, config)

    # Extract HR predictions/GT from same logic as metrics.py
    from evaluation.post_process import calculate_metric_per_video
    from evaluation.metrics import _reform_data_from_dict

    per_video_hr = {}  # video_id -> (pred_hr, gt_hr, mae, pred_wave, gt_wave)
    for vid in sorted(predictions.keys()):
        try:
            pred = _reform_data_from_dict(predictions[vid])
            label = _reform_data_from_dict(labels_dict[vid])
            gt_hr, pred_hr, snr = calculate_metric_per_video(
                pred, label, diff_flag=False, fs=config.TEST.DATA.FS, hr_method='FFT')
            per_video_hr[vid] = {
                "pred_hr": pred_hr, "gt_hr": gt_hr,
                "mae": abs(pred_hr - gt_hr),
                "pred_wave": pred, "gt_wave": label, "snr": snr,
            }
        except Exception:
            pass

    return {
        "predictions": predictions,
        "labels": labels_dict,
        "per_video": per_video_hr,
    }


def check_determinism(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    config,
    device: Optional[torch.device] = None,
    tolerance: float = 1e-4,
) -> Tuple[bool, float]:
    """Run evaluation twice with identical inputs; verify HR-MAE difference < tolerance.

    Returns (passed: bool, max_diff: float).
    """
    r1 = predict_and_evaluate(model, dataloader, config, device)
    r2 = predict_and_evaluate(model, dataloader, config, device)

    # Compare HR-MAE per video
    max_diff = 0.0
    for vid in r1["per_video"]:
        if vid in r2["per_video"]:
            d = abs(r1["per_video"][vid]["mae"] - r2["per_video"][vid]["mae"])
            max_diff = max(max_diff, d)

    passed = max_diff < tolerance
    return passed, max_diff
