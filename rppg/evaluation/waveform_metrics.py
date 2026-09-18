"""
DENSE WAVEFORM METRICS: Per-video evaluation beyond HR-MAE.

============================================================================
SIGN CONVENTIONS for paired_wilcoxon_test(metrics_a, metrics_b):
  median_diff = median(metric_a - metric_b)  ← ALWAYS osc_real - none

BETTER direction for each metric (osc better means osc_real > none for ↑, < for ↓):
  pearson_r:  ↑ better (higher correlation = better)
  macc:       ↑ better (higher cross-correlation = better)
  snr_db:     ↑ better (higher HR peak / band energy = cleaner spectrum)
  hr_mae:     ↓ better (lower error = better)

"favorable" = (显著 AND 方向对 osc 有利): pearson_r/macc/snr_db 的 median_diff>0,
             hr_mae 的 median_diff<0

Metrics computed per video/window:
  - Waveform Pearson r (pred BVP vs GT BVP)
  - MACC: Maximum Amplitude of Cross-Correlation
  - Frequency-domain SNR (HR peak energy / in-band energy, 0.7-4 Hz)
  - HR-MAE (retained as control)
  - HRV: SDNN, RMSSD, LF/HF from peak-derived inter-beat intervals

Usage:
  from evaluation.waveform_metrics import compute_waveform_metrics, paired_wilcoxon_test

  results = compute_waveform_metrics(predictions_dict, labels_dict, config)
  paired_wilcoxon_test(osc_metrics, none_metrics)
============================================================================
"""
import numpy as np
from scipy import stats, signal
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def _reform_from_dict(data_dict: dict) -> np.ndarray:
    """Sort chunks by index, concatenate to 1D waveform."""
    sorted_items = sorted(data_dict.items(), key=lambda x: x[0])
    tensors = [x[1].cpu().numpy().flatten() for x in sorted_items]
    return np.concatenate(tensors)


def compute_hrv_from_peaks(peaks: np.ndarray, fs: float) -> Dict:
    """Compute HRV metrics from beat peak indices.
    Args:
        peaks: indices of detected beats in the waveform
        fs: sampling frequency (Hz)
    Returns:
        dict with SDNN, RMSSD, mean_ibi_s, valid (bool)
    """
    if len(peaks) < 3:
        return {"SDNN": None, "RMSSD": None, "LF_HF": None, "valid": False, "n_beats": len(peaks)}

    ibi = np.diff(peaks) / fs  # inter-beat intervals in seconds
    ibi = ibi[(ibi > 0.3) & (ibi < 2.0)]  # filter: 30-200 bpm
    if len(ibi) < 2:
        return {"SDNN": None, "RMSSD": None, "LF_HF": None, "valid": False, "n_beats": len(peaks)}

    sdnn = np.std(ibi) * 1000  # ms
    rmssd = np.sqrt(np.mean(np.diff(ibi) ** 2)) * 1000  # ms

    # LF/HF: simple PSD of IBI (requires longer segments)
    lf_hf = None
    if len(ibi) > 10:
        try:
            ibi_time = np.cumsum(ibi)
            ibi_interp = np.interp(np.linspace(0, ibi_time[-1], max(64, len(ibi)*4)), ibi_time, ibi)
            freqs, psd = signal.welch(ibi_interp, fs=4.0, nperseg=min(64, len(ibi_interp)))
            lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
            hf_mask = (freqs >= 0.15) & (freqs <= 0.40)
            lf = np.trapz(psd[lf_mask], freqs[lf_mask]) if lf_mask.sum() > 0 else 0
            hf = np.trapz(psd[hf_mask], freqs[hf_mask]) if hf_mask.sum() > 0 else 1e-8
            lf_hf = lf / hf
        except Exception:
            pass

    return {"SDNN": sdnn, "RMSSD": rmssd, "LF_HF": lf_hf, "valid": True, "n_beats": len(peaks)}


def compute_waveform_metrics(
    predictions: Dict[str, Dict[int, np.ndarray]],
    labels: Dict[str, Dict[int, np.ndarray]],
    fs: float = 30.0,
    min_seconds: float = 5.0,
    detrend: bool = True,
) -> Dict[str, Dict]:
    """Compute dense per-video waveform metrics.

    Args:
        predictions: {video_id: {chunk_idx: tensor}} from evaluation
        labels: {video_id: {chunk_idx: tensor}}
        fs: sampling frequency
        min_seconds: minimum video length for reliable HRV
        detrend: apply scipy detrend before correlation

    Returns:
        {video_id: {metric_name: value, ...}}
    """
    results = {}
    for vid in sorted(predictions.keys()):
        try:
            pred = _reform_from_dict(predictions[vid])
            gt = _reform_from_dict(labels[vid])

            # Ensure same length
            L = min(len(pred), len(gt))
            pred, gt = pred[:L], gt[:L]

            if L < fs * min_seconds:
                continue

            if detrend:
                pred = signal.detrend(pred)
                gt = signal.detrend(gt)

            # Waveform Pearson r
            pearson_r, pearson_p = stats.pearsonr(pred, gt)

            # MACC
            xcorr = signal.correlate(pred - np.mean(pred), gt - np.mean(gt), mode='full')
            macc = np.max(np.abs(xcorr)) / (np.std(pred) * np.std(gt) * L + 1e-8)

            # Frequency-domain SNR
            freqs, psd_pred = signal.welch(pred, fs=fs, nperseg=min(256, L//2))
            _, psd_gt = signal.welch(gt, fs=fs, nperseg=min(256, L//2))
            hr_band = (freqs >= 0.7) & (freqs <= 4.0)

            if hr_band.sum() > 0:
                # Find HR peak in GT
                gt_band = psd_gt[hr_band]
                peak_idx = np.argmax(gt_band)
                # SNR: peak power / (total band power - peak power)
                peak_power = psd_pred[hr_band][max(0, peak_idx-1):min(len(psd_pred[hr_band]), peak_idx+2)].sum()
                band_power = psd_pred[hr_band].sum()
                snr = 10 * np.log10(peak_power / (band_power - peak_power + 1e-8))
            else:
                snr = 0.0

            # HR-MAE (control — same FFT HR as metrics.py)
            from evaluation.post_process import calculate_metric_per_video
            try:
                gt_hr, pred_hr, _ = calculate_metric_per_video(pred, gt, diff_flag=False, fs=fs, hr_method='FFT')
                hr_mae = abs(pred_hr - gt_hr)
            except Exception:
                hr_mae = None

            # HRV via peak detection
            try:
                peaks_pred = signal.find_peaks(pred, distance=int(fs * 0.3), prominence=np.std(pred) * 0.3)[0]
                peaks_gt = signal.find_peaks(gt, distance=int(fs * 0.3), prominence=np.std(gt) * 0.3)[0]
            except Exception:
                peaks_pred, peaks_gt = [], []

            hrv_pred = compute_hrv_from_peaks(peaks_pred, fs)
            hrv_gt = compute_hrv_from_peaks(peaks_gt, fs)

            results[vid] = {
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "macc": macc,
                "snr_db": snr,
                "hr_mae": hr_mae,
                "hrv_pred": {"SDNN": hrv_pred["SDNN"], "RMSSD": hrv_pred["RMSSD"], "LF_HF": hrv_pred["LF_HF"], "valid": hrv_pred["valid"]},
                "hrv_gt": {"SDNN": hrv_gt["SDNN"], "RMSSD": hrv_gt["RMSSD"], "LF_HF": hrv_gt["LF_HF"], "valid": hrv_gt["valid"]},
                "duration_sec": L / fs,
            }
        except Exception:
            pass

    return results


def paired_wilcoxon_test(
    metrics_a: Dict[str, Dict],
    metrics_b: Dict[str, Dict],
    metric_name: str,
    hrv_submetric: Optional[str] = None,
    min_sample: int = 10,
) -> Dict:
    """Paired video-level Wilcoxon signed-rank test between two methods.

    Args:
        metrics_a, metrics_b: outputs of compute_waveform_metrics for two methods
        metric_name: "pearson_r", "macc", "snr_db", "hr_mae"
        hrv_submetric: if metric_name=="hrv", then submetric like "SDNN", "RMSSD"
        min_sample: minimum matched pairs required

    Returns:
        dict with n, median_diff, p_value, matched_pairs_rank_biserial
    """
    common = sorted(set(metrics_a.keys()) & set(metrics_b.keys()))
    if len(common) < min_sample:
        return {"n": len(common), "error": "too few matched pairs"}

    vals_a, vals_b = [], []
    for vid in common:
        if metric_name in ("hrv_pred", "hrv_gt"):
            if hrv_submetric and metrics_a[vid][metric_name]["valid"] and metrics_b[vid][metric_name]["valid"]:
                va = metrics_a[vid][metric_name][hrv_submetric]
                vb = metrics_b[vid][metric_name][hrv_submetric]
                if va is not None and vb is not None:
                    vals_a.append(va); vals_b.append(vb)
            continue
        va = metrics_a[vid].get(metric_name)
        vb = metrics_b[vid].get(metric_name)
        if va is not None and vb is not None and not np.isnan(va) and not np.isnan(vb):
            vals_a.append(va); vals_b.append(vb)

    if len(vals_a) < min_sample:
        return {"n": len(vals_a), "error": "too few valid values"}

    vals_a = np.array(vals_a); vals_b = np.array(vals_b)
    diff = vals_a - vals_b

    # 全零差时 Wilcoxon 无定义 (zero_method 报错), 返回 p=1.0
    if np.allclose(diff, 0.0):
        return {"n": len(diff), "median_diff": 0.0, "p_value": 1.0,
                "effect_size": 0.0, "mean_a": float(np.mean(vals_a)),
                "mean_b": float(np.mean(vals_b)), "std_a": float(np.std(vals_a, ddof=1)),
                "std_b": float(np.std(vals_b, ddof=1)), "note": "all-zero-diff"}

    # Wilcoxon
    stat, p = stats.wilcoxon(vals_a, vals_b)
    median_diff = np.median(diff)

    # Effect size: matched-pairs rank-biserial
    n = len(diff)
    ranks = stats.rankdata(np.abs(diff))
    W_pos = np.sum(ranks[diff > 0])
    W_neg = np.sum(ranks[diff < 0])
    # rank-biserial = (W_pos - W_neg) / (W_pos + W_neg) ... or (W_pos/(W_pos+W_neg) - 0.5) * 2
    if (W_pos + W_neg) > 0:
        effect = (W_pos - W_neg) / (W_pos + W_neg)
    else:
        effect = 0.0

    return {
        "n": n,
        "median_diff": median_diff,
        "p_value": p,
        "effect_size": effect,
        "mean_a": np.mean(vals_a),
        "mean_b": np.mean(vals_b),
        "std_a": np.std(vals_a, ddof=1),
        "std_b": np.std(vals_b, ddof=1),
    }


def dump_waveforms_npy(predictions, labels, output_dir: str):
    """Save pred/GT waveforms as npy files for later re-analysis.

    Args:
        predictions: {video_id: {chunk_idx: tensor}}
        labels: {video_id: {chunk_idx: tensor}}
        output_dir: directory to save .npy files
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    for vid in predictions:
        pred = _reform_from_dict(predictions[vid])
        gt = _reform_from_dict(labels[vid])
        np.save(os.path.join(output_dir, f"{vid}_pred.npy"), pred)
        np.save(os.path.join(output_dir, f"{vid}_gt.npy"), gt)


def compute_all_paired_tests(
    metrics_osc: Dict, metrics_none: Dict,
    direction: str = "",
) -> List[Dict]:
    """Run paired Wilcoxon on all supported metrics, return as table rows."""
    rows = []
    for mname, hrv_sub in [("pearson_r", None), ("macc", None), ("snr_db", None),
                             ("hr_mae", None), ("hrv_pred", "SDNN"), ("hrv_pred", "RMSSD")]:
        label = f"{mname}" + (f"/{hrv_sub}" if hrv_sub else "")
        r = paired_wilcoxon_test(metrics_osc, metrics_none, mname, hrv_sub)
        if "error" in r:
            continue
        rows.append({
            "direction": direction,
            "metric": label,
            "n": r["n"],
            "mean_osc": r["mean_a"],
            "mean_none": r["mean_b"],
            "median_diff": r["median_diff"],
            "p_value": r["p_value"],
            "effect_size": r["effect_size"],
        })
    return rows
