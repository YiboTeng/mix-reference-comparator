#!/usr/bin/env python3
"""Experimental stereo-field diagnostics for vocal tracks.

This script does not attempt to identify a commercial plugin.  It measures
evidence associated with fixed delay/doubling, time-varying modulation,
double tracking, and diffuse reverb-like stereo fields.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from matplotlib import font_manager
from scipy import signal


PLUGIN_ID = "mix-reference-comparator"
ANALYSIS_MODE = "vocal-reference"
EPS = 1e-15
BANDS = [(80, 150), (150, 300), (300, 600), (600, 1200),
         (1200, 2500), (2500, 5000), (5000, 8000),
         (8000, 12000), (12000, 18000)]


def db(value):
    return 20.0 * np.log10(np.maximum(value, EPS))


def find_font():
    for candidate in [r"C:\Windows\Fonts\msyh.ttc",
                      r"C:\Windows\Fonts\simhei.ttf"]:
        if Path(candidate).exists():
            return font_manager.FontProperties(fname=candidate)
    return font_manager.FontProperties()


FONT = find_font()


def fp(size=11, weight="normal"):
    return font_manager.FontProperties(fname=FONT.get_file(), size=size,
                                       weight=weight) if FONT.get_file() else FONT.copy()


def confidence_title(title, level, limitation):
    """给每个子图单独标注当前 A/B 解释的可信度和主要限制。"""
    return f"{title}\n可信度：{level}｜{limitation}"


def load_stereo(path: Path, target_sr: int):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]
    if sr != target_sr:
        divisor = math.gcd(sr, target_sr)
        audio = signal.resample_poly(audio, target_sr // divisor,
                                     sr // divisor, axis=0).astype(np.float32)
        sr = target_sr
    return audio.astype(np.float64), sr


def strongest_window(mono, sr, seconds):
    length = min(len(mono), int(seconds * sr))
    if len(mono) <= length:
        return 0, len(mono)
    hop = max(1, sr // 10)
    squared = mono * mono
    cumulative = np.concatenate([[0.0], np.cumsum(squared)])
    starts = np.arange(0, len(mono) - length, hop)
    energy = (cumulative[starts + length] - cumulative[starts]) / length
    start = int(starts[np.argmax(energy)])
    return start, start + length


def bandpass(x, sr, lo=200, hi=12000, order=4):
    hi = min(hi, sr / 2 - 20)
    sos = signal.butter(order, [lo, hi], btype="bandpass", fs=sr,
                        output="sos")
    return signal.sosfiltfilt(sos, x, axis=0)


def frame_rms(x, sr, win_ms=50, hop_ms=10):
    win = max(1, int(sr * win_ms / 1000))
    hop = max(1, int(sr * hop_ms / 1000))
    if len(x) < win:
        return np.array([np.sqrt(np.mean(x * x) + EPS)])
    cumulative = np.concatenate([[0.0], np.cumsum(x * x)])
    starts = np.arange(0, len(x) - win + 1, hop)
    values = (cumulative[starts + win] - cumulative[starts]) / win
    return np.sqrt(np.maximum(values, 0.0) + EPS)


def ms_waveforms(audio):
    left, right = audio[:, 0], audio[:, 1]
    return (left + right) / 2.0, (left - right) / 2.0


def stage_width_and_tails(audio, sr):
    mid, side = ms_waveforms(audio)
    mid = bandpass(mid, sr)
    side = bandpass(side, sr)
    mr = frame_rms(mid, sr)
    srms = frame_rms(side, sr)
    mdb = db(mr)
    ratio = db(srms / (mr + EPS))
    top = float(np.percentile(mdb, 99))
    relative = mdb - top
    definitions = {
        "core": (-6, 0.01), "body": (-15, -6),
        "tails": (-30, -15), "very_quiet": (-45, -30),
    }
    stages = {}
    for key, (lo, hi) in definitions.items():
        mask = (relative >= lo) & (relative < hi)
        stages[key] = float(np.median(ratio[mask])) if np.any(mask) else None

    # Find downward transitions followed by at least 300 ms of low Mid energy.
    events, last = [], -1000
    horizon = 60  # 600 ms, because hop is 10 ms
    for index in range(12, len(relative) - horizon):
        if index - last < horizon:
            continue
        before = relative[index - 10:index]
        after = relative[index:index + 30]
        if np.max(before) > -8 and np.median(after) < -18:
            curve = db(srms[index:index + horizon] + EPS)
            curve = curve - np.median(curve[:5])
            events.append(curve)
            last = index
    if events:
        average = np.median(np.stack(events), axis=0)
        time = np.arange(len(average)) * 0.01
        # 只拟合前 300 ms；再往后很容易被下一句新的人声污染。
        fit_mask = time <= 0.3
        slope = float(np.polyfit(time[fit_mask], average[fit_mask], 1)[0])
        tail = {
            "event_count": len(events),
            "relative_side_db_100ms": float(average[10]),
            "relative_side_db_300ms": float(average[30]),
            "decay_slope_db_per_second": slope,
            "roughness_db_per_step": float(np.std(np.diff(average[:40]))),
            "_curve": average,
        }
    else:
        tail = {"event_count": 0, "relative_side_db_100ms": None,
                "relative_side_db_300ms": None,
                "decay_slope_db_per_second": None,
                "roughness_db_per_step": None, "_curve": None}
    core = stages.get("core")
    tails = stages.get("tails")
    stages["tail_lift_over_core_db"] = (
        float(tails - core) if tails is not None and core is not None else None
    )
    return stages, tail


def envelope_relation(audio, sr):
    """Measure whether Side follows Mid immediately or as a delayed/smeared field."""
    mid, side = ms_waveforms(audio)
    mid = bandpass(mid, sr, 200, 10000)
    side = bandpass(side, sr, 200, 10000)
    mr = frame_rms(mid, sr, win_ms=30, hop_ms=10)
    srms = frame_rms(side, sr, win_ms=30, hop_ms=10)
    count = min(len(mr), len(srms))
    # Log envelopes reduce domination by only the loudest syllables.
    m = db(mr[:count] + EPS)
    s = db(srms[:count] + EPS)
    active = m > np.percentile(m, 25)
    m = np.where(active, m, np.nan)
    s = np.where(active, s, np.nan)
    correlations = []
    max_lag_frames = 50  # 0 to 500 ms
    for lag in range(max_lag_frames + 1):
        left = m[:count - lag] if lag else m
        right = s[lag:] if lag else s
        valid = np.isfinite(left) & np.isfinite(right)
        if np.sum(valid) < 30 or np.std(left[valid]) < EPS or np.std(right[valid]) < EPS:
            correlations.append(np.nan)
        else:
            correlations.append(float(np.corrcoef(left[valid], right[valid])[0, 1]))
    values = np.asarray(correlations)
    if np.any(np.isfinite(values)):
        peak_index = int(np.nanargmax(values))
        peak = float(values[peak_index])
        zero = float(values[0])
        # A broad peak means the Side envelope is temporally smeared.
        threshold = peak - 0.10
        broad = np.where(values >= threshold)[0]
        width_ms = float((broad[-1] - broad[0]) * 10) if len(broad) else 0.0
    else:
        peak_index, peak, zero, width_ms = 0, None, None, None
    return {
        "zero_lag_envelope_correlation": zero,
        "peak_envelope_correlation": peak,
        "peak_side_delay_ms": float(peak_index * 10),
        "near_peak_width_ms": width_ms,
        "_correlations": values,
    }


def gcc_phat_track(audio, sr, max_lag_ms=40):
    filtered = bandpass(audio, sr, 250, 9000)
    left, right = filtered[:, 0], filtered[:, 1]
    win, hop = int(0.5 * sr), int(0.1 * sr)
    starts = np.arange(0, len(left) - win + 1, hop)
    energies = np.array([
        np.sqrt(np.mean(((left[s:s + win] + right[s:s + win]) / 2) ** 2) + EPS)
        for s in starts
    ])
    keep_threshold = np.percentile(energies, 45) if len(energies) else 0
    times, lags_ms, sharpness, zero_ratio = [], [], [], []
    for start, energy in zip(starts, energies):
        if energy < keep_threshold:
            continue
        x = left[start:start + win] * signal.windows.hann(win)
        y = right[start:start + win] * signal.windows.hann(win)
        nfft = 1 << int(np.ceil(np.log2(2 * win - 1)))
        cross = np.fft.rfft(x, nfft) * np.conj(np.fft.rfft(y, nfft))
        cross /= np.maximum(np.abs(cross), EPS)
        corr = np.fft.fftshift(np.fft.irfft(cross, nfft))
        lag_samples = np.arange(-nfft // 2, nfft // 2)
        valid = np.abs(lag_samples) <= int(max_lag_ms * sr / 1000)
        off_zero = valid & (np.abs(lag_samples) >= int(0.7 * sr / 1000))
        values = np.abs(corr)
        if not np.any(off_zero):
            continue
        candidates = np.where(off_zero)[0]
        peak_index = candidates[np.argmax(values[candidates])]
        zero_index = nfft // 2
        baseline = np.median(values[candidates]) + EPS
        times.append((start + win / 2) / sr)
        lags_ms.append(lag_samples[peak_index] * 1000 / sr)
        sharpness.append(values[peak_index] / baseline)
        zero_ratio.append(values[zero_index] / (values[peak_index] + EPS))

    lags_ms = np.asarray(lags_ms)
    times = np.asarray(times)
    if len(lags_ms):
        bins = np.arange(-max_lag_ms, max_lag_ms + 1.0, 1.0)
        histogram, _ = np.histogram(lags_ms, bins=bins)
        concentration = float(histogram.max() / max(histogram.sum(), 1))
        probability = histogram[histogram > 0] / max(histogram.sum(), 1)
        entropy = float(-np.sum(probability * np.log(probability)) /
                        np.log(max(len(histogram), 2)))
        absolute = np.abs(lags_ms)
        lag_median = float(np.median(absolute))
        lag_mad = float(np.median(np.abs(absolute - lag_median)))
        if len(absolute) >= 8 and np.std(absolute) > 0.05:
            fs_track = 1.0 / np.median(np.diff(times))
            fmod, pmod = signal.periodogram(signal.detrend(absolute), fs_track)
            mask = (fmod >= 0.05) & (fmod <= 3.0)
            periodicity = float(np.max(pmod[mask]) / (np.sum(pmod[mask]) + EPS)) if np.any(mask) else 0.0
            modulation_hz = float(fmod[mask][np.argmax(pmod[mask])]) if np.any(mask) else None
        else:
            periodicity, modulation_hz = 0.0, None
    else:
        concentration = entropy = lag_median = lag_mad = None
        periodicity, modulation_hz = None, None
        histogram, bins = np.array([]), np.array([])

    return {
        "frames": int(len(lags_ms)),
        "absolute_lag_median_ms": lag_median,
        "absolute_lag_mad_ms": lag_mad,
        "dominant_lag_bin_concentration": concentration,
        "lag_histogram_entropy": entropy,
        "peak_sharpness_median": float(np.median(sharpness)) if sharpness else None,
        "zero_to_offzero_peak_ratio_median": float(np.median(zero_ratio)) if zero_ratio else None,
        "lag_modulation_periodicity": periodicity,
        "dominant_modulation_hz": modulation_hz,
        "_lags_ms": lags_ms,
        "_times": times,
        "_hist": histogram,
        "_bins": bins,
    }


def coherence_and_phase(audio, sr):
    filtered = bandpass(audio, sr, 80, min(18000, sr / 2 - 20))
    left, right = filtered[:, 0], filtered[:, 1]
    nperseg = 4096
    freq, coherence = signal.coherence(left, right, fs=sr, nperseg=nperseg,
                                       noverlap=nperseg // 2)
    _, cross = signal.csd(left, right, fs=sr, nperseg=nperseg,
                          noverlap=nperseg // 2)
    by_band = {}
    for lo, hi in BANDS:
        mask = (freq >= lo) & (freq < min(hi, sr / 2))
        if np.any(mask):
            by_band[f"{lo}-{hi}"] = float(np.median(coherence[mask]))

    fit = (freq >= 300) & (freq <= 8000) & (coherence >= 0.2)
    if np.sum(fit) >= 20:
        x = freq[fit]
        y = np.unwrap(np.angle(cross[fit]))
        weights = np.sqrt(coherence[fit])
        design = np.column_stack([x, np.ones_like(x)])
        weighted = design * weights[:, None]
        coefficients, *_ = np.linalg.lstsq(weighted, y * weights, rcond=None)
        predicted = design @ coefficients
        mean = np.average(y, weights=weights)
        ss_res = np.sum(weights * (y - predicted) ** 2)
        ss_tot = np.sum(weights * (y - mean) ** 2) + EPS
        r2 = float(1 - ss_res / ss_tot)
        delay_ms = float(abs(coefficients[0]) / (2 * np.pi) * 1000)
    else:
        r2, delay_ms = None, None
    return {
        "coherence_by_band": by_band,
        "phase_linear_fit_r2": r2,
        "phase_slope_delay_ms": delay_ms,
        "_frequency": freq,
        "_coherence": coherence,
    }


def spatial_distribution(audio, sr):
    left, right = audio[:, 0], audio[:, 1]
    freq, _, zl = signal.stft(left, fs=sr, window="hann", nperseg=2048,
                              noverlap=1536, boundary=None)
    _, _, zr = signal.stft(right, fs=sr, window="hann", nperseg=2048,
                           noverlap=1536, boundary=None)
    energy = np.abs(zl) ** 2 + np.abs(zr) ** 2
    band = (freq >= 200) & (freq <= min(12000, sr / 2 - 20))
    band_energy = energy[band]
    threshold = max(np.percentile(band_energy, 55), np.max(band_energy) * 1e-6)
    mask = band[:, None] & (energy >= threshold)
    ild = np.clip(db((np.abs(zl) + EPS) / (np.abs(zr) + EPS)), -30, 30)[mask]
    ipd = np.angle(zl * np.conj(zr))[mask]
    weights = np.sqrt(energy[mask])
    ild_bins = np.linspace(-30, 30, 61)
    ipd_bins = np.linspace(-np.pi, np.pi, 61)
    histogram, _, _ = np.histogram2d(ild, ipd, bins=[ild_bins, ipd_bins],
                                      weights=weights)
    probability = histogram.ravel() / (np.sum(histogram) + EPS)
    nonzero = probability[probability > 0]
    entropy = float(-np.sum(nonzero * np.log(nonzero)) / np.log(histogram.size))
    top_cells = float(np.sum(np.sort(probability)[-10:]))

    ild_hist, _ = np.histogram(ild, bins=ild_bins, weights=weights)
    peaks, properties = signal.find_peaks(
        ild_hist, prominence=max(np.max(ild_hist) * 0.06, EPS), distance=4
    )
    centers = (ild_bins[:-1] + ild_bins[1:]) / 2
    peak_locations = [float(centers[index]) for index in peaks]
    total_weight = np.sum(weights) + EPS
    central = float(np.sum(weights[(np.abs(ild) < 2) & (np.abs(ipd) < 0.25)]) / total_weight)
    phase_spread = float(np.sum(weights[np.abs(ipd) > 0.7]) / total_weight)
    return {
        "normalized_spatial_entropy": entropy,
        "top_10_cell_concentration": top_cells,
        "ild_peak_count": int(len(peaks)),
        "ild_peak_locations_db": peak_locations,
        "central_tf_fraction": central,
        "wide_phase_tf_fraction": phase_spread,
        "_histogram": histogram,
        "_ild_bins": ild_bins,
        "_ipd_bins": ipd_bins,
    }


def heuristic_scores(metrics):
    gcc = metrics["gcc_phat"]
    coh = metrics["coherence_phase"]
    spatial = metrics["spatial_distribution"]
    stages = metrics["width_by_stage"]
    mid_coh_values = [value for key, value in coh["coherence_by_band"].items()
                      if key in {"300-600", "600-1200", "1200-2500", "2500-5000"}]
    mid_coh = float(np.mean(mid_coh_values)) if mid_coh_values else 0.5
    fixed = gcc["dominant_lag_bin_concentration"] or 0
    phase_r2 = max(0.0, coh["phase_linear_fit_r2"] or 0)
    entropy = spatial["normalized_spatial_entropy"]
    phase_spread = spatial["wide_phase_tf_fraction"]
    periodicity = gcc["lag_modulation_periodicity"] or 0

    # Compact/coherent and diffuse are deliberately separate from fixed delay.
    # A multi-voice detuned Doubler can have dispersed lag peaks but still sound
    # like a few stable, compact sources.
    compact = float(np.clip(0.50 * mid_coh + 0.35 * (1 - phase_spread) +
                            0.15 * (1 - entropy), 0, 1))
    diffuse = float(np.clip(0.45 * (1 - mid_coh) + 0.40 * phase_spread +
                            0.15 * entropy, 0, 1))
    fixed_delay = float(np.clip(0.70 * fixed + 0.30 * phase_r2, 0, 1))
    modulation = float(np.clip(periodicity, 0, 1))
    if diffuse > compact + 0.10:
        character = "更像相位铺开的低相干区域型声场"
    elif compact > diffuse + 0.10:
        character = "更像中央紧密、相干度较高的点状/少量副本声场"
    else:
        character = "混合型或证据不足"
    return {
        "compact_coherent_evidence_0_to_1": compact,
        "diffuse_field_evidence_0_to_1": diffuse,
        "fixed_delay_evidence_0_to_1": fixed_delay,
        "periodic_modulation_evidence_0_to_1": modulation,
        "summary": character,
        "warning": "启发式分数只用于A/B比较，不能识别具体插件。",
    }


def analyze(label, path, target_sr, segment_seconds):
    audio, sr = load_stereo(path, target_sr)
    mono = np.mean(audio, axis=1)
    start, end = strongest_window(mono, sr, segment_seconds)
    segment = audio[start:end]
    mid, side = ms_waveforms(audio)
    stages, tail = stage_width_and_tails(audio, sr)
    result = {
        "label": label,
        "path": str(path),
        "sample_rate": sr,
        "duration_seconds": len(audio) / sr,
        "selected_segment": {"start_seconds": start / sr,
                             "duration_seconds": (end - start) / sr},
        "full_file_side_to_mid_db": float(db(np.sqrt(np.mean(side ** 2) + EPS) /
                                                (np.sqrt(np.mean(mid ** 2) + EPS) + EPS))),
        "left_right_correlation": float(np.corrcoef(audio[:, 0], audio[:, 1])[0, 1]),
        "width_by_stage": stages,
        "tail_response": tail,
        "mid_side_envelope_relation": envelope_relation(audio, sr),
        "gcc_phat": gcc_phat_track(segment, sr),
        "coherence_phase": coherence_and_phase(segment, sr),
        "spatial_distribution": spatial_distribution(segment, sr),
    }
    result["heuristic"] = heuristic_scores(result)
    return result


def public_copy(value):
    if isinstance(value, dict):
        return {key: public_copy(item) for key, item in value.items()
                if not key.startswith("_")}
    if isinstance(value, list):
        return [public_copy(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def plot_results(results, out_dir):
    colors = ["#2d8cff", "#ff7133", "#55d68b", "#c58cff"]
    labels = list(results)

    fig, axes = plt.subplots(1, len(labels), figsize=(7 * len(labels), 5.6),
                             facecolor="#0c1117", constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, label in zip(axes, labels):
        spatial = results[label]["spatial_distribution"]
        image = ax.imshow(np.log10(spatial["_histogram"].T + 1), origin="lower",
                          extent=[-30, 30, -180, 180], aspect="auto", cmap="magma")
        ax.set_title(confidence_title(f"{label}｜时频声像分布", "中", "分离伪影也会提高相位铺展"), color="white", fontproperties=fp(12, "bold"))
        ax.set_xlabel("L/R 电平差 ILD（dB）", color="white", fontproperties=fp())
        ax.set_ylabel("左右相位差 IPD（度）", color="white", fontproperties=fp())
        ax.tick_params(colors="white")
        ax.set_facecolor("#0c1117")
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.savefig(out_dir / "01_spatial_distribution.png", dpi=170,
                facecolor=fig.get_facecolor())
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), facecolor="#0c1117",
                             constrained_layout=True)
    for ax in axes:
        ax.set_facecolor("#111820")
        ax.grid(alpha=0.2)
        ax.tick_params(colors="white")
    for index, label in enumerate(labels):
        coh = results[label]["coherence_phase"]
        axes[0].semilogx(coh["_frequency"], coh["_coherence"],
                         color=colors[index], label=label, lw=2)
        gcc = results[label]["gcc_phat"]
        if len(gcc["_hist"]):
            centers = (gcc["_bins"][:-1] + gcc["_bins"][1:]) / 2
            density = gcc["_hist"] / max(np.sum(gcc["_hist"]), 1)
            axes[1].plot(centers, density, color=colors[index], label=label, lw=2)
    axes[0].set_xlim(80, 18000)
    axes[0].set_ylim(0, 1.02)
    axes[0].set_title(confidence_title("分频段左右相干度", "中高", "左右关系可测，处理来源不唯一"), color="white", fontproperties=fp(14, "bold"))
    axes[0].set_xlabel("频率（Hz）", color="white", fontproperties=fp())
    axes[0].set_ylabel("Magnitude-squared coherence", color="white", fontproperties=fp())
    axes[1].set_title(confidence_title("非零左右延迟峰的出现位置", "中低", "中央干声、移调与分离会削弱或伪造峰"), color="white", fontproperties=fp(14, "bold"))
    axes[1].set_xlabel("延迟（ms）", color="white", fontproperties=fp())
    axes[1].set_ylabel("出现比例", color="white", fontproperties=fp())
    for ax in axes:
        ax.legend(prop=fp(), facecolor="#111820", labelcolor="white")
    fig.savefig(out_dir / "02_coherence_and_delay.png", dpi=170,
                facecolor=fig.get_facecolor())
    plt.close(fig)

    stage_keys = ["core", "body", "tails", "very_quiet"]
    stage_labels = ["核心", "主体", "句尾", "极弱"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor="#0c1117",
                             constrained_layout=True)
    for ax in axes:
        ax.set_facecolor("#111820")
        ax.grid(alpha=0.2)
        ax.tick_params(colors="white")
    x = np.arange(len(stage_keys))
    width = 0.8 / len(labels)
    for index, label in enumerate(labels):
        values = [results[label]["width_by_stage"].get(key) for key in stage_keys]
        axes[0].bar(x + (index - (len(labels)-1)/2) * width, values,
                    width, color=colors[index], label=label)
        curve = results[label]["tail_response"]["_curve"]
        if curve is not None:
            axes[1].plot(np.arange(len(curve)) * 0.01, curve,
                         color=colors[index], label=label, lw=2)
    axes[0].axhline(0, color="white", lw=1)
    axes[0].set_xticks(x, stage_labels, fontproperties=fp())
    axes[0].set_title(confidence_title("宽度随人声音量变化", "中", "弱段 Side 可能含分离残留"), color="white", fontproperties=fp(13, "bold"))
    axes[0].set_ylabel("Side/Mid（dB）", color="white", fontproperties=fp())
    axes[1].axhline(0, color="white", lw=1)
    tail_count = sum(results[label]["tail_response"].get("event_count", 0) or 0 for label in labels)
    tail_level = "中低" if tail_count >= 6 else "证据不足"
    tail_note = "短窗尾部可测，下一句可能污染" if tail_count >= 6 else "有效停顿事件不足"
    axes[1].set_title(confidence_title("停顿后 Side 尾部（相对起点）", tail_level, tail_note), color="white", fontproperties=fp(13, "bold"))
    axes[1].set_xlabel("停顿后时间（秒）", color="white", fontproperties=fp())
    axes[1].set_ylabel("Side 电平变化（dB）", color="white", fontproperties=fp())
    axes[1].set_xlim(0, 0.35)
    for ax in axes:
        ax.legend(prop=fp(), facecolor="#111820", labelcolor="white")
    fig.savefig(out_dir / "03_stage_width_and_tails.png", dpi=170,
                facecolor=fig.get_facecolor())
    plt.close(fig)


def write_report(results, out_dir, separated_labels):
    lines = [
        "# 人声声场实验诊断", "",
        "本报告使用左右互相关、分频段相干度、时频声像分布和停顿尾部进行概率性判断；不把结果解释为具体插件识别。", "",
    ]
    for label, item in results.items():
        gcc = item["gcc_phat"]
        spatial = item["spatial_distribution"]
        coh_values = item["coherence_phase"]["coherence_by_band"]
        mid_keys = ["300-600", "600-1200", "1200-2500", "2500-5000"]
        mid_coh = np.mean([coh_values[k] for k in mid_keys if k in coh_values])
        lines += [
            f"## {label}", "",
            f"- 启发式结论：**{item['heuristic']['summary']}**。",
            f"- 紧密/相干证据：{item['heuristic']['compact_coherent_evidence_0_to_1']:.3f}；扩散证据：{item['heuristic']['diffuse_field_evidence_0_to_1']:.3f}；固定延迟证据：{item['heuristic']['fixed_delay_evidence_0_to_1']:.3f}。",
            f"- 全文件 Side/Mid：{item['full_file_side_to_mid_db']:.2f} dB；L/R 相关度：{item['left_right_correlation']:.3f}。",
            f"- 中频平均相干度：{mid_coh:.3f}；时频空间熵：{spatial['normalized_spatial_entropy']:.3f}。",
            f"- 非零延迟峰：中位绝对延迟 {gcc['absolute_lag_median_ms']:.2f} ms，最常见延迟格集中度 {gcc['dominant_lag_bin_concentration']:.3f}。" if gcc['absolute_lag_median_ms'] is not None else "- 未取得可靠非零延迟峰。",
            f"- 音量降低时的宽度提升：{item['width_by_stage']['tail_lift_over_core_db']:.2f} dB。" if item['width_by_stage']['tail_lift_over_core_db'] is not None else "- 无法计算句尾宽度提升。",
            f"- Side 包络与 Mid 最相似时滞后约 {item['mid_side_envelope_relation']['peak_side_delay_ms']:.0f} ms，相关峰宽约 {item['mid_side_envelope_relation']['near_peak_width_ms']:.0f} ms。",
        ]
        if label in separated_labels:
            lines.append("- 该文件被标记为人声分离参考；低相干、高空间熵和不规则延迟也可能来自分离伪影。")
        lines.append("")
    lines += [
        "## 图表", "",
        "- `01_spatial_distribution.png`：每个输入面板可信度中；少数紧密亮团更像离散声像，连续铺开也可能来自分离伪影。",
        "- `02_coherence_and_delay.png`：相干度中高、延迟峰中低；延迟峰会被中央干声、移调和分离削弱或伪造。",
        "- `03_stage_width_and_tails.png`：阶段宽度中；Side 尾部按有效事件数标为中低或证据不足。",
        "",
        "## 边界", "",
        "真实双轨、调制 Doubler、宽混响和人声分离可能产生重叠特征。若没有原始单声道干声、湿声单独导出和完整混音，结果只能作为证据权重，不能作为插件指纹。",
    ]
    (out_dir / "experiment_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_input(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("输入格式必须是 标签=文件路径")
    label, path = value.split("=", 1)
    return label.strip(), Path(path.strip())


def main():
    parser = argparse.ArgumentParser(description="实验性人声立体声声场诊断")
    parser.add_argument("--input", action="append", type=parse_input, required=True,
                        help="可重复传入：标签=音频路径")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--segment-seconds", type=float, default=20.0)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--separated-label", action="append", default=[])
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for label, path in args.input:
        if not path.exists():
            raise FileNotFoundError(path)
        results[label] = analyze(label, path, args.sample_rate,
                                 args.segment_seconds)
    plot_results(results, args.out_dir)
    write_report(results, args.out_dir, set(args.separated_label))
    public = {label: public_copy(item) for label, item in results.items()}
    (args.out_dir / "spatial_metrics.json").write_text(
        json.dumps(public, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "plugin_id": PLUGIN_ID,
        "analysis_mode": ANALYSIS_MODE,
        "metrics": str(args.out_dir / "spatial_metrics.json"),
        "report": str(args.out_dir / "experiment_report.md"),
        "charts": ["01_spatial_distribution.png",
                   "02_coherence_and_delay.png",
                   "03_stage_width_and_tails.png"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
