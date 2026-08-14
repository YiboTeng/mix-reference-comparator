#!/usr/bin/env python3
"""Vocal diagnostics for timbre, articulation, dynamics and texture.

The program compares non-aligned vocal files.  It measures observable evidence;
it does not claim to identify a specific EQ, compressor, tuner or saturator.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from matplotlib.font_manager import FontProperties
from scipy import ndimage, signal


EPS = 1e-12
BANDS = [
    (80, 150, "80–150"), (150, 300, "150–300"),
    (300, 600, "300–600"), (600, 1200, "600–1.2k"),
    (1200, 2500, "1.2–2.5k"), (2500, 5000, "2.5–5k"),
    (5000, 8000, "5–8k"), (8000, 12000, "8–12k"),
    (12000, 18000, "12–18k"),
]
PALETTE = ["#386cb0", "#ef8a17", "#4daf4a", "#984ea3"]


def db_amp(x):
    return 20.0 * np.log10(np.maximum(np.asarray(x), EPS))


def db_power(x):
    return 10.0 * np.log10(np.maximum(np.asarray(x), EPS))


def finite_obj(value):
    if isinstance(value, dict):
        return {k: finite_obj(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [finite_obj(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else round(float(value), 6)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def get_font():
    candidates = [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"]
    for path in candidates:
        if Path(path).exists():
            return FontProperties(fname=path)
    return FontProperties()


FONT = get_font()


def fp(size=10, weight="normal"):
    return FontProperties(fname=FONT.get_file(), size=size, weight=weight) if FONT.get_file() else FONT.copy()


def confidence_title(title, level, limitation):
    """给每个子图单独标注当前 A/B 解释的可信度和主要限制。"""
    return f"{title}\n可信度：{level}｜{limitation}"


def load_mono(path: Path, target_sr: int):
    audio, sr = sf.read(path, always_2d=True, dtype="float32")
    if audio.size == 0:
        raise ValueError(f"空音频：{path}")
    mono = np.mean(audio[:, :2], axis=1).astype(np.float64)
    if sr != target_sr:
        common = math.gcd(sr, target_sr)
        mono = signal.resample_poly(mono, target_sr // common, sr // common)
        sr = target_sr
    return mono, sr, float(len(mono) / sr)


def strongest_window(y, sr, seconds=35.0):
    length = min(len(y), int(seconds * sr))
    if length >= len(y):
        return y.copy(), 0.0
    hop = max(1, sr // 5)
    sq = y * y
    cumulative = np.concatenate([[0.0], np.cumsum(sq)])
    starts = np.arange(0, len(y) - length + 1, hop)
    energy = cumulative[starts + length] - cumulative[starts]
    start = int(starts[int(np.argmax(energy))])
    return y[start:start + length].copy(), start / sr


def frame_rms(y, frame=2048, hop=512):
    return librosa.feature.rms(y=y, frame_length=frame, hop_length=hop, center=True)[0]


def active_mask(rms):
    level = db_amp(rms)
    ceiling = float(np.percentile(level, 95))
    return level > ceiling - 32.0


def band_matrix(power, freqs):
    rows = []
    for lo, hi, _ in BANDS:
        mask = (freqs >= lo) & (freqs < hi)
        rows.append(np.sum(power[mask], axis=0) + EPS)
    return np.asarray(rows)


def robust_iqr(values):
    values = np.asarray(values)
    return float(np.percentile(values, 75) - np.percentile(values, 25))


def spectral_features(y, sr):
    n_fft, hop = 4096, 512
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop, window="hann")
    mag = np.abs(stft)
    power = mag * mag
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    rms = librosa.feature.rms(S=mag, frame_length=n_fft, hop_length=hop)[0]
    active = active_mask(rms)
    bands = band_matrix(power, freqs)
    total = np.sum(power[(freqs >= 80) & (freqs < 18000)], axis=0) + EPS
    rel_bands = db_power(bands / total[None, :])
    centroid = librosa.feature.spectral_centroid(S=mag, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(S=mag)[0]
    flux = np.concatenate([[0.0], np.sqrt(np.mean(np.maximum(np.diff(mag, axis=1), 0.0) ** 2, axis=0))])
    flux /= np.median(flux[active]) + EPS
    levels = db_amp(rms)
    active_levels = levels[active]
    q33, q67 = np.percentile(active_levels, [33, 67])
    stages = {
        "弱声": active & (levels <= q33),
        "普通": active & (levels > q33) & (levels < q67),
        "强声": active & (levels >= q67),
    }
    stage_timbre = {name: np.median(rel_bands[:, mask], axis=1) for name, mask in stages.items()}
    band_percentiles = {
        BANDS[i][2]: np.percentile(rel_bands[i, active], [10, 25, 50, 75, 90])
        for i in range(len(BANDS))
    }

    # Persistent resonances: subtract a broad log-frequency envelope from the
    # active median spectrum. This identifies stable peaks, not causal EQ moves.
    valid = (freqs >= 100) & (freqs <= 10000)
    log_freq = np.geomspace(100, 10000, 480)
    median_spec = db_power(np.median(power[:, active], axis=1))
    interp = np.interp(np.log(log_freq), np.log(freqs[valid]), median_spec[valid])
    smooth = ndimage.gaussian_filter1d(interp, sigma=18)
    residual = interp - smooth
    peaks, props = signal.find_peaks(residual, prominence=0.7, distance=18)
    ranking = sorted(peaks, key=lambda idx: residual[idx], reverse=True)[:6]
    resonances = []
    for i in ranking:
        hz = float(log_freq[i])
        narrow = (freqs >= hz * 0.98) & (freqs <= hz * 1.02)
        broad = (freqs >= hz / (2 ** (1 / 6))) & (freqs <= hz * (2 ** (1 / 6)))
        broad &= ~narrow
        narrow_mean = np.mean(power[narrow], axis=0) + EPS
        broad_mean = np.mean(power[broad], axis=0) + EPS
        frame_excess = db_power(narrow_mean / broad_mean)
        occurrence = float(np.mean(frame_excess[active] > 3.0))
        resonances.append({"hz": hz, "excess_db": float(residual[i]),
                           "occurrence_fraction": occurrence})

    # Sibilance candidates: active frames with exceptional 5–12 kHz fraction
    # and above-median spectral flux. This estimates event prevalence only.
    sib_energy = bands[6] + bands[7]
    sib_ratio = db_power(sib_energy / total)
    sib_thr = np.percentile(sib_ratio[active], 90)
    sib_mask = active & (sib_ratio >= sib_thr) & (flux >= np.median(flux[active]))
    air_ratio = db_power(bands[8] / total)
    presence_ratio = db_power((bands[5] + bands[6]) / total)
    low_mid_masking = db_power((bands[1] + bands[2]) /
                               (bands[3] + bands[4] + bands[5] + EPS))
    # Low-flux adjacent-frame spectral movement is a vowel-stability proxy.
    stable = active & (flux <= np.median(flux[active]))
    stable_pairs = stable[1:] & stable[:-1]
    frame_change = np.median(np.abs(np.diff(rel_bands, axis=1)), axis=0)

    return {
        "n_fft": n_fft, "hop": hop, "mag": mag, "power": power,
        "freqs": freqs, "rms": rms, "levels": levels, "active": active,
        "bands": bands, "rel_bands": rel_bands, "stages": stages,
        "stage_timbre": stage_timbre, "band_percentiles": band_percentiles,
        "centroid": centroid, "flatness": flatness, "flux": flux,
        "resonances": resonances, "sib_mask": sib_mask,
        "sib_ratio": sib_ratio, "air_ratio": air_ratio,
        "presence_ratio": presence_ratio,
        "low_mid_masking": low_mid_masking,
        "spectral_flux_median": float(np.median(flux[active])),
        "spectral_flux_iqr": robust_iqr(flux[active]),
        "vowel_spectral_change_db": float(np.median(frame_change[stable_pairs])) if np.any(stable_pairs) else np.nan,
    }


def onset_and_syllable_features(y, sr, spec):
    hop = spec["hop"]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=hop,
        backtrack=False, units="frames", delta=0.12, wait=max(1, int(0.09 * sr / hop)))
    max_frame = min(len(spec["rms"]), len(onset_env))
    onset_frames = onset_frames[(onset_frames > 3) & (onset_frames < max_frame - int(0.55 * sr / hop))]
    onset_frames = onset_frames[spec["active"][np.minimum(onset_frames, len(spec["active"]) - 1)]]

    # Align 50 ms before to 500 ms after each detected syllabic onset.
    pre = int(round(0.05 * sr / hop))
    post = int(round(0.50 * sr / hop))
    curves, bright_curves = [], []
    for idx in onset_frames:
        if idx - pre >= 0 and idx + post < len(spec["rms"]):
            curve = db_amp(spec["rms"][idx - pre:idx + post + 1])
            curve -= np.max(curve[:max(pre + 8, 9)])
            curves.append(curve)
            bright_curves.append(spec["presence_ratio"][idx - pre:idx + post + 1])
    if curves:
        mean_curve = np.median(np.asarray(curves), axis=0)
        mean_bright = np.median(np.asarray(bright_curves), axis=0)
    else:
        mean_curve = np.zeros(pre + post + 1)
        mean_bright = np.zeros(pre + post + 1)
    time_ms = (np.arange(-pre, post + 1) * hop / sr) * 1000.0

    # Consonant/vowel proxy: high-band energy in first 70 ms versus voiced body
    # energy 90–260 ms later. It is content-sensitive, so reliability is medium.
    consonant_ratios, body_drops, body_levels = [], [], []
    early = max(1, int(0.07 * sr / hop))
    b0, b1 = int(0.09 * sr / hop), int(0.26 * sr / hop)
    for idx in onset_frames:
        if idx + b1 >= spec["bands"].shape[1]:
            continue
        high_early = np.mean(spec["bands"][5:8, idx:idx + early])
        mid_body = np.mean(spec["bands"][2:6, idx + b0:idx + b1])
        consonant_ratios.append(float(db_power(high_early / (mid_body + EPS))))
        peak = np.max(spec["rms"][idx:idx + early])
        body = np.mean(spec["rms"][idx + b0:idx + b1])
        body_drops.append(float(db_amp(peak / (body + EPS))))
        body_levels.append(float(db_amp(body)))

    # Phrase-end broadband decay. Keep only transitions followed by at least
    # 450 ms with almost no reactivation, and include 120 ms before the edge so
    # the curve starts in audible material instead of at the noise floor.
    active = spec["active"]
    transitions = np.where(active[:-1] & ~active[1:])[0] + 1
    pre_tail = int(0.12 * sr / hop)
    post_tail = int(0.55 * sr / hop)
    tails = []
    for idx in transitions:
        future = active[idx:idx + post_tail]
        if idx - pre_tail >= 0 and idx + post_tail < len(spec["rms"]) and np.mean(future) <= 0.08:
            t = db_amp(spec["rms"][idx - pre_tail:idx + post_tail])
            anchor = np.max(t[:pre_tail + 1])
            tails.append(t - anchor)
    tail_curve = np.median(np.asarray(tails), axis=0) if tails else np.zeros(pre_tail + post_tail)
    tail_time_ms = (np.arange(-pre_tail, post_tail) * hop / sr) * 1000.0
    return {
        "onset_count": int(len(onset_frames)), "time_ms": time_ms,
        "syllable_curve_db": mean_curve, "brightness_curve_db": mean_bright,
        "consonant_vowel_proxy_db": float(np.median(consonant_ratios)) if consonant_ratios else np.nan,
        "transient_body_db": float(np.median(body_drops)) if body_drops else np.nan,
        "syllable_body_level_iqr_db": robust_iqr(body_levels) if body_levels else np.nan,
        "tail_curve_db": tail_curve, "tail_time_ms": tail_time_ms,
        "tail_count": int(len(tails)),
    }


def dynamics_features(y, sr, spec, onset):
    # Macro dynamics on 1 s windows, micro dynamics on 20 ms windows.
    macro = frame_rms(y, frame=sr, hop=sr // 4)
    micro_frame = max(32, int(0.02 * sr))
    micro = frame_rms(y, frame=micro_frame, hop=max(16, micro_frame // 2))
    macro_db, micro_db = db_amp(macro), db_amp(micro)
    macro_active = macro_db > np.percentile(macro_db, 95) - 32
    micro_active = micro_db > np.percentile(micro_db, 95) - 32
    active_samples = y[np.abs(y) > np.percentile(np.abs(y), 20)]
    peak = float(np.max(np.abs(y)))
    near_peak = float(np.mean(np.abs(y) >= peak * (10 ** (-1.0 / 20))))
    near_full = float(np.mean(np.abs(y) >= 0.999))
    exact_plateau = float(np.mean(np.abs(np.diff(y)) < 1e-10))
    true_peak_4x = float(np.max(np.abs(signal.resample_poly(y, 4, 1))))

    band_dr = {}
    for i, (_, _, name) in enumerate(BANDS):
        vals = db_power(spec["bands"][i, spec["active"]])
        band_dr[name] = {
            "iqr_db": robust_iqr(vals),
            "p90_p10_db": float(np.percentile(vals, 90) - np.percentile(vals, 10)),
        }
    # Peak-density is measured in 20 ms active frames within 3 dB of the
    # active 95th percentile, avoiding dependence on one isolated sample.
    threshold = np.percentile(micro_db[micro_active], 95) - 3.0
    dense = float(np.mean(micro_db[micro_active] >= threshold))
    return {
        "macro_iqr_db": robust_iqr(macro_db[macro_active]),
        "macro_p90_p10_db": float(np.percentile(macro_db[macro_active], 90) - np.percentile(macro_db[macro_active], 10)),
        "micro_iqr_db": robust_iqr(micro_db[micro_active]),
        "micro_p90_p10_db": float(np.percentile(micro_db[micro_active], 90) - np.percentile(micro_db[micro_active], 10)),
        "micro_peak_density": dense,
        "sample_peak_dbfs": float(db_amp(peak)),
        "true_peak_4x_dbfs": float(db_amp(true_peak_4x)),
        "near_peak_sample_fraction": near_peak,
        "near_full_scale_fraction": near_full,
        "flat_step_fraction": exact_plateau,
        "band_dynamics": band_dr,
        "transient_body_db": onset["transient_body_db"],
        "active_sample_rms_dbfs": float(db_amp(np.sqrt(np.mean(active_samples ** 2) + EPS))),
    }


def pitch_and_texture_features(y, sr, spec):
    hop, n_fft = spec["hop"], spec["n_fft"]
    # pYIN provides a voiced probability, allowing unvoiced/sibilant frames to
    # be excluded from pitch and harmonic measurements.
    f0, voiced, voiced_prob = librosa.pyin(
        y, fmin=65, fmax=700, sr=sr, frame_length=n_fft, hop_length=hop,
        fill_na=np.nan)
    n = min(len(f0), spec["mag"].shape[1])
    f0, voiced_prob = f0[:n], voiced_prob[:n]
    valid = np.isfinite(f0) & (voiced_prob >= 0.55) & spec["active"][:n]
    mag, freqs = spec["mag"][:, :n], spec["freqs"]
    harmonic_ratios, slopes, high_harmonic = [], [], []
    for frame_idx in np.where(valid)[0][::2]:
        base = f0[frame_idx]
        hs, hs_freq = [], []
        for h in range(1, 13):
            target = base * h
            if target >= min(10000, sr / 2 - 100):
                break
            center = int(np.argmin(np.abs(freqs - target)))
            lo, hi = max(0, center - 1), min(len(freqs), center + 2)
            hs.append(float(np.max(mag[lo:hi, frame_idx]) + EPS))
            hs_freq.append(target)
        if len(hs) < 5:
            continue
        hs = np.asarray(hs)
        total_mag = np.sum(mag[(freqs >= base * 0.7) & (freqs <= min(10000, base * 12.5)), frame_idx]) + EPS
        harmonic_ratios.append(float(np.sum(hs) / total_mag))
        x = np.log2(np.asarray(hs_freq) / hs_freq[0])
        slopes.append(float(np.polyfit(x, db_amp(hs / hs[0]), 1)[0]))
        high_harmonic.append(float(np.sum(hs[4:]) / (np.sum(hs) + EPS)))

    # Analyse only contiguous voiced runs; joining gaps would turn melody jumps
    # into fake vibrato and pitch-step evidence.
    valid_idx = np.where(valid)[0]
    splits = np.where(np.diff(valid_idx) > 1)[0] + 1
    runs = np.split(valid_idx, splits) if len(valid_idx) else []
    local_parts, step_parts, vibrato_rates, vibrato_depths = [], [], [], []
    frame_rate = sr / hop
    for run in runs:
        if len(run) < 8:
            continue
        cents_run = 1200.0 * np.log2(f0[run])
        step_parts.append(np.abs(np.diff(cents_run)))
        size = min(15, len(cents_run) if len(cents_run) % 2 else len(cents_run) - 1)
        size = max(3, size)
        trend = ndimage.median_filter(cents_run, size=size, mode="nearest")
        local = cents_run - trend
        local_parts.append(local)
        if len(local) >= int(frame_rate * 0.7):
            f = np.fft.rfftfreq(len(local), d=1 / frame_rate)
            p = np.abs(np.fft.rfft(local * np.hanning(len(local)))) ** 2
            region = (f >= 3.0) & (f <= 9.0)
            if np.any(region):
                peak_i = np.where(region)[0][int(np.argmax(p[region]))]
                vibrato_rates.append(float(f[peak_i]))
                vibrato_depths.append(float(np.std(local) * math.sqrt(2)))
    local_pitch = np.concatenate(local_parts) if local_parts else np.array([])
    cents_step = np.concatenate(step_parts) if step_parts else np.array([])

    flat_active = spec["flatness"][spec["active"]]
    # Loudness-linked texture: compare high-harmonic spectral share in the
    # loudest and quietest active thirds. Performance and vowels also affect it.
    levels = spec["levels"]
    act_vals = levels[spec["active"]]
    q33, q67 = np.percentile(act_vals, [33, 67])
    high_share = db_power((spec["bands"][5] + spec["bands"][6] + spec["bands"][7]) /
                          (np.sum(spec["bands"], axis=0) + EPS))
    low_mask = spec["active"] & (levels <= q33)
    high_mask = spec["active"] & (levels >= q67)
    growth = float(np.median(high_share[high_mask]) - np.median(high_share[low_mask]))

    # Crude formant proxy from smoothed spectral envelope peaks in voiced frames.
    # Kept intentionally low-confidence because vowels and singers differ.
    formant_candidates = []
    for frame_idx in np.where(valid)[0][::12]:
        envelope = ndimage.gaussian_filter1d(db_amp(mag[:, frame_idx]), sigma=9)
        region = (freqs >= 250) & (freqs <= 4000)
        pk, prop = signal.find_peaks(envelope[region], prominence=1.0, distance=18)
        hz = freqs[region][pk]
        if len(hz) >= 2:
            formant_candidates.append(hz[:3])
    formants, formant_iqr = [], []
    for order in range(3):
        vals = [row[order] for row in formant_candidates if len(row) > order]
        formants.append(float(np.median(vals)) if vals else np.nan)
        formant_iqr.append(robust_iqr(vals) if vals else np.nan)

    return {
        "voiced_frame_fraction": float(np.mean(valid)),
        "harmonic_bin_concentration": float(np.median(harmonic_ratios)) if harmonic_ratios else np.nan,
        "harmonic_bin_concentration_iqr": robust_iqr(harmonic_ratios) if harmonic_ratios else np.nan,
        "harmonic_slope_db_per_oct": float(np.median(slopes)) if slopes else np.nan,
        "harmonic_slope_iqr_db_per_oct": robust_iqr(slopes) if slopes else np.nan,
        "upper_harmonic_share": float(np.median(high_harmonic)) if high_harmonic else np.nan,
        "spectral_flatness_median": float(np.median(flat_active)),
        "pitch_local_std_cents": float(np.std(local_pitch)) if len(local_pitch) else np.nan,
        "pitch_step_median_cents": float(np.median(cents_step)) if len(cents_step) else np.nan,
        "pitch_plateau_fraction": float(np.mean(cents_step < 5.0)) if len(cents_step) else np.nan,
        "vibrato_rate_proxy_hz": float(np.median(vibrato_rates)) if vibrato_rates else np.nan,
        "vibrato_depth_proxy_cents": float(np.median(vibrato_depths)) if vibrato_depths else np.nan,
        "high_texture_growth_loud_vs_soft_db": growth,
        "formant_proxy_hz": formants, "formant_proxy_iqr_hz": formant_iqr,
    }


def analyze(label, path, target_sr, segment_seconds):
    full, sr, duration = load_mono(path, target_sr)
    segment, start = strongest_window(full, sr, segment_seconds)
    spec = spectral_features(segment, sr)
    onset = onset_and_syllable_features(segment, sr, spec)
    dynamics = dynamics_features(full, sr, spec, onset)
    texture = pitch_and_texture_features(segment, sr, spec)
    summary = {
        "label": label, "path": str(path), "duration_seconds": duration,
        "analysis_segment_start_seconds": start,
        "analysis_segment_seconds": len(segment) / sr,
        "timbre": {
            "band_percentiles_db_relative": spec["band_percentiles"],
            "level_stage_timbre_db_relative": spec["stage_timbre"],
            "persistent_resonance_candidates": spec["resonances"],
            "presence_median_db_relative": float(np.median(spec["presence_ratio"][spec["active"]])),
            "air_median_db_relative": float(np.median(spec["air_ratio"][spec["active"]])),
            "low_mid_masking_db": float(np.median(spec["low_mid_masking"][spec["active"]])),
            "spectral_flux_median_normalized": spec["spectral_flux_median"],
            "spectral_flux_iqr_normalized": spec["spectral_flux_iqr"],
            "vowel_spectral_change_db": spec["vowel_spectral_change_db"],
            "sibilant_candidate_fraction_active": float(np.mean(spec["sib_mask"][spec["active"]])),
            "sibilant_candidate_level_db_relative": float(np.median(spec["sib_ratio"][spec["sib_mask"]])) if np.any(spec["sib_mask"]) else np.nan,
        },
        "articulation": onset,
        "dynamics": dynamics,
        "texture_pitch": texture,
    }
    return summary, spec


def plot_timbre(records, specs, out):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    x = np.arange(len(BANDS)); labels = [b[2] for b in BANDS]
    for i, (rec, spec) in enumerate(zip(records, specs)):
        color = PALETTE[i]
        med = np.array([rec["timbre"]["band_percentiles_db_relative"][name][2] for name in labels])
        p10 = np.array([rec["timbre"]["band_percentiles_db_relative"][name][0] for name in labels])
        p90 = np.array([rec["timbre"]["band_percentiles_db_relative"][name][4] for name in labels])
        axes[0, 0].plot(x, med, marker="o", color=color, label=rec["label"])
        axes[0, 0].fill_between(x, p10, p90, color=color, alpha=.13)
        strong = np.asarray(rec["timbre"]["level_stage_timbre_db_relative"]["强声"])
        weak = np.asarray(rec["timbre"]["level_stage_timbre_db_relative"]["弱声"])
        axes[0, 1].plot(x, strong - weak, marker="o", color=color, label=rec["label"])
        rfreq = [v["hz"] for v in rec["timbre"]["persistent_resonance_candidates"]]
        rex = [v["excess_db"] for v in rec["timbre"]["persistent_resonance_candidates"]]
        axes[1, 0].scatter(rfreq, rex, s=70, color=color, label=rec["label"], alpha=.85)
        axes[1, 1].bar(np.arange(3) + (i - .5) * .34,
                       [100 * 10 ** (rec["timbre"]["presence_median_db_relative"] / 10),
                        100 * 10 ** (rec["timbre"]["air_median_db_relative"] / 10),
                        100 * rec["timbre"]["sibilant_candidate_fraction_active"]],
                       width=.34, color=color, label=rec["label"])
    axes[0, 0].set_title(confidence_title("动态频段分布：中位数与10–90%范围", "中高", "可判常驻/偶发；仍受音素影响"), fontproperties=fp(11, "bold"))
    axes[0, 1].set_title(confidence_title("强声减弱声：音量相关音色", "中", "表演力度与元音是混杂项"), fontproperties=fp(11, "bold"))
    axes[1, 0].set_title(confidence_title("持续共振候选（相对宽谱包络）", "中低", "须扫听确认，不能直接作为 EQ 点"), fontproperties=fp(11, "bold"))
    axes[1, 1].set_title(confidence_title("Presence / Air 能量与齿音事件占比", "中高", "能量可靠；齿音事件受歌词影响"), fontproperties=fp(11, "bold"))
    for ax in axes.flat:
        ax.grid(alpha=.22); ax.legend(prop=fp(9))
    for ax in axes[0]:
        ax.set_xticks(x, labels, rotation=28, fontproperties=fp(8)); ax.axhline(0, color="#777", lw=.8)
    axes[1, 0].set_xscale("log"); axes[1, 0].set_xlabel("Hz", fontproperties=fp()); axes[1, 0].set_ylabel("突出量 dB", fontproperties=fp())
    axes[1, 1].set_xticks(range(3), ["Presence能量\n2.5–8k", "Air能量\n12–18k", "齿音候选\n活跃帧占比"], fontproperties=fp(9))
    axes[1, 1].set_ylabel("%", fontproperties=fp())
    fig.suptitle("细分图 D1｜音色、共振与高频成分", fontproperties=fp(17, "bold"))
    fig.tight_layout(); fig.savefig(out / "D1_timbre_detail.png", dpi=165); plt.close(fig)


def plot_articulation(records, out):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for i, rec in enumerate(records):
        a = rec["articulation"]; color = PALETTE[i]
        axes[0, 0].plot(a["time_ms"], a["syllable_curve_db"], color=color, label=rec["label"])
        axes[0, 1].plot(a["time_ms"], a["brightness_curve_db"], color=color, label=rec["label"])
        axes[1, 0].plot(a["tail_time_ms"], a["tail_curve_db"], color=color,
                        label=f"{rec['label']}（{a['tail_count']}个句尾）")
        axes[1, 1].bar(np.arange(2) + (i - .5) * .34,
                       [a["consonant_vowel_proxy_db"], a["transient_body_db"]],
                       width=.34, color=color, label=rec["label"])
    if sum(rec["articulation"]["tail_count"] for rec in records) == 0:
        axes[1, 0].text(.5, .5, "本片段没有检测到合格的长句尾\n该指标证据不足，不作比较",
                        transform=axes[1, 0].transAxes, ha="center", va="center",
                        fontproperties=fp(13, "bold"), color="#555")
    tail_count = sum(rec["articulation"]["tail_count"] for rec in records)
    tail_confidence = ("中低", "仅在足够长句尾中成立") if tail_count >= 6 else ("证据不足", "有效长句尾数量不足")
    titles = [
        confidence_title("对齐音节的平均电平包络", "中高", "起始检测可靠；歌词未对齐"),
        confidence_title("音节起始后的 Presence 变化", "中", "音素与歌手差异明显"),
        confidence_title("停顿后的全频衰减", *tail_confidence),
        confidence_title("辅音/元音与字头/主体代理", "中低", "高度依赖歌词音素"),
    ]
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title, fontproperties=fp(13, "bold")); ax.grid(alpha=.22); ax.legend(prop=fp(9)); ax.axhline(0, color="#777", lw=.8)
    axes[0, 0].set_xlabel("起始点前后 ms", fontproperties=fp()); axes[0, 0].set_ylabel("相对峰值 dB", fontproperties=fp())
    axes[0, 1].set_xlabel("起始点前后 ms", fontproperties=fp()); axes[0, 1].set_ylabel("2.5–8 kHz相对量 dB", fontproperties=fp())
    axes[1, 0].set_xlabel("检测到句尾前后 ms", fontproperties=fp()); axes[1, 0].set_ylabel("相对句尾前峰值 dB", fontproperties=fp())
    axes[1, 1].set_xticks(range(2), ["辅音高频/\n元音主体", "字头峰值/\n音节主体"], fontproperties=fp(9)); axes[1, 1].set_ylabel("dB", fontproperties=fp())
    fig.suptitle("细分图 D2｜咬字、音节包络与停顿", fontproperties=fp(17, "bold"))
    fig.tight_layout(); fig.savefig(out / "D2_articulation_time.png", dpi=165); plt.close(fig)


def plot_dynamics(records, out):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    width=.34; x=np.arange(2)
    for i, rec in enumerate(records):
        d=rec["dynamics"]; color=PALETTE[i]
        axes[0,0].bar(x+(i-.5)*width,[d["macro_p90_p10_db"],d["micro_p90_p10_db"]],width=width,color=color,label=rec["label"])
        axes[0,1].bar(np.arange(2)+(i-.5)*width,[d["micro_peak_density"]*100,d["near_peak_sample_fraction"]*100],width=width,color=color,label=rec["label"])
        band_names=[b[2] for b in BANDS]
        axes[1,0].plot(np.arange(len(BANDS)),[d["band_dynamics"][b]["p90_p10_db"] for b in band_names],marker="o",color=color,label=rec["label"])
        axes[1,1].bar(np.arange(3)+(i-.5)*width,[d["transient_body_db"],d["near_full_scale_fraction"]*1e5,d["flat_step_fraction"]*1e5],width=width,color=color,label=rec["label"])
    titles=[
        confidence_title("宏观与微观动态范围（P90–P10）", "高", "直接电平统计"),
        confidence_title("峰值密度", "高", "直接短窗与样本统计"),
        confidence_title("各频段动态范围", "中高", "可定位频段；低频受音高进入影响"),
        confidence_title("压缩/削波行为代理", "中低", "可描述行为，不能反推插件参数"),
    ]
    for ax,title in zip(axes.flat,titles): ax.set_title(title,fontproperties=fp(13,"bold")); ax.grid(alpha=.22); ax.legend(prop=fp(9))
    axes[0,0].set_xticks(x,["宏观 1秒","微观 20ms"],fontproperties=fp()); axes[0,0].set_ylabel("dB",fontproperties=fp())
    axes[0,1].set_xticks(range(2),["活跃短窗\n近峰占比 %","近最高样本\n占比 %"],fontproperties=fp(9))
    axes[1,0].set_xticks(range(len(BANDS)),[b[2] for b in BANDS],rotation=28,fontproperties=fp(8)); axes[1,0].set_ylabel("P90–P10 dB",fontproperties=fp())
    axes[1,1].set_xticks(range(3),["字头/主体\ndB","近满幅占比\n×10⁵","平坦步进占比\n×10⁵"],fontproperties=fp(9))
    fig.suptitle("细分图 D3｜宏/微动态、分频动态与峰值行为",fontproperties=fp(17,"bold"))
    fig.tight_layout(); fig.savefig(out/"D3_dynamics_detail.png",dpi=165); plt.close(fig)


def plot_texture(records, out):
    fig, axes=plt.subplots(2,2,figsize=(15,10)); width=.34
    keys1=["harmonic_bin_concentration","spectral_flatness_median","upper_harmonic_share"]
    for i,rec in enumerate(records):
        t=rec["texture_pitch"]; color=PALETTE[i]
        axes[0,0].bar(np.arange(3)+(i-.5)*width,[t[k] for k in keys1],width=width,color=color,label=rec["label"])
        axes[0,1].bar(np.arange(2)+(i-.5)*width,[t["harmonic_slope_db_per_oct"],t["high_texture_growth_loud_vs_soft_db"]],width=width,color=color,label=rec["label"])
        axes[1,0].bar(np.arange(3)+(i-.5)*width,[t["pitch_local_std_cents"],t["pitch_step_median_cents"],t["pitch_plateau_fraction"]*100],width=width,color=color,label=rec["label"])
        axes[1,1].bar(np.arange(3)+(i-.5)*width,t["formant_proxy_hz"],width=width,color=color,label=rec["label"])
    titles=[
        confidence_title("谐波集中、噪声平坦与高阶份额", "中", "音高跟踪、歌手与分离影响"),
        confidence_title("谐波衰减及响度相关纹理", "中低", "发声力度与元音可产生相同变化"),
        confidence_title("局部音高行为（辅助证据）", "低", "不同旋律不可直接 A/B"),
        confidence_title("共振峰代理", "低", "不同歌手/歌词不可直接 A/B"),
    ]
    for ax,title in zip(axes.flat,titles): ax.set_title(title,fontproperties=fp(13,"bold")); ax.grid(alpha=.22); ax.legend(prop=fp(9)); ax.axhline(0,color="#777",lw=.8)
    axes[0,0].set_xticks(range(3),["谐波bin\n集中度","频谱\n平坦度","5阶以上\n谐波份额"],fontproperties=fp(9))
    axes[0,1].set_xticks(range(2),["谐波衰减\ndB/oct","强声-弱声\n高频纹理dB"],fontproperties=fp(9))
    axes[1,0].set_xticks(range(3),["局部波动\ncents","帧间步进\ncents","<5 cents\n占比%"],fontproperties=fp(9))
    axes[1,1].set_xticks(range(3),["F1代理","F2代理","F3代理"],fontproperties=fp()); axes[1,1].set_ylabel("Hz",fontproperties=fp())
    fig.suptitle("细分图 D4｜谐波、噪声、响度相关失真与音高",fontproperties=fp(17,"bold"))
    fig.tight_layout(); fig.savefig(out/"D4_texture_pitch.png",dpi=165); plt.close(fig)


def write_report(records, out):
    a,b=records
    def diff(path):
        va=a
        vb=b
        for key in path: va=va[key]; vb=vb[key]
        return float(vb)-float(va)
    total_tails = sum(item["articulation"]["tail_count"] for item in records)
    tail_verdict = "条件性有用｜中低" if total_tails >= 6 else "不可比较｜证据不足"
    rows=[
        ("动态频段分布","有用｜中高","可区分长期偏色与偶发音素，仍受音素影响。"),
        ("音量相关音色","有用｜中","能发现强唱或弱唱才出现的问题；表演力度与元音是混杂项。"),
        ("持续共振候选","辅助｜中低","适合提示扫听位置；不同元音造成的峰不能直接当EQ处方。"),
        ("Presence / Air / 齿音分离","有用｜中高","宽频能量和事件占比可直接指导Presence、De-esser与Air。"),
        ("字头与音节平均包络","有用｜中高","能显示字头/主体关系和时间涂抹；非对齐歌词仍限制因果。"),
        ("辅音—元音比例","辅助｜中低","可提示硬、糊或齿音偏重，但高度依赖歌词音素。"),
        ("停顿后全频衰减",tail_verdict,"只有检测到足够多的长停顿才可靠；分离参考和呼吸会污染结果。"),
        ("宏观/微观动态","有用｜高","比单一Crest Factor更清楚地区分段落起伏和字头密度。"),
        ("音节间一致性/峰值密度","有用｜高","直接短窗与样本统计；能辅助选择Clip Gain、Rider、压缩或限幅。"),
        ("分频段动态","有用｜中高","可定位动态处理频段；低频受音高进入或离开频段影响。"),
        ("压缩/削波指纹","辅助｜中低","能看行为，不能反推压缩器型号、Attack或Release精确值。"),
        ("谐波/噪声比例","有用｜中","能区分规则颗粒与噪声型亮度；歌手、分离和MP3会产生伪影。"),
        ("谐波衰减与高阶份额","辅助｜中低","可描述明亮与饱和倾向，发声力度和元音仍是混杂因素。"),
        ("失真随响度变化","辅助｜中低","强声高频增长可支持非线性假设，但发声和元音也会造成同样结果。"),
        ("音高稳定/颤音/平台","辅助｜低","已避免跨停顿连接，但不同旋律仍无法公平A/B，不能用来证明Auto-Tune。"),
        ("共振峰运动/代理","当前不宜比较｜低","歌手和歌词不同，代理峰不足以指导插件或Formant参数。"),
    ]
    lines=["# 人声音色、咬字、动态与谐波细分诊断","",f"- {a['label']}：`{a['path']}`",f"- {b['label']}：`{b['path']}`","", "## 细分图","",
           "![D1](D1_timbre_detail.png)","","![D2](D2_articulation_time.png)","","![D3](D3_dynamics_detail.png)","","![D4](D4_texture_pitch.png)","",
           "## 这对音频实际测出的主要差异","",
           f"- 动态：{a['label']} 的宏观 P90–P10 为 {a['dynamics']['macro_p90_p10_db']:.2f} dB、微观为 {a['dynamics']['micro_p90_p10_db']:.2f} dB；{b['label']} 分别为 {b['dynamics']['macro_p90_p10_db']:.2f} 与 {b['dynamics']['micro_p90_p10_db']:.2f} dB。",
           f"- 字头/主体：中位代理为 {a['articulation']['transient_body_db']:.2f} dB 对 {b['articulation']['transient_body_db']:.2f} dB。值越高，检测到的起始峰相对后续主体越突出。",
           f"- 辅音/元音：高频辅音对中频元音主体代理为 {a['articulation']['consonant_vowel_proxy_db']:.2f} dB 对 {b['articulation']['consonant_vowel_proxy_db']:.2f} dB；只能结合音素看。",
           f"- 高频结构：谐波 bin 集中度为 {a['texture_pitch']['harmonic_bin_concentration']:.3f} 对 {b['texture_pitch']['harmonic_bin_concentration']:.3f}；频谱平坦度为 {a['texture_pitch']['spectral_flatness_median']:.5f} 对 {b['texture_pitch']['spectral_flatness_median']:.5f}。",
           f"- 谐波衰减：{a['texture_pitch']['harmonic_slope_db_per_oct']:.2f} 对 {b['texture_pitch']['harmonic_slope_db_per_oct']:.2f} dB/oct；越负表示高次谐波平均衰减越快。",
           f"- 音量相关高频纹理：强声减弱声为 {a['texture_pitch']['high_texture_growth_loud_vs_soft_db']:+.2f} 对 {b['texture_pitch']['high_texture_growth_loud_vs_soft_db']:+.2f} dB。它不是失真器识别结果。",
           f"- 低中频遮蔽代理：150–600 Hz 相对 600 Hz–5 kHz 为 {a['timbre']['low_mid_masking_db']:+.2f} 对 {b['timbre']['low_mid_masking_db']:+.2f} dB。",
           f"- 元音稳定代理：低变化帧的相邻频谱变化为 {a['timbre']['vowel_spectral_change_db']:.3f} 对 {b['timbre']['vowel_spectral_change_db']:.3f} dB；越低只代表短时谱形更稳定。",
           f"- 音节主体一致性：检测音节主体电平 IQR 为 {a['articulation']['syllable_body_level_iqr_db']:.2f} 对 {b['articulation']['syllable_body_level_iqr_db']:.2f} dB。",
           f"- 4倍过采样峰值：{a['dynamics']['true_peak_4x_dbfs']:+.2f} 对 {b['dynamics']['true_peak_4x_dbfs']:+.2f} dBFS；这是近似 True Peak 风险提示。",
           f"- 合格长句尾数量：{a['articulation']['tail_count']} 对 {b['articulation']['tail_count']}；任一数量过少时不应比较尾部曲线。",
           "", "## 每个细分角度是否有用", "", "| 角度 | 结论 | 原因与限制 |", "|---|---|---|"]
    lines += [f"| {name} | {verdict.split('｜')[0]}（可信度：{verdict.split('｜')[1]}） | {note} |" for name,verdict,note in rows]
    lines += ["", "## 总结", "", "脚本有用，但用途应分层：动态频段、宏/微动态、分频动态、音节包络、Presence/Air/齿音和谐波/噪声分离值得进入正式报告；辅音比例、压缩指纹和响度相关失真只作为辅助证据；音高平台与共振峰代理不应在这两个不同歌手、不同歌词且参考经过分离的文件之间做强结论。", "", "本实验没有识别任何具体插件。所有差异都还可能混入歌手、音域、歌词、录音、MP3和人声分离因素。"]
    (out/"experiment_report.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",action="append",required=True,help="标签=音频路径")
    parser.add_argument("--out-dir",type=Path,required=True)
    parser.add_argument("--sample-rate",type=int,default=44100)
    parser.add_argument("--segment-seconds",type=float,default=35.0)
    args=parser.parse_args(); args.out_dir.mkdir(parents=True,exist_ok=True)
    parsed=[]
    for item in args.input:
        if "=" not in item: raise SystemExit("--input 格式必须为 标签=路径")
        label,path=item.split("=",1); parsed.append((label,Path(path)))
    if len(parsed)!=2: raise SystemExit("本实验脚本要求恰好两个输入")
    records=[]; specs=[]
    for label,path in parsed:
        print(f"分析 {label} ...",flush=True)
        rec,spec=analyze(label,path,args.sample_rate,args.segment_seconds)
        records.append(rec); specs.append(spec)
    public=finite_obj({"schema":"vocal-detail-diagnostics-v1","records":records})
    (args.out_dir/"metrics.json").write_text(json.dumps(public,ensure_ascii=False,indent=2),encoding="utf-8")
    plot_timbre(records,specs,args.out_dir); plot_articulation(records,args.out_dir)
    plot_dynamics(records,args.out_dir); plot_texture(records,args.out_dir)
    write_report(records,args.out_dir)
    print(f"完成：{args.out_dir}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
