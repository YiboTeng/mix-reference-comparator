#!/usr/bin/env python3
"""对比一首 Target Mix 与一个或多个 Reference Mix，并生成中文图文报告。"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from mix_common import (
    BANDS, PALETTE, TARGET_SR, automatic_sections, band_frame_db,
    butter_filter, cap_confidence, confidence_title, db, envelope, finite,
    insufficient_figure, loudness_stats, mono, read_audio, rms, save_figure,
    section_stats, source_confidence, stereo_stats, style_axes, tonal_stats,
    transient_stats, write_json,
)

PLUGIN_ID = "mix-reference-comparator"
ANALYSIS_MODE = "mix-reference"
STEM_NAMES = ["vocal", "drums", "bass", "lead", "pad"]
STEM_ORIGINS = {"original_stems", "official_stems", "source_separated", "master_only"}


def resolve_path(value: str, base: Path) -> Path:
    """把项目清单中的相对路径稳定解析到清单所在目录。"""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def validate_sections(sections: list[dict[str, Any]], duration: float, label: str) -> list[dict[str, Any]]:
    """验证人工段落，防止越界或重叠的标签污染同段落比较。"""
    normalized = []
    for raw in sections:
        name = str(raw.get("name", "")).strip()
        start, end = float(raw.get("start", -1)), float(raw.get("end", -1))
        if not name or start < 0 or end <= start or end > duration + 0.05:
            raise ValueError(f"{label} 的段落无效：{raw}")
        normalized.append({"name": name, "start": start, "end": min(end, duration), "automatic": False})
    normalized.sort(key=lambda item: item["start"])
    for left, right in zip(normalized, normalized[1:]):
        if right["start"] < left["end"]:
            raise ValueError(f"{label} 的段落发生重叠：{left['name']} / {right['name']}")
    return normalized


def load_project(path: Path) -> dict[str, Any]:
    """读取并验证任务清单，尽早阻止路径、标签和来源字段错误。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "mix-reference-project-v1":
        raise ValueError("项目清单 schema 必须为 mix-reference-project-v1")
    if not isinstance(payload.get("target"), dict):
        raise ValueError("项目清单必须包含一个 target")
    references = payload.get("references")
    if not isinstance(references, list) or not references:
        raise ValueError("项目清单必须包含至少一个 reference")
    items = [payload["target"], *references]
    labels = [str(item.get("label", "")).strip() for item in items]
    if any(not label for label in labels) or len(set(labels)) != len(labels):
        raise ValueError("Target 与 Reference 的 label 必须非空且唯一")
    base = path.parent
    for item, label in zip(items, labels):
        if not item.get("mix"):
            raise ValueError(f"{label} 缺少 mix 路径")
        item["label"] = label
        item["mix"] = str(resolve_path(str(item["mix"]), base))
        if not Path(item["mix"]).exists():
            raise FileNotFoundError(item["mix"])
        if item.get("bpm") is not None and float(item["bpm"]) <= 0:
            raise ValueError(f"{label} 的 bpm 必须为正数")
        item["stem_origin"] = str(item.get("stem_origin", "master_only"))
        if item["stem_origin"] not in STEM_ORIGINS:
            allowed = ", ".join(sorted(STEM_ORIGINS))
            raise ValueError(f"{label} 的 stem_origin 无效；允许值：{allowed}")
        stems = item.get("stems") or {}
        resolved_stems = {}
        for name, value in stems.items():
            if name not in STEM_NAMES:
                continue
            stem_path = resolve_path(str(value), base)
            if not stem_path.exists():
                raise FileNotFoundError(stem_path)
            resolved_stems[name] = str(stem_path)
        if resolved_stems and item["stem_origin"] == "master_only":
            raise ValueError(f"{label} 已提供 Stems，必须声明其真实 stem_origin")
        if not resolved_stems and item["stem_origin"] != "master_only":
            raise ValueError(f"{label} 未提供 Stems，stem_origin 应为 master_only")
        item["stems"] = resolved_stems
    return payload


def active_mask(values: np.ndarray, percentile: float = 35) -> np.ndarray:
    """用轨道自身分位数定义活动帧，降低不同绝对增益造成的偏差。"""
    return values > np.percentile(values, percentile)


def conflict_profile(vocal: np.ndarray, instrument: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """统计两者同时活动时 Instrument 距 Vocal 不足 6 dB 的概率。"""
    _, vocal_bands = band_frame_db(mono(vocal))
    _, instrument_bands = band_frame_db(mono(instrument))
    count = min(vocal_bands.shape[1], instrument_bands.shape[1])
    vocal_bands, instrument_bands = vocal_bands[:, :count], instrument_bands[:, :count]
    vocal_active = np.mean(vocal_bands, axis=0) > np.percentile(np.mean(vocal_bands, axis=0), 35)
    instrument_active = np.mean(instrument_bands, axis=0) > np.percentile(np.mean(instrument_bands, axis=0), 25)
    valid = vocal_active & instrument_active
    if not np.any(valid):
        return np.zeros(len(BANDS)), np.zeros(len(BANDS))
    risk = np.mean(instrument_bands[:, valid] >= vocal_bands[:, valid] - 6.0, axis=1)
    margin = np.median(instrument_bands[:, valid] - vocal_bands[:, valid], axis=1)
    return risk, margin


def band_level_margin(vocal: np.ndarray, other: np.ndarray,
                      start_band: int, end_band: int) -> float:
    """只在人声活动帧比较指定频带的相对电平，不把它解释成具体 EQ。"""
    _, vocal_bands = band_frame_db(mono(vocal))
    _, other_bands = band_frame_db(mono(other))
    count = min(vocal_bands.shape[1], other_bands.shape[1])
    vocal_bands, other_bands = vocal_bands[:, :count], other_bands[:, :count]
    active = np.mean(vocal_bands, axis=0) > np.percentile(np.mean(vocal_bands, axis=0), 35)
    if not np.any(active):
        return math.nan
    return float(np.median(other_bands[start_band:end_band, :][:, active]
                           - vocal_bands[start_band:end_band, :][:, active]))


def sum_and_trim(stems: list[np.ndarray]) -> np.ndarray:
    """把可用乐器 Stem 裁到共同长度后相加，避免尾部长度不一致。"""
    length = min(len(stem) for stem in stems)
    return np.sum([stem[:length] for stem in stems], axis=0)


def section_vir(vocal: np.ndarray, instrument: np.ndarray,
                sections: list[dict[str, Any]]) -> dict[str, float | None]:
    """按人工或自动段落统计 Vocal-to-Instrument Ratio 的中位数。"""
    result = {}
    length = min(len(vocal), len(instrument))
    vocal, instrument = vocal[:length], instrument[:length]
    for section in sections:
        start = int(section["start"] * TARGET_SR)
        end = min(length, int(section["end"] * TARGET_SR))
        vocal_env, instrument_env = envelope(vocal[start:end]), envelope(instrument[start:end])
        count = min(len(vocal_env), len(instrument_env))
        vocal_env, instrument_env = vocal_env[:count], instrument_env[:count]
        mask = active_mask(vocal_env, 35)
        result[section["name"]] = (
            float(np.median(db(vocal_env[mask] / (instrument_env[mask] + 1e-12))))
            if np.any(mask) else None
        )
    return result


def analyze_stems(item: dict[str, Any], sections: list[dict[str, Any]]) -> dict[str, Any] | None:
    """仅在 Vocal 与至少一个乐器 Stem 同时存在时启用关系测量。"""
    paths = item.get("stems") or {}
    if "vocal" not in paths:
        return None
    instrument_names = [name for name in ["drums", "bass", "lead", "pad"] if name in paths]
    if not instrument_names:
        return None
    loaded, metadata = {}, {}
    for name, value in paths.items():
        loaded[name], metadata[name] = read_audio(Path(value))
    length = min(len(loaded[name]) for name in ["vocal", *instrument_names])
    vocal = loaded["vocal"][:length]
    instrument = sum_and_trim([loaded[name][:length] for name in instrument_names])
    vocal_env, instrument_env = envelope(vocal), envelope(instrument)
    count = min(len(vocal_env), len(instrument_env))
    vocal_env, instrument_env = vocal_env[:count], instrument_env[:count]
    valid = active_mask(vocal_env, 35)
    vir_frames = db(vocal_env[valid] / (instrument_env[valid] + 1e-12))
    risk, margin = conflict_profile(vocal, instrument)
    stem_db = {name: float(db(rms(audio[:length]))) for name, audio in loaded.items()}
    max_level = max(stem_db.values())
    relative_levels = {name: value - max_level for name, value in stem_db.items()}

    # Stem 活动数先按各自 P75 归一化，避免单纯由轨道增益决定“是否活动”。
    envelopes = {name: envelope(audio[:length]) for name, audio in loaded.items()}
    frame_count = min(len(values) for values in envelopes.values())
    normalized = np.column_stack([
        envelopes[name][:frame_count] / (np.percentile(envelopes[name], 75) + 1e-12)
        for name in loaded
    ])
    active_count = np.sum(normalized > 0.16, axis=1)
    vocal_index = list(loaded).index("vocal")
    vocal_active = normalized[:, vocal_index] > 0.16
    _, instrument_bands = band_frame_db(mono(instrument))
    spectral_occupancy = float(np.mean(np.sum(
        instrument_bands > np.max(instrument_bands, axis=0, keepdims=True) - 25.0,
        axis=0,
    )))
    vocal_width = stereo_stats(vocal)["side_mid_db"]
    instrument_width = stereo_stats(instrument)["side_mid_db"]
    result: dict[str, Any] = {
        "origin": item.get("stem_origin", "master_only"),
        "available_stems": list(loaded),
        "stem_metadata": metadata,
        "vir_median_db": float(np.median(vir_frames)) if len(vir_frames) else None,
        "vir_iqr_db": float(np.percentile(vir_frames, 75) - np.percentile(vir_frames, 25)) if len(vir_frames) else None,
        "section_vir_db": section_vir(vocal, instrument, sections),
        "mask_risk_by_band": risk,
        "instrument_margin_by_band_db": margin,
        "presence_mask_risk": float(np.mean(risk[5:7])),
        "hat_sibilance_mask_risk": float(np.mean(risk[7:9])),
        "vocal_side_mid_db": vocal_width,
        "instrument_side_mid_db": instrument_width,
        "spatial_width_gap_db": float(abs(vocal_width - instrument_width)),
        "stem_relative_db": relative_levels,
        "active_stems_mean": float(np.mean(active_count)),
        "active_stems_vocal": float(np.mean(active_count[vocal_active])) if np.any(vocal_active) else None,
        "dense_frame_pct": float(np.mean(active_count >= min(4, len(loaded))) * 100),
        "spectral_occupancy_bands": spectral_occupancy,
    }
    if "drums" in loaded:
        drums = loaded["drums"][:length]
        result["drums"] = {
            "overall": transient_stats(drums),
            "kick_low": transient_stats(butter_filter(drums, 20, 150)),
            "snare_mid": transient_stats(butter_filter(drums, 150, 2500)),
            "hat_high": transient_stats(butter_filter(drums, 5000, 11000)),
        }
    if "bass" in loaded:
        bass = loaded["bass"][:length]
        _, bass_bands = band_frame_db(mono(bass))
        energy = 10 ** (np.median(bass_bands, axis=1) / 10)
        total = np.sum(energy) + 1e-12
        bass_env_db = db(envelope(bass))
        bass_record = {
            "sub_pct": float(100 * energy[0] / total),
            "body_pct": float(100 * np.sum(energy[1:3]) / total),
            "upper_pct": float(100 * np.sum(energy[3:7]) / total),
            "crest_db": float(db(np.max(np.abs(bass)) / rms(bass))),
            "envelope_iqr_db": float(np.percentile(bass_env_db, 75) - np.percentile(bass_env_db, 25)),
            "side_mid_db": stereo_stats(bass)["side_mid_db"],
        }
        if "drums" in relative_levels:
            bass_record["relative_to_drums_db"] = relative_levels["bass"] - relative_levels["drums"]
        result["bass"] = bass_record
    if "lead" in loaded:
        result["lead"] = {
            "presence_margin_db": band_level_margin(vocal, loaded["lead"][:length], 5, 7),
            "relative_db": relative_levels["lead"],
        }
    if "pad" in loaded:
        result["pad"] = {
            "vocal_width_gap_db": float(abs(stereo_stats(loaded["pad"][:length])["side_mid_db"] - vocal_width)),
            "relative_db": relative_levels["pad"],
        }
    return result


def analyze_item(item: dict[str, Any]) -> dict[str, Any]:
    """汇总单首 Mix 的 Master 指标、结构指标和可选 Stem 指标。"""
    audio, metadata = read_audio(Path(item["mix"]))
    automatic, novelty = automatic_sections(audio)
    sections = validate_sections(item.get("sections") or [], metadata["duration_seconds"], item["label"]) or automatic
    return {
        "label": item["label"],
        "role": item["role"],
        "mix_metadata": metadata,
        "sections": sections,
        "section_metrics": section_stats(audio, sections),
        "section_source": "manual" if item.get("sections") else "automatic",
        "novelty": novelty,
        "loudness": loudness_stats(audio),
        "tonal": tonal_stats(audio),
        "stereo": stereo_stats(audio),
        "stems": analyze_stems(item, sections),
    }


def group_stem_records(records: dict[str, dict[str, Any]], required: list[str] | None = None) -> dict[str, dict[str, Any]]:
    """筛出具备同一组必要 Stem 的歌曲，防止缺失项进入成组比较。"""
    selected = {}
    for label, record in records.items():
        stems = record.get("stems")
        if not stems:
            continue
        if required and not all(key in stems for key in required):
            continue
        selected[label] = stems
    return selected


def group_confidence(stems: dict[str, dict[str, Any]], base: str) -> tuple[str, str]:
    """以组内最弱 Stem 来源作为该子图的可信度上限。"""
    caps, origins = [], set()
    for item in stems.values():
        _, cap, _ = source_confidence(item["origin"])
        caps.append(cap); origins.add(item["origin"])
    if not caps:
        return "证据不足", "缺少可用 Stem"
    level = base
    for cap in caps:
        level = cap_confidence(level, cap)
    if "source_separated" in origins:
        note = "含分离 Stem；泄漏与伪影已降级"
    elif "official_stems" in origins:
        note = "含官方 Stem；可能带总线处理"
    elif origins == {"original_stems"}:
        note = "工程原始 Stem"
    else:
        note = "Stem 来源不完全一致"
    return level, note


def labels_and_colors(records: dict[str, Any]) -> tuple[list[str], list[str]]:
    labels = list(records)
    return labels, [PALETTE[index % len(PALETTE)] for index in range(len(labels))]


def plot_master_loudness(records: dict[str, Any], out_dir: Path) -> None:
    labels, colors = labels_and_colors(records); x = np.arange(len(labels))
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].bar(x - 0.18, [records[k]["loudness"]["lufs_i"] for k in labels], 0.36, label="LUFS-I")
    axes[0, 0].bar(x + 0.18, [records[k]["loudness"]["true_peak_proxy_dbtp"] for k in labels], 0.36, label="dBTP proxy")
    axes[0, 0].set_title(confidence_title("Integrated Loudness / True Peak", "中高", "True Peak 为 4× 过采样代理"))
    axes[0, 1].bar(x - 0.18, [records[k]["loudness"]["plr_db"] for k in labels], 0.36, label="PLR")
    axes[0, 1].bar(x + 0.18, [records[k]["loudness"]["crest_db"] for k in labels], 0.36, label="Crest")
    axes[0, 1].set_title(confidence_title("Macro / micro dynamics", "中高", "编曲和段落会改变峰均关系"))
    axes[1, 0].bar(x - 0.18, [records[k]["loudness"]["lra_proxy_db"] for k in labels], 0.36, label="LRA proxy")
    axes[1, 0].bar(x + 0.18, [records[k]["loudness"]["limiter_density_pct"] for k in labels], 0.36, label="Limiter density %")
    axes[1, 0].set_title(confidence_title("LRA / limiter-density evidence", "中", "密度不能识别具体 Limiter"))
    for label, color in zip(labels, colors):
        axes[1, 1].plot(records[label]["loudness"]["momentary_times"], records[label]["loudness"]["momentary_proxy_db"], label=label, color=color, alpha=0.8)
    axes[1, 1].set_title(confidence_title("400 ms momentary-loudness proxy", "中高", "RMS 代理，不含完整 ITU gate"))
    axes[1, 1].set_xlabel("seconds")
    for axis in axes.flat:
        style_axes(axis); axis.legend(fontsize=8)
    for axis in axes.ravel()[:3]:
        axis.set_xticks(x, labels, rotation=15)
    save_figure(figure, out_dir / "M01_master_loudness_dynamics.png")


def plot_tonal(records: dict[str, Any], out_dir: Path) -> None:
    labels, colors = labels_and_colors(records); band_x = np.arange(len(BANDS))
    figure, axes = plt.subplots(1, 2, figsize=(15, 5))
    for label, color in zip(labels, colors):
        axes[0].plot(band_x, records[label]["tonal"]["band_relative_db"], marker="o", label=label, color=color)
        axes[1].plot(band_x, records[label]["tonal"]["band_dynamic_range_db"], marker="o", label=label, color=color)
    axes[0].set_title(confidence_title("Loudness-independent tonal balance", "中高", "宽频带相对化，不能直接当 EQ 增益"))
    axes[1].set_title(confidence_title("Dynamic tonal balance by band", "中高", "P90–P10 同时受编曲活动度影响"))
    for axis in axes:
        axis.set_xticks(band_x, [band[2] for band in BANDS], rotation=45, ha="right")
        axis.legend(fontsize=8); style_axes(axis)
    save_figure(figure, out_dir / "M02_tonal_balance.png")


def plot_stereo(records: dict[str, Any], out_dir: Path) -> None:
    labels, colors = labels_and_colors(records); x = np.arange(len(labels)); band_x = np.arange(len(BANDS))
    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    axes[0].bar(x - 0.18, [records[k]["stereo"]["side_mid_db"] for k in labels], 0.36, label="Side/Mid")
    axes[0].bar(x + 0.18, [records[k]["stereo"]["low_side_mid_db"] for k in labels], 0.36, label="Low Side/Mid")
    axes[0].set_title(confidence_title("Global / low-end width", "高", "M/S RMS 直接测量；负值不代表错误"))
    axes[1].bar(x - 0.18, [records[k]["stereo"]["correlation"] for k in labels], 0.36, label="Correlation")
    axes[1].bar(x + 0.18, [records[k]["stereo"]["mono_fold_loss_db"] for k in labels], 0.36, label="Mono loss dB")
    axes[1].set_title(confidence_title("Correlation / mono translation", "高", "直接由 L/R 波形计算"))
    for label, color in zip(labels, colors):
        axes[2].plot(band_x, records[label]["stereo"]["band_side_mid_db"], marker="o", label=label, color=color)
    axes[2].set_title(confidence_title("Frequency-dependent width", "高", "滤波边界附近存在少量泄漏"))
    axes[2].set_xticks(band_x, [band[2] for band in BANDS], rotation=45, ha="right")
    for axis in axes[:2]: axis.set_xticks(x, labels, rotation=15)
    for axis in axes:
        axis.legend(fontsize=8); style_axes(axis)
    save_figure(figure, out_dir / "M03_stereo_translation.png")


def normalized_section_series(record: dict[str, Any], key: str) -> tuple[np.ndarray, list[float]]:
    """把不同歌曲的段落映射到 0–100% 位置，只比较结构走势而非语义。"""
    values = list(record["section_metrics"].values())
    positions = np.linspace(0, 1, len(values)) if values else np.asarray([])
    return positions, [item[key] for item in values]


def plot_sections(records: dict[str, Any], out_dir: Path) -> None:
    labels, colors = labels_and_colors(records)
    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    for label, color in zip(labels, colors):
        positions, values = normalized_section_series(records[label], "lufs")
        axes[0].plot(positions, values, marker="o", label=label, color=color)
        positions, values = normalized_section_series(records[label], "side_mid_db")
        axes[1].plot(positions, values, marker="o", label=label, color=color)
    axes[0].set_title(confidence_title("Normalized section loudness", "中", "不同歌曲位置归一化，不等于同语义段落"))
    axes[1].set_title(confidence_title("Normalized section width", "中", "人工段落可提升到中高"))
    target = records[labels[0]]
    axes[2].plot(target["novelty"]["times"], target["novelty"]["novelty"], color=colors[0], label=labels[0])
    for boundary in target["novelty"]["boundaries"]:
        axes[2].axvline(boundary, color="#ff7f0e", ls=":", alpha=0.7)
    axes[2].set_title(confidence_title("Automatic structure evidence", "中", "只检测变化，不知道 Verse/Chorus 语义"))
    axes[2].set_xlabel("seconds")
    for axis in axes:
        axis.legend(fontsize=8); style_axes(axis)
    save_figure(figure, out_dir / "M04_section_structure.png")


def plot_vocal_balance(records: dict[str, Any], out_dir: Path) -> None:
    stems = group_stem_records(records)
    if list(records)[0] not in stems or len(stems) < 2:
        insufficient_figure(out_dir / "M05_vocal_instrument_balance.png",
                            ["Vocal-to-Instrument Ratio", "Balance stability", "Section balance"],
                            "缺少成组可比 Vocal/Instrument Stems")
        return
    labels = list(stems); x = np.arange(len(labels)); level, note = group_confidence(stems, "高")
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].bar(x, [stems[k]["vir_median_db"] for k in labels], color=PALETTE[:len(labels)])
    axes[0].set_title(confidence_title("Vocal-to-Instrument Ratio", level, note))
    axes[1].bar(x, [stems[k]["vir_iqr_db"] for k in labels], color=PALETTE[:len(labels)])
    axes[1].set_title(confidence_title("Vocal balance stability", cap_confidence("中高", level), "IQR 反映波动，不等同压缩好坏"))
    for label, color in zip(labels, PALETTE):
        values = [value for value in stems[label]["section_vir_db"].values() if value is not None]
        if values:
            axes[2].plot(np.linspace(0, 1, len(values)), values, marker="o", label=label, color=color)
    axes[2].set_title(confidence_title("Section vocal balance", cap_confidence("中", level), "自动段落只按归一化位置比较"))
    for axis in axes[:2]: axis.set_xticks(x, labels, rotation=15)
    for axis in axes:
        style_axes(axis)
    axes[2].legend(fontsize=8)
    save_figure(figure, out_dir / "M05_vocal_instrument_balance.png")


def plot_conflict(records: dict[str, Any], out_dir: Path) -> None:
    stems = group_stem_records(records)
    if list(records)[0] not in stems or len(stems) < 2:
        insufficient_figure(out_dir / "M06_frequency_conflict.png",
                            ["Active-time masking", "Presence conflict", "Hat/Sibilance conflict"],
                            "缺少成组可比 Vocal/Instrument Stems")
        return
    labels = list(stems); x = np.arange(len(labels)); band_x = np.arange(len(BANDS))
    level, note = group_confidence(stems, "中高")
    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    for label, color in zip(labels, PALETTE):
        axes[0].plot(band_x, np.asarray(stems[label]["mask_risk_by_band"]) * 100, marker="o", label=label, color=color)
    axes[0].set_xticks(band_x, [band[2] for band in BANDS], rotation=45, ha="right")
    axes[0].set_title(confidence_title("Active-time masking probability", level, note))
    axes[1].bar(x, [stems[k]["presence_mask_risk"] * 100 for k in labels], color=PALETTE[:len(labels)])
    axes[1].set_title(confidence_title("Vocal conflict 1.2–5 kHz", level, "能量阈值代理，不是完整听觉掩蔽模型"))
    axes[2].bar(x, [stems[k]["hat_sibilance_mask_risk"] * 100 for k in labels], color=PALETTE[:len(labels)])
    axes[2].set_title(confidence_title("Hat/Sibilance conflict 5–12 kHz", cap_confidence("中", level), "分离伪影会明显污染高频"))
    for axis in axes[1:]: axis.set_xticks(x, labels, rotation=15)
    axes[0].legend(fontsize=8)
    for axis in axes: style_axes(axis)
    save_figure(figure, out_dir / "M06_frequency_conflict.png")


def plot_fusion(records: dict[str, Any], out_dir: Path) -> None:
    stems = group_stem_records(records)
    if list(records)[0] not in stems or len(stems) < 2:
        insufficient_figure(out_dir / "M07_fusion_components.png",
                            ["Level integration", "Spectral integration", "Spatial integration"],
                            "缺少成组可比 Vocal/Instrument Stems")
        return
    labels = list(stems); x = np.arange(len(labels)); refs = labels[1:]
    reference_vir = float(np.median([stems[k]["vir_median_db"] for k in refs]))
    level, note = group_confidence(stems, "中高")
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    level_distance = [abs(stems[k]["vir_median_db"] - reference_vir) for k in labels]
    axes[0].bar(x, level_distance, color=PALETTE[:len(labels)])
    axes[0].set_title(confidence_title("Level distance to references", level, "越小仅表示接近参考，不代表绝对更好"))
    axes[1].bar(x, [stems[k]["presence_mask_risk"] * 100 for k in labels], color=PALETTE[:len(labels)])
    axes[1].set_title(confidence_title("Spectral-integration conflict", level, "活动帧能量关系，不识别具体 EQ"))
    axes[2].bar(x, [stems[k]["spatial_width_gap_db"] for k in labels], color=PALETTE[:len(labels)])
    axes[2].set_title(confidence_title("Vocal–instrument spatial gap", level, note))
    for axis in axes:
        axis.set_xticks(x, labels, rotation=15); style_axes(axis)
    save_figure(figure, out_dir / "M07_fusion_components.png")


def plot_arrangement(records: dict[str, Any], out_dir: Path) -> None:
    stems = group_stem_records(records)
    if list(records)[0] not in stems or len(stems) < 2:
        insufficient_figure(out_dir / "M08_stem_balance_arrangement.png",
                            ["Stem balance", "Active Stem count", "Spectral occupancy"],
                            "缺少成组可比 Stems")
        return
    labels = list(stems); stem_names = sorted(set.intersection(*[set(item["stem_relative_db"]) for item in stems.values()]))
    x = np.arange(len(labels)); stem_x = np.arange(len(stem_names)); level, note = group_confidence(stems, "中高")
    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    for label, color in zip(labels, PALETTE):
        axes[0].plot(stem_x, [stems[label]["stem_relative_db"][name] for name in stem_names], marker="o", label=label, color=color)
    axes[0].set_xticks(stem_x, stem_names)
    axes[0].set_title(confidence_title("Stem balance profile", level, note))
    axes[1].bar(x - 0.18, [stems[k]["active_stems_mean"] for k in labels], 0.36, label="all")
    axes[1].bar(x + 0.18, [stems[k]["active_stems_vocal"] for k in labels], 0.36, label="vocal active")
    axes[1].set_title(confidence_title("Arrangement density", level, "依赖 Stem 划分粒度与活动阈值"))
    axes[2].bar(x, [stems[k]["spectral_occupancy_bands"] for k in labels], color=PALETTE[:len(labels)])
    axes[2].set_title(confidence_title("Spectral occupancy", level, "每帧距最强频带 25 dB 内的频带数"))
    for axis in axes[1:]: axis.set_xticks(x, labels, rotation=15)
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
    for axis in axes: style_axes(axis)
    save_figure(figure, out_dir / "M08_stem_balance_arrangement.png")


def plot_drums_bass(records: dict[str, Any], out_dir: Path) -> None:
    stems = group_stem_records(records, ["drums", "bass"])
    if list(records)[0] not in stems or len(stems) < 2:
        insufficient_figure(out_dir / "M09_drums_bass.png",
                            ["Drum band transients", "Drum punch", "Bass spectrum", "Bass dynamics", "Bass mono", "Bass vs drums"],
                            "缺少成组可比 Drums/Bass Stems")
        return
    labels = list(stems); x = np.arange(len(labels)); width = 0.25
    level, note = group_confidence(stems, "中高")
    figure, axes = plt.subplots(2, 3, figsize=(18, 10))
    for index, key in enumerate(["kick_low", "snare_mid", "hat_high"]):
        axes[0, 0].bar(x + (index - 1) * width,
                       [stems[k]["drums"][key]["transient_density_per_min"] for k in labels],
                       width, label=key)
    axes[0, 0].set_title(confidence_title("Kick/Snare/Hat band proxies", cap_confidence("中", level), "频带代理不是鼓件分类器"))
    axes[0, 1].bar(x - 0.18, [stems[k]["drums"]["overall"]["crest_db"] for k in labels], 0.36, label="Crest")
    axes[0, 1].bar(x + 0.18, [stems[k]["drums"]["overall"]["p95_flux"] * 1000 for k in labels], 0.36, label="Flux ×1000")
    axes[0, 1].set_title(confidence_title("Drum punch / attack", level, "受 Drum Stem 总线处理影响"))
    for index, key in enumerate(["sub_pct", "body_pct", "upper_pct"]):
        axes[0, 2].bar(x + (index - 1) * width, [stems[k]["bass"][key] for k in labels], width, label=key)
    axes[0, 2].set_title(confidence_title("Bass spectral balance", level, note))
    axes[1, 0].bar(x - 0.18, [stems[k]["bass"]["crest_db"] for k in labels], 0.36, label="Crest")
    axes[1, 0].bar(x + 0.18, [stems[k]["bass"]["envelope_iqr_db"] for k in labels], 0.36, label="Envelope IQR")
    axes[1, 0].set_title(confidence_title("Bass dynamics", level, "不等同 Compressor 参数"))
    axes[1, 1].bar(x, [stems[k]["bass"]["side_mid_db"] for k in labels], color=PALETTE[:len(labels)])
    axes[1, 1].set_title(confidence_title("Bass mono / side energy", level, "Bass Stem M/S 直接测量"))
    axes[1, 2].bar(x, [stems[k]["bass"].get("relative_to_drums_db", math.nan) for k in labels], color=PALETTE[:len(labels)])
    axes[1, 2].set_title(confidence_title("Bass relative to drums", level, "只比较电平关系，不推断 Sidechain"))
    for axis in axes.flat:
        axis.set_xticks(x, labels, rotation=15); style_axes(axis)
    for axis in [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0]]: axis.legend(fontsize=8)
    save_figure(figure, out_dir / "M09_drums_bass.png")


def plot_lead_pad(records: dict[str, Any], out_dir: Path) -> None:
    stems = group_stem_records(records, ["lead", "pad"])
    if list(records)[0] not in stems or len(stems) < 2:
        insufficient_figure(out_dir / "M10_lead_pad_occupancy.png",
                            ["Lead presence margin", "Pad spatial gap", "Lead/Pad relative level"],
                            "缺少成组可比 Vocal/Lead/Pad Stems")
        return
    labels = list(stems); x = np.arange(len(labels)); level, note = group_confidence(stems, "中高")
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].bar(x, [stems[k]["lead"]["presence_margin_db"] for k in labels], color=PALETTE[:len(labels)])
    axes[0].set_title(confidence_title("Lead-vs-Vocal presence margin", level, "1.2–5 kHz 活动帧电平差"))
    axes[1].bar(x, [stems[k]["pad"]["vocal_width_gap_db"] for k in labels], color=PALETTE[:len(labels)])
    axes[1].set_title(confidence_title("Pad-vs-Vocal width gap", level, "M/S 差不等同前后距离"))
    axes[2].bar(x - 0.18, [stems[k]["lead"]["relative_db"] for k in labels], 0.36, label="lead")
    axes[2].bar(x + 0.18, [stems[k]["pad"]["relative_db"] for k in labels], 0.36, label="pad")
    axes[2].set_title(confidence_title("Lead / Pad relative level", level, note))
    for axis in axes:
        axis.set_xticks(x, labels, rotation=15); style_axes(axis)
    axes[2].legend(fontsize=8)
    save_figure(figure, out_dir / "M10_lead_pad_occupancy.png")


def reference_features(records: dict[str, Any]) -> tuple[list[str], dict[str, list[float]]]:
    """提取跨歌曲较稳定的 Master 特征，供稳健参考区间计算。"""
    names = ["LUFS-I", "PLR", "Sub", "Presence", "High Side", "Mono Loss"]
    values = {}
    for label, record in records.items():
        values[label] = [
            record["loudness"]["lufs_i"],
            record["loudness"]["plr_db"],
            record["tonal"]["band_relative_db"][0],
            float(np.mean(record["tonal"]["band_relative_db"][5:7])),
            float(np.mean(record["stereo"]["band_side_mid_db"][7:9])),
            record["stereo"]["mono_fold_loss_db"],
        ]
    return names, values


def reference_summary(records: dict[str, Any]) -> dict[str, Any]:
    """用 Median/IQR 估计参考中心、方向一致性和离群程度。"""
    names, features = reference_features(records)
    labels = list(records); target = np.asarray(features[labels[0]], dtype=float)
    references = np.asarray([features[label] for label in labels[1:]], dtype=float)
    median = np.median(references, axis=0)
    q1, q3 = np.percentile(references, [25, 75], axis=0)
    scale = np.maximum(np.maximum(q3 - q1, np.std(references, axis=0)), 0.25)
    differences = target[None, :] - references
    direction = np.maximum(np.mean(differences >= 0, axis=0), np.mean(differences <= 0, axis=0))
    feature_summary = {
        name: {
            "target": target[index], "reference_median": median[index],
            "reference_q1": q1[index], "reference_q3": q3[index],
            "robust_deviation": (target[index] - median[index]) / scale[index],
            "direction_consistency": direction[index],
        }
        for index, name in enumerate(names)
    }
    outliers = {}
    for row, label in zip(references, labels[1:]):
        outliers[label] = float(np.median(np.abs((row - median) / scale)))
    return {"features": feature_summary, "outliers": outliers}


def cohort_confidence(count: int) -> tuple[str, str]:
    """参考数量只决定统计区间上限，不代表素材本身一定可比。"""
    if count == 1: return "低", "只有一个参考，不能称为参考共性"
    if count == 2: return "中低", "两个参考只能形成初步范围"
    if count < 5: return "中", "3–4 个独立参考可形成初步稳健区间"
    return "中高", "至少 5 个可比参考；仍不是跨曲风工业常模"


def plot_reference_set(records: dict[str, Any], summary: dict[str, Any], out_dir: Path) -> None:
    labels = list(records); refs = labels[1:]; names = list(summary["features"])
    confidence, limitation = cohort_confidence(len(refs))
    x = np.arange(len(names))
    figure, axes = plt.subplots(1, 2, figsize=(16, 5))
    deviations = [summary["features"][name]["robust_deviation"] for name in names]
    axes[0].bar(x, deviations, color=PALETTE[0])
    axes[0].axhline(0, color="#222", lw=1)
    axes[0].set_xticks(x, names, rotation=35, ha="right")
    axes[0].set_title(confidence_title("Target robust deviation", confidence, limitation))
    ref_x = np.arange(len(refs))
    axes[1].bar(ref_x, [summary["outliers"][label] for label in refs], color=PALETTE[1:1+len(refs)])
    axes[1].set_xticks(ref_x, refs, rotation=15)
    axes[1].set_title(confidence_title("Reference outlier score", confidence, "小样本只作提醒，不自动剔除"))
    for axis in axes: style_axes(axis)
    save_figure(figure, out_dir / "M11_reference_intervals.png")


def compatibility(records: dict[str, Any], project: dict[str, Any]) -> dict[str, Any]:
    """用音色、时长和可选 BPM 做参考筛选代理，不充当 Genre/Key 分类器。"""
    labels = list(records); target = records[labels[0]]
    target_tonal = np.asarray(target["tonal"]["band_relative_db"])
    target_duration = target["mix_metadata"]["duration_seconds"]
    project_items = {item["label"]: item for item in [project["target"], *project["references"]]}
    result = {}
    for label in labels[1:]:
        record = records[label]
        tonal = np.asarray(record["tonal"]["band_relative_db"])
        tonal_distance = float(np.sqrt(np.mean((target_tonal - tonal) ** 2)))
        tonal_score = max(0.0, 100 - tonal_distance * 8)
        duration_ratio = min(target_duration, record["mix_metadata"]["duration_seconds"]) / max(target_duration, record["mix_metadata"]["duration_seconds"])
        duration_score = duration_ratio * 100
        target_bpm = project_items[labels[0]].get("bpm")
        reference_bpm = project_items[label].get("bpm")
        if target_bpm and reference_bpm:
            tempo_score = max(0.0, 100 - abs(float(target_bpm) - float(reference_bpm)) * 6)
            score = 0.45 * tonal_score + 0.35 * tempo_score + 0.20 * duration_score
            confidence = "中"
            note = "BPM 来自项目清单；频谱距离仍不是 Genre/Key 分类"
        else:
            tempo_score = None
            score = 0.70 * tonal_score + 0.30 * duration_score
            confidence = "中低"
            note = "未提供 BPM；只使用音色与时长兼容性代理"
        result[label] = {
            "score": score, "tonal_score": tonal_score, "tempo_score": tempo_score,
            "duration_score": duration_score, "confidence": confidence, "limitation": note,
        }
    return result


def plot_matching(records: dict[str, Any], project: dict[str, Any], compatibility_data: dict[str, Any], out_dir: Path) -> None:
    refs = list(records)[1:]; x = np.arange(len(refs))
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(x, [compatibility_data[label]["score"] for label in refs], color=PALETTE[1:1+len(refs)])
    axes[0].set_xticks(x, refs, rotation=15); axes[0].set_ylim(0, 105)
    overall_conf = "中" if all(compatibility_data[label]["tempo_score"] is not None for label in refs) else "中低"
    axes[0].set_title(confidence_title("Reference compatibility", overall_conf, "用于筛选参考，不是曲风分类器"))
    items = {item["label"]: item for item in [project["target"], *project["references"]]}
    source_scores, source_levels = [], []
    for label in refs:
        score, level, _ = source_confidence(items[label].get("stem_origin", "master_only"))
        source_scores.append(score * 100); source_levels.append(level)
    axes[1].bar(x, source_scores, color=PALETTE[1:1+len(refs)])
    axes[1].set_xticks(x, refs, rotation=15); axes[1].set_ylim(0, 105)
    axes[1].set_title(confidence_title("Source-quality confidence", "高", "规则透明；来源标签必须人工确认"))
    for axis in axes: style_axes(axis)
    save_figure(figure, out_dir / "M12_reference_matching_confidence.png")


def public_record(record: dict[str, Any]) -> dict[str, Any]:
    """移除仅用于绘图的大数组，保持主指标 JSON 可读。"""
    return {
        "label": record["label"], "role": record["role"],
        "mix_metadata": record["mix_metadata"],
        "section_source": record["section_source"],
        "sections": record["sections"], "section_metrics": record["section_metrics"],
        "detected_boundaries": record["novelty"]["boundaries"],
        "loudness": {key: value for key, value in record["loudness"].items()
                     if key not in ["momentary_proxy_db", "momentary_times"]},
        "tonal": record["tonal"], "stereo": record["stereo"], "stems": record["stems"],
    }


def median_reference(records: dict[str, Any], getter) -> float:
    labels = list(records)[1:]
    return float(np.median([getter(records[label]) for label in labels]))


def format_delta(value: float | None) -> str:
    return "—" if value is None or not np.isfinite(value) else f"{value:+.2f}"


def write_report(project: dict[str, Any], records: dict[str, Any], summary: dict[str, Any],
                 compatibility_data: dict[str, Any], out_dir: Path) -> None:
    """把可测差异、可信度、优先级和明确限制写成中文 Markdown 报告。"""
    labels = list(records); target = records[labels[0]]; reference_count = len(labels) - 1
    cohort_level, cohort_note = cohort_confidence(reference_count)
    rows = []
    comparisons = [
        ("LUFS-I", target["loudness"]["lufs_i"], median_reference(records, lambda r: r["loudness"]["lufs_i"]), "中高"),
        ("PLR dB", target["loudness"]["plr_db"], median_reference(records, lambda r: r["loudness"]["plr_db"]), "中高"),
        ("Sub relative dB", target["tonal"]["band_relative_db"][0], median_reference(records, lambda r: r["tonal"]["band_relative_db"][0]), "中高"),
        ("High Side/Mid dB", float(np.mean(target["stereo"]["band_side_mid_db"][7:9])), median_reference(records, lambda r: float(np.mean(r["stereo"]["band_side_mid_db"][7:9]))), "高"),
        ("Mono fold loss dB", target["stereo"]["mono_fold_loss_db"], median_reference(records, lambda r: r["stereo"]["mono_fold_loss_db"]), "高"),
    ]
    stem_records = group_stem_records(records)
    if labels[0] in stem_records and len(stem_records) >= 2:
        stem_refs = [stem_records[label] for label in labels[1:] if label in stem_records]
        source_level, _ = group_confidence({label: stem_records[label] for label in stem_records}, "中高")
        comparisons += [
            ("Vocal/Instrument dB", stem_records[labels[0]]["vir_median_db"], float(np.median([item["vir_median_db"] for item in stem_refs])), source_level),
            ("Presence mask %", stem_records[labels[0]]["presence_mask_risk"] * 100, float(np.median([item["presence_mask_risk"] * 100 for item in stem_refs])), source_level),
            ("Spatial width gap dB", stem_records[labels[0]]["spatial_width_gap_db"], float(np.median([item["spatial_width_gap_db"] for item in stem_refs])), source_level),
        ]
    for name, target_value, reference_value, confidence in comparisons:
        rows.append((name, float(target_value), float(reference_value), float(target_value - reference_value), confidence))

    priorities = []
    lookup = {row[0]: row for row in rows}
    if abs(lookup["LUFS-I"][3]) > 1:
        priorities.append("先把 Target 与参考做等响度 A/B，再判断音色和动态，避免把更响误判为更好。")
    if lookup["Sub relative dB"][3] > 1.5:
        priorities.append("检查 20–150 Hz 的 Bass/Kick 总量与 Headroom，优先用编曲、电平或 Dynamic EQ 做可撤回调整。")
    if lookup["High Side/Mid dB"][3] > 2.5:
        priorities.append("收窄 5–12 kHz 的 Side 或降低宽化返回，并重点复核 Mono Fold-down。")
    if "Vocal/Instrument dB" in lookup and lookup["Vocal/Instrument dB"][3] < -1.5:
        priorities.append("Target 人声相对伴奏偏低；先试 Vocal/Instrument 电平与 1.2–5 kHz 让位，再考虑增加压缩。")
    if "Presence mask %" in lookup and lookup["Presence mask %"][3] > 5:
        priorities.append("Vocal 活动时 1.2–5 kHz 冲突偏高；对 Lead/伴奏做动态让位，并用旁路等响度复核。")
    if not priorities:
        priorities.append("主要宏观指标已接近参考中位数；优先监听段落过渡、瞬态和人声可懂度，而不是追逐小数值差。")
    priorities = priorities[:5]

    lines = [
        "# 完整混音参考对比报告", "",
        "## 一句话结论", "",
        f"Target 已与 {reference_count} 个完整混音参考完成 Master 层比较；"
        + ("同时具备可比 Stem，已启用关系诊断。" if labels[0] in stem_records and len(stem_records) >= 2 else "缺少成组可比 Stem，关系图按证据不足处理。"),
        "", "## 输入与证据边界", "",
        "| 角色 | 标签 | 文件 | Stem 来源 | 段落 |", "|---|---|---|---|---|",
    ]
    project_items = {item["label"]: item for item in [project["target"], *project["references"]]}
    for label in labels:
        item = project_items[label]
        lines.append(f"| {records[label]['role']} | {label} | `{item['mix']}` | {item.get('stem_origin', 'master_only')} | {records[label]['section_source']} |")
    lines += [
        "",
        f"多参考区间可信度：**{cohort_level}**。{cohort_note}。Master 可以直接支持响度、频谱、动态和立体声测量；Vocal/Instrument、鼓、Bass、Lead/Pad 结论只在相应 Stem 存在时成立。",
        "", "## 核心测量", "",
        "| 指标 | Target | Reference Median | Target − Reference | 可信度 |", "|---|---:|---:|---:|---|",
    ]
    for name, target_value, reference_value, delta, confidence in rows:
        lines.append(f"| {name} | {target_value:.2f} | {reference_value:.2f} | {format_delta(delta)} | {confidence} |")
    lines += ["", "差值只描述方向，不等于建议直接复制成 EQ、Gain 或 Width 参数。", "", "## 图表", ""]
    charts = [
        ("M01_master_loudness_dynamics.png", "Master 响度、动态与峰值"),
        ("M02_tonal_balance.png", "静态与动态音色平衡"),
        ("M03_stereo_translation.png", "立体声与 Mono Translation"),
        ("M04_section_structure.png", "段落与结构"),
        ("M05_vocal_instrument_balance.png", "人声与伴奏电平"),
        ("M06_frequency_conflict.png", "频率冲突"),
        ("M07_fusion_components.png", "融合组件拆解"),
        ("M08_stem_balance_arrangement.png", "Stem 平衡与编曲密度"),
        ("M09_drums_bass.png", "Drums 与 Bass"),
        ("M10_lead_pad_occupancy.png", "Lead/Pad 占位"),
        ("M11_reference_intervals.png", "多参考稳健偏差与离群"),
        ("M12_reference_matching_confidence.png", "参考匹配与来源可信度"),
    ]
    for filename, title in charts:
        lines += [f"### {title}", "", f"![{title}]({filename})", ""]
    lines += ["## 最高优先级调整", ""]
    for index, priority in enumerate(priorities, 1):
        lines.append(f"{index}. {priority}")
    lines += [
        "", "## 处理链结构", "",
        "```text", "Input/Stems → Gain staging → Tonal balance → Dynamic control → Vocal/Instrument space → Stereo/mono check → Limiter", "```",
        "", "## 限制", "",
        "- 不从最终 Master 反推具体 Plugin、Preset、母带链或精确单轨参数。",
        "- 自动结构只检测变化，不知道真实 Verse/Chorus；人工段落优先。",
        "- Source Separation 会引入泄漏、瞬态涂抹、Musical Noise 和虚假宽度。",
        "- 简单 Vocal/Instrument 或 Kick/Bass 包络相关没有通过受控实验，报告不把它写成 Sidechain 证明。",
        "- 曲风、编曲、母带年代、平台版本和 Codec 会改变参考可比性。",
        "", "每项调整都应先做等响度旁路 A/B；当清晰度、冲击力或 Mono Translation 开始变差时停止。",
    ]
    (out_dir / "mix-reference-report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    """统一编排验证、分析、12 张图、指标、报告和可复现执行日志。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    started = time.time()
    project_path = args.project.resolve()
    out_dir = args.out_dir.resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    project = load_project(project_path)
    project["target"]["role"] = "target"
    for reference in project["references"]:
        reference["role"] = "reference"
    items = [project["target"], *project["references"]]
    records = {item["label"]: analyze_item(item) for item in items}

    plot_master_loudness(records, out_dir)
    plot_tonal(records, out_dir)
    plot_stereo(records, out_dir)
    plot_sections(records, out_dir)
    plot_vocal_balance(records, out_dir)
    plot_conflict(records, out_dir)
    plot_fusion(records, out_dir)
    plot_arrangement(records, out_dir)
    plot_drums_bass(records, out_dir)
    plot_lead_pad(records, out_dir)
    summary = reference_summary(records)
    plot_reference_set(records, summary, out_dir)
    compatibility_data = compatibility(records, project)
    plot_matching(records, project, compatibility_data, out_dir)
    write_report(project, records, summary, compatibility_data, out_dir)

    charts = [f"M{index:02d}_{name}.png" for index, name in enumerate([
        "master_loudness_dynamics", "tonal_balance", "stereo_translation", "section_structure",
        "vocal_instrument_balance", "frequency_conflict", "fusion_components",
        "stem_balance_arrangement", "drums_bass", "lead_pad_occupancy",
        "reference_intervals", "reference_matching_confidence",
    ], 1)]
    payload = {
        "plugin_id": PLUGIN_ID, "analysis_mode": ANALYSIS_MODE,
        "schema": "mix-reference-metrics-v1", "project": str(project_path),
        "target": project["target"]["label"],
        "references": [item["label"] for item in project["references"]],
        "records": {label: public_record(record) for label, record in records.items()},
        "reference_set": summary, "compatibility": compatibility_data,
        "charts": charts, "report": "mix-reference-report.md",
    }
    write_json(out_dir / "mix-reference-metrics.json", payload)
    execution = {
        "plugin_id": PLUGIN_ID, "analysis_mode": ANALYSIS_MODE,
        "schema": "mix-reference-execution-v1", "python": sys.executable,
        "project": str(project_path), "out_dir": str(out_dir),
        "elapsed_seconds": round(time.time() - started, 3),
        "status": "completed", "charts": charts,
    }
    write_json(out_dir / "execution-log.json", execution)
    print(json.dumps({
        "plugin_id": PLUGIN_ID, "analysis_mode": ANALYSIS_MODE,
        "report": str(out_dir / "mix-reference-report.md"),
        "metrics": str(out_dir / "mix-reference-metrics.json"),
        "execution_log": str(out_dir / "execution-log.json"),
        "charts": [str(out_dir / name) for name in charts],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
