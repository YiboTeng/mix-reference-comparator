#!/usr/bin/env python3
"""将一条待评估人声与一个或多个参考人声进行分层对比。

脚本把现有基础、细节和空间诊断器作为一对一内核逐个运行，再生成参考区间、
共同偏差、参考离群度、中文汇总图和报告。它只读取源音频，不修改源文件。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
except ImportError as exc:
    raise SystemExit(
        f"缺少依赖：{exc.name}。请先运行同目录 bootstrap_deps.py 安装分析依赖。"
    ) from exc


PLUGIN_ID = "mix-reference-comparator"
ANALYSIS_MODE = "vocal-reference"
BANDS = [
    "20-80", "80-150", "150-300", "300-600", "600-1200",
    "1200-2500", "2500-5000", "5000-8000", "8000-12000",
    "12000-18000",
]
BAND_LABELS = [
    "20–80", "80–150", "150–300", "300–600", "600–1.2k",
    "1.2–2.5k", "2.5–5k", "5–8k", "8–12k", "12–18k",
]


@dataclass(frozen=True)
class AudioSpec:
    label: str
    path: Path
    separated: bool = False


@dataclass(frozen=True)
class FeatureSpec:
    key: str
    label: str
    path: tuple[str, ...]
    unit: str
    practical_floor: float
    confidence: str
    limitation: str
    group: str


FEATURES = [
    FeatureSpec("integrated_lufs", "综合响度", ("basic", "integrated_lufs"), "LUFS", 1.0,
                "高", "只描述电平，不代表音质", "levels"),
    FeatureSpec("rms_dbfs", "全曲 RMS", ("basic", "rms_dbfs"), "dBFS", 1.0,
                "高", "静音比例和段落结构会影响", "levels"),
    FeatureSpec("crest_factor_db", "峰均比", ("basic", "crest_factor_db"), "dB", 0.8,
                "高", "无法唯一反推压缩器", "dynamics"),
    FeatureSpec("active_range_db", "活跃电平范围", ("basic", "active_50ms_rms_db", "p90_p10_range"), "dB", 1.0,
                "高", "演唱力度和段落结构会影响", "dynamics"),
    FeatureSpec("presence_db", "Presence", ("detail", "timbre", "presence_median_db_relative"), "dB", 1.0,
                "中高", "受歌手、音域和辅音比例影响", "timbre"),
    FeatureSpec("air_db", "Air", ("detail", "timbre", "air_median_db_relative"), "dB", 1.5,
                "中高", "受编码、齿音和分离残留影响", "timbre"),
    FeatureSpec("low_mid_masking_db", "低中频遮蔽", ("detail", "timbre", "low_mid_masking_db"), "dB", 1.0,
                "中", "声线和元音构成是混杂因素", "timbre"),
    FeatureSpec("macro_range_db", "宏观动态", ("detail", "dynamics", "macro_p90_p10_db"), "dB", 1.0,
                "高", "段落结构会改变范围", "dynamics"),
    FeatureSpec("micro_range_db", "微观动态", ("detail", "dynamics", "micro_p90_p10_db"), "dB", 0.8,
                "高", "无法唯一反推压缩参数", "dynamics"),
    FeatureSpec("harmonic_concentration", "规则谐波集中", ("detail", "texture_pitch", "harmonic_bin_concentration"), "比例", 0.04,
                "中", "音高跟踪和分离伪影会影响", "texture"),
    FeatureSpec("upper_harmonic_share", "高阶谐波份额", ("detail", "texture_pitch", "upper_harmonic_share"), "比例", 0.04,
                "中", "亮辅音与规则谐波会混合", "texture"),
    FeatureSpec("spectral_flatness", "频谱平坦度", ("detail", "texture_pitch", "spectral_flatness_median"), "比例", 0.015,
                "中", "噪声、气声、编码和分离均可提高", "texture"),
    FeatureSpec("side_mid_db", "全曲 Side/Mid", ("basic", "stereo_full_file", "side_to_mid_db"), "dB", 1.0,
                "中", "宽度可测，但来源并不唯一", "spatial"),
    FeatureSpec("lr_correlation", "左右相关度", ("basic", "stereo_full_file", "correlation"), "相关系数", 0.08,
                "中", "分离残留和混响都会改变", "spatial"),
    FeatureSpec("width_core_db", "核心宽度", ("basic", "width_by_stage", "core"), "dB", 1.0,
                "中", "无法单独区分双轨、延迟和调制", "spatial"),
    FeatureSpec("width_tails_db", "尾部宽度", ("basic", "width_by_stage", "tails"), "dB", 1.0,
                "中", "静音门限与分离残留会影响", "spatial"),
]


def parse_labeled_audio(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("格式必须为 标签=音频路径")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    path = Path(raw_path.strip()).expanduser()
    if not label or not raw_path.strip():
        raise argparse.ArgumentTypeError("标签和音频路径都不能为空")
    return label, path


def setup_font(explicit: str | None) -> str | None:
    candidates = [explicit] if explicit else []
    if os.name == "nt":
        candidates += [
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\simhei.ttf",
            r"C:\Windows\Fonts\simsun.ttc",
        ]
    candidates += ["/System/Library/Fonts/PingFang.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            name = font_manager.FontProperties(fname=candidate).get_name()
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return candidate
    plt.rcParams["axes.unicode_minus"] = False
    return None


def confidence_title(title: str, level: str, limitation: str) -> str:
    return f"{title}\n可信度：{level}｜{limitation}"


def safe_slug(label: str, index: int) -> str:
    ascii_part = re.sub(r"[^a-zA-Z0-9]+", "-", label).strip("-").lower()
    return f"ref-{index:02d}" + (f"-{ascii_part[:36]}" if ascii_part else "")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run_checked(command: list[str], log: list[dict[str, Any]]) -> None:
    started = time.time()
    result = subprocess.run(command, text=True, capture_output=True, encoding="utf-8", errors="replace")
    entry = {
        "command": command,
        "returncode": result.returncode,
        "duration_seconds": round(time.time() - started, 3),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    log.append(entry)
    if result.returncode != 0:
        raise RuntimeError(
            "子分析器运行失败：\n"
            + " ".join(command)
            + f"\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def record_by_label(records: list[dict[str, Any]], label: str) -> dict[str, Any]:
    for record in records:
        if record.get("label") == label:
            return record
    raise KeyError(f"细分指标中缺少标签：{label}")


def build_pairwise(
    target: AudioSpec,
    references: list[AudioSpec],
    out_dir: Path,
    scripts_dir: Path,
    analysis_level: str,
    basic_seconds: float,
    detail_seconds: float,
    spatial_seconds: float,
    sample_rate: int,
    font: str | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pairwise: dict[str, Any] = {}
    execution_log: list[dict[str, Any]] = []
    for index, reference in enumerate(references, 1):
        slug = safe_slug(reference.label, index)
        pair_root = out_dir / "pairwise" / slug
        basic_dir = pair_root / "basic"
        detail_dir = pair_root / "detail"
        spatial_dir = pair_root / "spatial"
        basic_cmd = [
            sys.executable, str(scripts_dir / "analyze_vocals.py"),
            "--mix", str(target.path), "--reference", str(reference.path),
            "--out-dir", str(basic_dir), "--mix-label", target.label,
            "--reference-label", reference.label,
            "--segment-seconds", str(basic_seconds), "--target-sr", str(sample_rate),
        ]
        if font:
            basic_cmd += ["--font", font]
        if reference.separated:
            basic_cmd.append("--reference-is-separated")
        print(f"[{index}/{len(references)}] 基础诊断：{target.label} vs {reference.label}", flush=True)
        run_checked(basic_cmd, execution_log)
        basic = read_json(basic_dir / "metrics.json")

        detail_target = detail_reference = None
        if analysis_level in {"detail", "full"}:
            detail_cmd = [
                sys.executable, str(scripts_dir / "detail_diagnostics.py"),
                "--input", f"{target.label}={target.path}",
                "--input", f"{reference.label}={reference.path}",
                "--out-dir", str(detail_dir), "--sample-rate", str(sample_rate),
                "--segment-seconds", str(detail_seconds),
            ]
            print(f"[{index}/{len(references)}] 细节诊断：{target.label} vs {reference.label}", flush=True)
            run_checked(detail_cmd, execution_log)
            detail = read_json(detail_dir / "metrics.json")
            detail_target = record_by_label(detail["records"], target.label)
            detail_reference = record_by_label(detail["records"], reference.label)

        spatial_target = spatial_reference = None
        if analysis_level == "full":
            spatial_cmd = [
                sys.executable, str(scripts_dir / "spatial_diagnostics.py"),
                "--input", f"{target.label}={target.path}",
                "--input", f"{reference.label}={reference.path}",
                "--out-dir", str(spatial_dir), "--sample-rate", str(sample_rate),
                "--segment-seconds", str(spatial_seconds),
            ]
            if reference.separated:
                spatial_cmd += ["--separated-label", reference.label]
            print(f"[{index}/{len(references)}] 空间诊断：{target.label} vs {reference.label}", flush=True)
            run_checked(spatial_cmd, execution_log)
            spatial = read_json(spatial_dir / "spatial_metrics.json")
            spatial_target = spatial[target.label]
            spatial_reference = spatial[reference.label]

        pairwise[reference.label] = {
            "slug": slug,
            "reference": {"label": reference.label, "path": str(reference.path), "separated": reference.separated},
            "target": {
                "basic": basic[target.label], "detail": detail_target, "spatial": spatial_target,
            },
            "reference_metrics": {
                "basic": basic[reference.label], "detail": detail_reference, "spatial": spatial_reference,
            },
            "output": {
                "root": str(pair_root), "basic": str(basic_dir),
                "detail": str(detail_dir) if detail_target is not None else None,
                "spatial": str(spatial_dir) if spatial_target is not None else None,
            },
        }
    return pairwise, execution_log


def nested(record: dict[str, Any], path: tuple[str, ...]) -> float | None:
    value: Any = record
    for part in path:
        if value is None or not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def cohort_confidence(count: int) -> tuple[str, str]:
    if count == 1:
        return "证据不足", "只有一个参考，只能描述一对一差异"
    if count == 2:
        return "中低", "两个参考不足以稳定定义行业范围"
    if count <= 4:
        return "中", "参考数量有限且仍受选曲偏差影响"
    return "中高", "参考集仍不等于全部行业样本"


def summarize_feature(spec: FeatureSpec, target_record: dict[str, Any], reference_records: list[dict[str, Any]]) -> dict[str, Any] | None:
    target = nested(target_record, spec.path)
    refs = [nested(record, spec.path) for record in reference_records]
    refs = [value for value in refs if value is not None]
    if target is None or not refs:
        return None
    values = np.asarray(refs, dtype=float)
    q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
    iqr = float(q75 - q25)
    robust_scale = max(iqr / 1.349, spec.practical_floor)
    delta = float(target - median)
    signs = np.sign(target - values)
    positive = float(np.mean(signs > 0))
    negative = float(np.mean(signs < 0))
    agreement = max(positive, negative)
    direction = "高于" if delta > 0 else "低于" if delta < 0 else "接近"
    return {
        "label": spec.label,
        "unit": spec.unit,
        "group": spec.group,
        "confidence": spec.confidence,
        "limitation": spec.limitation,
        "practical_floor": spec.practical_floor,
        "target": target,
        "reference_count": len(refs),
        "reference_values": refs,
        "reference_min": float(np.min(values)),
        "reference_q25": float(q25),
        "reference_median": float(median),
        "reference_q75": float(q75),
        "reference_max": float(np.max(values)),
        "target_minus_reference_median": delta,
        "robust_deviation": delta / robust_scale,
        "direction": direction,
        "direction_agreement_fraction": agreement,
        "outside_reference_iqr": bool(target < q25 - spec.practical_floor or target > q75 + spec.practical_floor),
        "outside_reference_range": bool(target < np.min(values) - spec.practical_floor or target > np.max(values) + spec.practical_floor),
    }


def canonical_records(pairwise: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    first = next(iter(pairwise.values()))
    target_record = first["target"]
    references = [item["reference_metrics"] for item in pairwise.values()]
    labels = list(pairwise.keys())
    return target_record, references, labels


def reference_band_matrix(target_record: dict[str, Any], reference_records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    target_bands = target_record["basic"]["selected_segment_relative_band_energy_db"]
    target = np.asarray([float(target_bands[key]) for key in BANDS])
    refs = []
    for record in reference_records:
        bands = record["basic"]["selected_segment_relative_band_energy_db"]
        refs.append([float(bands[key]) for key in BANDS])
    return target, np.asarray(refs, dtype=float)


def feature_stats(features: dict[str, Any], keys: list[str]) -> list[dict[str, Any]]:
    return [features[key] for key in keys if key in features]


def style_axis(ax) -> None:
    ax.grid(True, alpha=0.22)
    for spine in ax.spines.values():
        spine.set_alpha(0.35)


def plot_interval_comparison(ax, stats: list[dict[str, Any]], title: str, level: str, limitation: str) -> None:
    labels = [item["label"] for item in stats]
    y = np.arange(len(stats))
    target = np.asarray([item["target"] for item in stats])
    median = np.asarray([item["reference_median"] for item in stats])
    low = np.asarray([
        item["reference_q25"] if item["reference_count"] >= 3 else item["reference_min"] for item in stats
    ])
    high = np.asarray([
        item["reference_q75"] if item["reference_count"] >= 3 else item["reference_max"] for item in stats
    ])
    ax.errorbar(median, y, xerr=np.vstack([median - low, high - median]), fmt="o", color="#4C78A8",
                capsize=5, label="参考中位数与区间")
    ax.scatter(target, y, marker="D", s=54, color="#E45756", label="待评估人声", zorder=3)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("dB / LUFS（按指标原单位）")
    ax.set_title(confidence_title(title, level, limitation), fontsize=12)
    ax.legend(fontsize=9)
    style_axis(ax)


def plot_charts(
    out_dir: Path,
    target_record: dict[str, Any],
    reference_records: list[dict[str, Any]],
    reference_labels: list[str],
    features: dict[str, Any],
) -> list[str]:
    count = len(reference_records)
    cohort_level, cohort_limit = cohort_confidence(count)
    charts: list[str] = []

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.8))
    level_stats = feature_stats(features, ["integrated_lufs", "rms_dbfs", "crest_factor_db", "active_range_db"])
    plot_interval_comparison(
        axes[0], level_stats, "R1A 绝对电平与动态轮廓", "高",
        "直接电平统计；不代表音质优劣",
    )
    target_bands, reference_bands = reference_band_matrix(target_record, reference_records)
    x = np.arange(len(BANDS))
    median = np.median(reference_bands, axis=0)
    if count >= 3:
        low, high = np.quantile(reference_bands, [0.25, 0.75], axis=0)
        interval_label = "参考四分位区间"
    else:
        low, high = np.min(reference_bands, axis=0), np.max(reference_bands, axis=0)
        interval_label = "参考最小—最大范围"
    axes[1].fill_between(x, low, high, color="#4C78A8", alpha=0.2, label=interval_label)
    axes[1].plot(x, median, "o-", color="#4C78A8", label="参考中位数")
    axes[1].plot(x, target_bands, "D-", color="#E45756", label="待评估人声")
    axes[1].set_xticks(x, BAND_LABELS, rotation=38, ha="right")
    axes[1].set_ylabel("相对频段能量（dB）")
    axes[1].set_title(confidence_title("R1B 参考音色包络", "中", "歌手、音域、元音和分离算法会改变频谱"), fontsize=12)
    axes[1].legend(fontsize=9)
    style_axis(axes[1])
    fig.tight_layout()
    name = "R1_levels_and_timbre.png"
    fig.savefig(out_dir / name, dpi=165, bbox_inches="tight")
    plt.close(fig)
    charts.append(name)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.8))
    dynamics_stats = feature_stats(features, ["crest_factor_db", "active_range_db", "macro_range_db", "micro_range_db"])
    plot_interval_comparison(
        axes[0], dynamics_stats, "R2A 动态参考区间", "中高",
        "统计可靠，但段落结构与演唱力度会影响",
    )
    texture_stats = feature_stats(features, ["presence_db", "air_db", "low_mid_masking_db", "harmonic_concentration", "upper_harmonic_share", "spectral_flatness"])
    y = np.arange(len(texture_stats))
    if texture_stats:
        z = np.asarray([item["robust_deviation"] for item in texture_stats])
        axes[1].axvspan(-1, 1, color="#4C78A8", alpha=0.12, label="接近参考中心")
        colors = ["#E45756" if abs(value) >= 1 else "#72B7B2" for value in z]
        axes[1].barh(y, z, color=colors)
        axes[1].axvline(0, color="black", linewidth=0.8)
        axes[1].set_yticks(y, [item["label"] for item in texture_stats])
        axes[1].invert_yaxis()
        axes[1].legend(fontsize=9)
        texture_level, texture_limit = "中", "代理指标不能唯一识别饱和、气声或插件"
    else:
        axes[1].text(0.5, 0.5, "当前分析级别未生成细节纹理指标", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_yticks([])
        texture_level, texture_limit = "证据不足", "需要 --analysis-level detail 或 full"
    axes[1].set_xlabel("相对参考中位数的稳健偏差（正=数值更高）")
    axes[1].set_title(confidence_title("R2B 音色与纹理偏差", texture_level, texture_limit), fontsize=12)
    style_axis(axes[1])
    fig.tight_layout()
    name = "R2_dynamics_and_texture.png"
    fig.savefig(out_dir / name, dpi=165, bbox_inches="tight")
    plt.close(fig)
    charts.append(name)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.8))
    stages = ["core", "body", "tails", "very_quiet"]
    stage_labels = ["核心", "主体", "尾部", "极弱"]
    target_stage = np.asarray([target_record["basic"]["width_by_stage"][stage] for stage in stages], dtype=float)
    reference_stage = np.asarray([
        [record["basic"]["width_by_stage"][stage] for stage in stages] for record in reference_records
    ], dtype=float)
    sx = np.arange(len(stages))
    smedian = np.median(reference_stage, axis=0)
    if count >= 3:
        slow, shigh = np.quantile(reference_stage, [0.25, 0.75], axis=0)
    else:
        slow, shigh = np.min(reference_stage, axis=0), np.max(reference_stage, axis=0)
    axes[0].fill_between(sx, slow, shigh, color="#4C78A8", alpha=0.2, label="参考区间")
    axes[0].plot(sx, smedian, "o-", color="#4C78A8", label="参考中位数")
    axes[0].plot(sx, target_stage, "D-", color="#E45756", label="待评估人声")
    axes[0].set_xticks(sx, stage_labels)
    axes[0].set_ylabel("Side/Mid（dB；越接近 0 通常越宽）")
    axes[0].set_title(confidence_title("R3A 不同电平阶段的宽度", "中", "宽度可测，但双轨、延迟、调制与分离残留并非唯一解"), fontsize=12)
    axes[0].legend(fontsize=9)
    style_axis(axes[0])
    correlations = [record["basic"]["stereo_full_file"]["correlation"] for record in reference_records]
    side_mid = [record["basic"]["stereo_full_file"]["side_to_mid_db"] for record in reference_records]
    corr_target = target_record["basic"]["stereo_full_file"]["correlation"]
    side_target = target_record["basic"]["stereo_full_file"]["side_to_mid_db"]
    px = np.arange(count)
    axes[1].scatter(correlations, side_mid, s=70, color="#4C78A8", label="参考")
    for xval, yval, label in zip(correlations, side_mid, reference_labels):
        axes[1].annotate(label, (xval, yval), xytext=(5, 4), textcoords="offset points", fontsize=8)
    axes[1].scatter([corr_target], [side_target], marker="D", s=85, color="#E45756", label="待评估人声")
    axes[1].set_xlabel("左右相关系数")
    axes[1].set_ylabel("全曲 Side/Mid（dB）")
    axes[1].set_title(confidence_title("R3B 声场位置分布", "中", "分离音轨可能产生虚假宽度和相位结构"), fontsize=12)
    axes[1].legend(fontsize=9)
    style_axis(axes[1])
    fig.tight_layout()
    name = "R3_spatial_reference_field.png"
    fig.savefig(out_dir / name, dpi=165, bbox_inches="tight")
    plt.close(fig)
    charts.append(name)

    fig, axes = plt.subplots(1, 2, figsize=(17, 7.6))
    consensus_stats = list(features.values())
    y = np.arange(len(consensus_stats))
    z = np.clip([item["robust_deviation"] for item in consensus_stats], -5, 5)
    axes[0].axvspan(-1, 1, color="#4C78A8", alpha=0.12)
    colors = ["#E45756" if abs(value) >= 1 else "#72B7B2" for value in z]
    axes[0].barh(y, z, color=colors)
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_yticks(y, [item["label"] for item in consensus_stats], fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("相对参考中位数的稳健偏差（截断于 ±5）")
    axes[0].set_title(confidence_title("R4A 待评估人声相对参考集的位置", cohort_level, cohort_limit), fontsize=12)
    style_axis(axes[0])

    usable = [item for item in consensus_stats if len(item["reference_values"]) == count]
    distances = []
    for ref_index in range(count):
        per_feature = []
        for item in usable:
            values = np.asarray(item["reference_values"], dtype=float)
            median = item["reference_median"]
            scale = max((item["reference_q75"] - item["reference_q25"]) / 1.349, item["practical_floor"])
            per_feature.append(min(abs(values[ref_index] - median) / scale, 5.0))
        distances.append(float(np.median(per_feature)) if per_feature else 0.0)
    outlier_level = cohort_level if count >= 3 else "证据不足"
    outlier_limit = cohort_limit if count >= 3 else "至少三个独立参考才能判断离群参考"
    if count >= 3:
        axes[1].barh(px, distances, color="#F2CF5B")
        axes[1].set_yticks(px, reference_labels)
        axes[1].invert_yaxis()
        axes[1].set_xlabel("相对参考集中心的综合距离（越高越像离群参考）")
    else:
        axes[1].text(0.5, 0.5, "参考少于三个，不计算离群度", ha="center", va="center",
                     transform=axes[1].transAxes, fontsize=13)
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        axes[1].set_xlabel("需要至少三个独立参考")
    axes[1].set_title(confidence_title("R4B 参考个性与离群度", outlier_level, outlier_limit), fontsize=12)
    style_axis(axes[1])
    fig.tight_layout()
    name = "R4_consensus_and_outliers.png"
    fig.savefig(out_dir / name, dpi=165, bbox_inches="tight")
    plt.close(fig)
    charts.append(name)
    return charts


def render_value(value: float, unit: str) -> str:
    if unit == "比例" or unit == "相关系数":
        return f"{value:.3f}"
    return f"{value:.2f} {unit}"


def write_report(
    out_dir: Path,
    target: AudioSpec,
    references: list[AudioSpec],
    features: dict[str, Any],
    pairwise: dict[str, Any],
    charts: list[str],
    duplicate_paths: list[str],
) -> None:
    level, limit = cohort_confidence(len(references))
    lines = [
        "# 人声参考集对比报告草稿",
        "",
        f"- 待评估人声：**{target.label}** — `{target.path}`",
        f"- 参考数量：**{len(references)}**",
        f"- 参考共性可信度：**{level}**（{limit}）",
        "- 重要说明：参考分离音轨是含处理和伪影的证据，不是原始录音室干声。",
        "",
        "## 参考输入",
        "",
        "| 标签 | 来源 | 是否标记为分离音轨 |",
        "|---|---|---|",
    ]
    for reference in references:
        lines.append(f"| {reference.label} | `{reference.path}` | {'是' if reference.separated else '否'} |")
    if duplicate_paths:
        lines += ["", f"> 警告：以下路径重复出现，不能视为独立行业参考：{', '.join(duplicate_paths)}"]

    lines += ["", "## 图表及可信度", ""]
    chart_notes = {
        "R1_levels_and_timbre.png": "R1A 高；R1B 中",
        "R2_dynamics_and_texture.png": (
            "R2A 中高；R2B 中" if "presence_db" in features
            else "R2A 中高；R2B 证据不足（当前分析级别未生成细节指标）"
        ),
        "R3_spatial_reference_field.png": "R3A 中；R3B 中",
        "R4_consensus_and_outliers.png": f"R4A {level}；R4B {level if len(references) >= 3 else '证据不足'}",
    }
    for chart in charts:
        lines.append(f"- `{chart}`：{chart_notes[chart]}（以图内逐子图限制为准）")

    common = [
        item for item in features.values()
        if item["reference_count"] >= 3
        and item["direction_agreement_fraction"] >= 0.75
        and item["outside_reference_iqr"]
        and abs(item["robust_deviation"]) >= 1.0
    ]
    common.sort(key=lambda item: abs(item["robust_deviation"]), reverse=True)
    lines += ["", "## 参考共性", ""]
    if len(references) < 3:
        lines.append("参考少于三个，不把任何差异称为“参考共性”；以下只能进入一对一结论。")
    elif common:
        for item in common[:8]:
            lines.append(
                f"- **{item['label']}**：待评估值 {render_value(item['target'], item['unit'])}，"
                f"{item['direction']}参考中位数 {render_value(item['reference_median'], item['unit'])}；"
                f"方向一致率 {item['direction_agreement_fraction']:.0%}，指标可信度 {item['confidence']}。"
            )
    else:
        lines.append("没有发现同时满足方向一致、超出参考四分位区间和实际差异门限的稳定共同偏差。")

    lines += ["", "## 最大测量差异（不等于处理因果）", ""]
    ranked = sorted(features.values(), key=lambda item: abs(item["robust_deviation"]), reverse=True)
    for item in ranked[:10]:
        lines.append(
            f"- {item['label']}：{item['direction']}参考中位数 "
            f"{abs(item['target_minus_reference_median']):.2f} {item['unit']}；"
            f"测量可信度 {item['confidence']}，限制：{item['limitation']}。"
        )

    lines += ["", "## 一对一深度报告", ""]
    for label, item in pairwise.items():
        root = Path(item["output"]["root"])
        relative = root.relative_to(out_dir).as_posix()
        lines.append(f"- **{target.label} vs {label}**：`{relative}/`（基础、细节及按分析级别生成的空间诊断）")

    lines += [
        "",
        "## 解释规则",
        "",
        "1. 一个参考只能支持一对一差异，不能定义行业标准。",
        "2. 三个以上相对独立参考才开始讨论共性；五个以上且风格匹配时可信度更高。",
        "3. 参考中位数描述中心，四分位区间描述常见范围；不要把所有参考简单平均成一条“正确声音”。",
        "4. 先根据一对一图判断差异来源，再把多参考共性转换为 EQ、动态、饱和或空间实验。",
        "5. 标为低或证据不足的图不能支持具体插件识别。",
    ]
    (out_dir / "reference-set-report.md").write_text("\n".join(lines), encoding="utf-8")


def validate_inputs(target: AudioSpec, references: list[AudioSpec]) -> list[str]:
    if not target.path.is_file():
        raise SystemExit(f"待评估人声不存在：{target.path}")
    if not references:
        raise SystemExit("至少需要一个 --reference")
    labels = [target.label] + [item.label for item in references]
    if len(labels) != len(set(labels)):
        raise SystemExit("待评估人声和所有参考必须使用互不重复的标签")
    for reference in references:
        if not reference.path.is_file():
            raise SystemExit(f"参考人声不存在：{reference.path}")
    target_path = str(target.path.resolve()).casefold()
    original_paths = [str(item.path.resolve()) for item in references]
    normalized = [path.casefold() for path in original_paths]
    if target_path in normalized:
        raise SystemExit("待评估人声不能同时作为参考输入")
    duplicate_paths = sorted({
        original_paths[index] for index, path in enumerate(normalized) if normalized.count(path) > 1
    })
    if duplicate_paths:
        raise SystemExit(
            "同一参考文件不能重复计权；请移除重复输入：" + ", ".join(duplicate_paths)
        )
    return duplicate_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, type=parse_labeled_audio, help="标签=待评估人声路径")
    parser.add_argument("--reference", action="append", required=True, type=parse_labeled_audio,
                        help="可重复传入：标签=参考人声路径")
    parser.add_argument("--separated-reference", action="append", default=[],
                        help="标记对应参考标签为人声分离音轨，可重复传入")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--analysis-level", choices=["basic", "detail", "full"], default="full",
                        help="basic=基础；detail=基础+细节；full=基础+细节+高级空间")
    parser.add_argument("--basic-segment-seconds", type=float, default=12.0)
    parser.add_argument("--detail-segment-seconds", type=float, default=35.0)
    parser.add_argument("--spatial-segment-seconds", type=float, default=20.0)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--font")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    separated = set(args.separated_reference)
    reference_labels = {label for label, _ in args.reference}
    unknown_separated = separated - reference_labels
    if unknown_separated:
        raise SystemExit("--separated-reference 标签不存在于 --reference：" + ", ".join(sorted(unknown_separated)))
    target = AudioSpec(args.target[0], args.target[1].resolve())
    references = [AudioSpec(label, path.resolve(), label in separated) for label, path in args.reference]
    duplicate_paths = validate_inputs(target, references)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    font_path = setup_font(args.font)
    scripts_dir = Path(__file__).resolve().parent
    pairwise, execution_log = build_pairwise(
        target, references, args.out_dir, scripts_dir, args.analysis_level,
        args.basic_segment_seconds, args.detail_segment_seconds,
        args.spatial_segment_seconds, args.sample_rate, font_path,
    )
    target_record, reference_records, labels = canonical_records(pairwise)
    summaries: dict[str, Any] = {}
    for spec in FEATURES:
        summary = summarize_feature(spec, target_record, reference_records)
        if summary is not None:
            summaries[spec.key] = summary
    charts = plot_charts(args.out_dir, target_record, reference_records, labels, summaries)
    write_report(args.out_dir, target, references, summaries, pairwise, charts, duplicate_paths)
    public_pairwise = {
        label: {
            "reference": item["reference"],
            "output": item["output"],
        }
        for label, item in pairwise.items()
    }
    payload = {
        "plugin_id": PLUGIN_ID,
        "analysis_mode": ANALYSIS_MODE,
        "schema": "vocal-reference-set-v1",
        "analysis_level": args.analysis_level,
        "target": {"label": target.label, "path": str(target.path)},
        "references": [
            {"label": item.label, "path": str(item.path), "separated": item.separated}
            for item in references
        ],
        "cohort": {
            "count": len(references),
            "confidence": cohort_confidence(len(references))[0],
            "limitation": cohort_confidence(len(references))[1],
            "duplicate_reference_paths": duplicate_paths,
        },
        "features": summaries,
        "pairwise": public_pairwise,
        "charts": charts,
        "report": "reference-set-report.md",
    }
    (args.out_dir / "reference-set-metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out_dir / "execution-log.json").write_text(
        json.dumps(execution_log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "plugin_id": PLUGIN_ID,
        "analysis_mode": ANALYSIS_MODE,
        "metrics": str(args.out_dir / "reference-set-metrics.json"),
        "report": str(args.out_dir / "reference-set-report.md"),
        "charts": [str(args.out_dir / name) for name in charts],
        "pairwise_root": str(args.out_dir / "pairwise"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
