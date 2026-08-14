#!/usr/bin/env python3
"""对比两份人声，生成测量数据、PNG 图表和中文报告草稿。

这是比较诊断工具，不是插件识别器；它不会修改源音频。人声分离参考必须结合人工判断。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from pathlib import Path


def _import_dependencies():
    """集中导入第三方库，并在缺失依赖时给出可直接执行的安装指引。"""
    try:
        import numpy as np
        import soundfile as sf
        from scipy import ndimage, signal
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pyloudnorm as pyln
        import librosa
    except ImportError as exc:
        script_dir = Path(__file__).resolve().parent
        raise SystemExit(
            f"缺少依赖：{exc.name}。请运行：\n"
            f'  "{sys.executable}" "{script_dir / "bootstrap_deps.py"}" '
            f'--target <task-local-deps-dir>\n'
            "然后把该目录加入 PYTHONPATH 并重试。"
        ) from exc
    return np, sf, signal, ndimage, plt, pyln, librosa


np, sf, signal, ndimage, plt, pyln, librosa = _import_dependencies()

BANDS = [
    (20, 80), (80, 150), (150, 300), (300, 600), (600, 1200),
    (1200, 2500), (2500, 5000), (5000, 8000), (8000, 12000),
    (12000, 18000),
]
# 统一使用宽频段做统计，避免把不同歌手的细小谱峰误判成可照抄的 EQ 动作。


def safe_db(value: float, floor: float = -160.0) -> float:
    """把线性幅度转换为 dB，并用 floor 防止 log(0) 和负无穷。"""
    return max(floor, 20.0 * math.log10(max(float(value), 10 ** (floor / 20))))


def finite(value):
    """把 NaN/Inf 转成 JSON 可安全序列化的 None。"""
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def find_cjk_font(explicit: str | None) -> str | None:
    """优先使用显式字体，否则按操作系统寻找常见中文字体。"""
    candidates = [explicit] if explicit else []
    if platform.system() == "Windows":
        candidates += [
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\simhei.ttf",
            r"C:\Windows\Fonts\simsun.ttc",
        ]
    elif platform.system() == "Darwin":
        candidates += ["/System/Library/Fonts/PingFang.ttc"]
    else:
        candidates += [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def font_properties(font_path: str | None, size=None, weight=None):
    """创建 Matplotlib 字体配置；找不到中文字体时退回系统默认字体。"""
    from matplotlib.font_manager import FontProperties
    kwargs = {"size": size, "weight": weight}
    return FontProperties(fname=font_path, **kwargs) if font_path else FontProperties(**kwargs)


def load_audio(path: Path, target_sr: int = 44100):
    """读取音频并重采样到统一分析采样率，同时保留源文件格式和原始峰值。"""
    try:
        data, sr = sf.read(path, always_2d=True, dtype="float32")
        info = sf.info(path)
    except Exception as exc:
        raise SystemExit(
            f"无法解码 {path}：{exc}\n"
            "请用 ffmpeg 把不受支持的 AAC/M4A 转成 WAV 后重试。"
        ) from exc
    if data.size == 0:
        raise SystemExit(f"音频为空：{path}")
    original_sr = sr
    # 峰值必须在重采样前读取；重采样滤波可能产生并非源文件真实存在的过冲。
    original_sample_peak = float(np.max(np.abs(data)))
    if sr != target_sr:
        # 用最大公约数化简 resample_poly 的升/降采样比例，减少不必要计算。
        common = math.gcd(sr, target_sr)
        data = signal.resample_poly(data, target_sr // common, sr // common, axis=0)
        sr = target_sr
    return data, sr, info, original_sr, original_sample_peak


def moving_rms(mono, sr, win_ms=50, hop_ms=10):
    """按滑动窗口计算 RMS 包络；默认 50 ms 窗口、10 ms 步长。"""
    win = max(1, int(sr * win_ms / 1000))
    hop = max(1, int(sr * hop_ms / 1000))
    if len(mono) < win:
        return np.array([np.sqrt(np.mean(mono * mono) + 1e-20)])
    squared = np.convolve(mono * mono, np.ones(win) / win, mode="valid")
    return np.sqrt(squared[::hop] + 1e-20)


def strongest_window(mono, sr, seconds):
    """寻找平均能量最高的连续片段，避免静音比例干扰两首未对齐歌曲的比较。"""
    requested = max(2.0, min(seconds, len(mono) / sr))
    hop = max(1, sr // 10)
    block = max(1, sr // 2)
    if len(mono) <= int(requested * sr):
        return 0, len(mono)
    # 先以 0.5 秒块得到较稳定的局部 RMS，再对目标时长做滑动平均。
    env = np.array([
        np.sqrt(np.mean(mono[i:i + block] ** 2) + 1e-20)
        for i in range(0, len(mono) - block, hop)
    ])
    span = max(1, int(requested * sr / hop))
    smooth = np.convolve(env, np.ones(span) / span, mode="valid")
    start = int(np.argmax(smooth)) * hop
    return start, min(len(mono), start + int(requested * sr))


def welch_spectrum(mono, sr):
    """用 Welch PSD 估计长期功率谱，降低单个 FFT 对瞬时音素的敏感度。"""
    n = min(8192, len(mono))
    freqs, power = signal.welch(mono, sr, nperseg=n, noverlap=n // 2)
    return freqs, power


def band_energy_db(freqs, power):
    """计算每个宽频段占 20 Hz–18 kHz 总能量的相对 dB。"""
    usable = (freqs >= 20) & (freqs <= min(18000, freqs[-1]))
    total = np.sum(power[usable]) + 1e-30
    result = {}
    for lo, hi in BANDS:
        hi = min(hi, freqs[-1])
        idx = (freqs >= lo) & (freqs < hi)
        if idx.any():
            result[f"{lo}-{hi}"] = 10 * np.log10(np.sum(power[idx]) / total + 1e-20)
    return result


def format_band_label(key: str) -> str:
    """把内部频段键转换成适合图表显示的紧凑标签，如 1200→1.2k。"""
    lo_text, hi_text = key.split("-", 1)
    def compact(text: str) -> str:
        """把四位数 Hz 值缩写成 kHz 标签。"""
        value = int(float(text))
        return f"{value / 1000:g}k" if value >= 1000 else str(value)
    return f"{compact(lo_text)}–{compact(hi_text)}"


def smooth_relative_spectrum(mono, sr):
    """生成近似 1/12 octave 平滑频谱，并在 800–1200 Hz 区间归一化。"""
    freqs, power = welch_spectrum(mono, sr)
    values = 10 * np.log10(power + 1e-30)
    grid = np.geomspace(50, min(18000, sr / 2 - 1), 600)
    smoothed = []
    # 在对数频率轴上做恒定倍频程宽度平滑，视觉上比固定 Hz 平滑更符合听觉尺度。
    for center in grid:
        lo, hi = center / 2 ** (1 / 24), center * 2 ** (1 / 24)
        idx = (freqs >= lo) & (freqs <= hi)
        smoothed.append(np.mean(values[idx]) if idx.any() else np.interp(center, freqs, values))
    smoothed = np.asarray(smoothed)
    # 只比较音色形状，不让两份文件的绝对响度差主导曲线高度。
    norm = (grid >= 800) & (grid <= 1200)
    smoothed -= np.mean(smoothed[norm])
    return grid, smoothed


def stereo_metrics(data):
    """计算全文件左右相关度，以及 Side/Mid 的整体能量比。"""
    if data.shape[1] < 2:
        return {"correlation": 1.0, "side_to_mid_db": -160.0}
    left, right = data[:, 0].astype(float), data[:, 1].astype(float)
    correlation = np.corrcoef(left, right)[0, 1]
    # Mid 表示左右共同成分，Side 表示左右差异；Side/Mid 越接近 0 dB 通常越宽。
    mid, side = (left + right) / 2, (left - right) / 2
    ratio = safe_db((np.sqrt(np.mean(side * side)) + 1e-20) /
                    (np.sqrt(np.mean(mid * mid)) + 1e-20))
    return {"correlation": finite(float(correlation)), "side_to_mid_db": ratio}


def width_by_stage(data, sr):
    """按主唱核心、主体、句尾和极弱部分统计 Side/Mid，观察宽度是否随音量展开。"""
    if data.shape[1] < 2:
        return {k: -160.0 for k in ["core", "body", "tails", "very_quiet"]}
    left, right = data[:, 0], data[:, 1]
    mid, side = (left + right) / 2, (left - right) / 2
    # 去掉低频振动和极高频噪声，让阶段判断聚焦于主要人声频段。
    sos = signal.butter(3, [200, min(12000, sr / 2 - 10)], btype="bandpass", fs=sr, output="sos")
    mid, side = signal.sosfiltfilt(sos, mid), signal.sosfiltfilt(sos, side)
    mid_rms = moving_rms(mid, sr)
    side_rms = moving_rms(side, sr)
    mid_db = 20 * np.log10(mid_rms + 1e-20)
    # 用 99 分位而非绝对最大值作为参照，避免单个爆破音把全部阶段阈值抬高。
    top = np.percentile(mid_db, 99)
    ratio = 20 * np.log10((side_rms + 1e-15) / (mid_rms + 1e-15))
    definitions = {"core": (-6, 0.01), "body": (-15, -6), "tails": (-30, -15), "very_quiet": (-45, -30)}
    output = {}
    for key, (lo, hi) in definitions.items():
        mask = (mid_db - top >= lo) & (mid_db - top < hi)
        output[key] = finite(float(np.median(ratio[mask]))) if mask.any() else None
    return output


def stereo_by_band(data, sr):
    """逐频段计算 Side/Mid，区分全频宽化与只发生在高频效果中的宽化。"""
    if data.shape[1] < 2:
        return {f"{lo}-{hi}": -160.0 for lo, hi in BANDS if lo >= 80}
    left, right = data[:, 0].astype(float), data[:, 1].astype(float)
    mid, side = (left + right) / 2, (left - right) / 2
    output = {}
    for lo, hi in BANDS:
        if lo < 80 or lo >= sr / 2 - 20:
            continue
        hi = min(hi, sr / 2 - 20)
        sos = signal.butter(4, [lo, hi], btype="bandpass", fs=sr, output="sos")
        bm, bs = signal.sosfiltfilt(sos, mid), signal.sosfiltfilt(sos, side)
        output[f"{lo}-{hi}"] = safe_db(
            (np.sqrt(np.mean(bs * bs)) + 1e-15) /
            (np.sqrt(np.mean(bm * bm)) + 1e-15)
        )
    return output


def harmonic_concentration(mono, sr):
    """估计 5–10 kHz 能量中靠近基频整数倍谐波的比例，作为颗粒结构比较指标。"""
    # 目标最高只分析到 10 kHz，因此降到 22.05 kHz 可显著减少 pYIN/STFT 计算量。
    analysis_sr = min(sr, 22050)
    if analysis_sr != sr:
        common = math.gcd(sr, analysis_sr)
        mono = signal.resample_poly(mono, analysis_sr // common, sr // common)
        sr = analysis_sr
    f0, _, voiced_prob = librosa.pyin(
        mono, fmin=65, fmax=600, sr=sr, frame_length=2048, hop_length=256
    )
    spectrum = np.abs(librosa.stft(mono, n_fft=4096, hop_length=256, win_length=2048)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
    ratios = []
    # 只保留置信度较高的有声音框，避免把气声、齿音和分离噪声当作基频。
    valid = np.where(np.isfinite(f0) & (voiced_prob > 0.75))[0]
    for index in valid:
        if index >= spectrum.shape[1]:
            continue
        power = spectrum[:, index]
        high = (freqs >= 5000) & (freqs <= min(10000, sr / 2 - 20))
        denominator = np.sum(power[high]) + 1e-30
        mask = np.zeros_like(freqs, dtype=bool)
        harmonic = 1
        # 围绕每个整数倍谐波建立窄掩码；宽度随频率略增以容纳谱泄漏和轻微颤音。
        while harmonic * f0[index] <= min(10000, sr / 2 - 20):
            target = harmonic * f0[index]
            width = max(18, target * 0.004)
            mask |= np.abs(freqs - target) <= width
            harmonic += 1
        ratios.append(np.sum(power[mask & high]) / denominator)
    return {
        "voiced_frames": int(len(valid)),
        "high_band_harmonic_fraction_median": finite(float(np.median(ratios))) if ratios else None,
    }


def analyze_file(label, path, segment_seconds, target_sr):
    """完成单个文件的读取、片段选择、响度、频谱、空间和谐波测量。"""
    data, sr, info, original_sr, original_sample_peak = load_audio(path, target_sr)
    mono = np.mean(data, axis=1).astype(float)
    start, end = strongest_window(mono, sr, segment_seconds)
    segment = data[start:end]
    segment_mono = np.mean(segment, axis=1).astype(float)
    peak = original_sample_peak
    rms = float(np.sqrt(np.mean(mono * mono) + 1e-30))
    meter = pyln.Meter(sr)
    lufs = float(meter.integrated_loudness(data if data.shape[1] > 1 else mono))
    active = moving_rms(mono, sr)
    # 排除低于 -45 dBFS 的安静帧，减少前后空白对人声动态分布的影响。
    active_db = 20 * np.log10(active[active > 10 ** (-45 / 20)] + 1e-20)
    freqs, power = welch_spectrum(mono, sr)
    centroid = float(np.sum(freqs * power) / (np.sum(power) + 1e-30))
    return {
        "label": label,
        "path": str(path.resolve()),
        "format": info.format,
        "subtype": info.subtype,
        "sample_rate_original": original_sr,
        "analysis_sample_rate": sr,
        "channels": int(data.shape[1]),
        "duration_seconds": float(len(data) / sr),
        "sample_peak_dbfs": safe_db(peak),
        "integrated_lufs": finite(lufs),
        "rms_dbfs": safe_db(rms),
        "crest_factor_db": safe_db(peak) - safe_db(rms),
        "active_50ms_rms_db": {
            "p10": finite(float(np.percentile(active_db, 10))),
            "median": finite(float(np.median(active_db))),
            "p90": finite(float(np.percentile(active_db, 90))),
            "p90_p10_range": finite(float(np.percentile(active_db, 90) - np.percentile(active_db, 10))),
        },
        "spectral_centroid_hz": centroid,
        "relative_band_energy_db": band_energy_db(freqs, power),
        "selected_segment_relative_band_energy_db": band_energy_db(*welch_spectrum(segment_mono, sr)),
        "stereo_full_file": stereo_metrics(data),
        "selected_segment": {"start_seconds": start / sr, "duration_seconds": (end - start) / sr},
        "width_by_stage": width_by_stage(data, sr),
        "selected_segment_stereo_by_band": stereo_by_band(segment, sr),
        "selected_segment_harmonics": harmonic_concentration(segment_mono, sr),
        # 下划线字段只供后续绘图使用，public_metrics() 会在写 JSON 前移除。
        "_audio": data,
        "_mono": mono,
        "_segment": segment,
        "_segment_mono": segment_mono,
        "_sr": sr,
    }


def public_metrics(record):
    """移除体积较大的内部音频数组，只保留可写入 metrics.json 的公开指标。"""
    return {key: value for key, value in record.items() if not key.startswith("_")}


def set_style(ax):
    """统一所有图表的深色主题、网格和坐标轴样式。"""
    ax.set_facecolor("#111820")
    ax.grid(True, alpha=0.18, color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#63707e")


def confidence_title(title, level, limitation):
    """给每个子图单独标注当前 A/B 解释的可信度和主要限制。"""
    return f"{title}\n可信度：{level}｜{limitation}"


def make_charts(records, out_dir, font_path):
    """根据两份分析记录生成四张中文 PNG 诊断图。"""
    fp = lambda size=None, weight=None: font_properties(font_path, size, weight)
    labels = list(records)
    colors = {labels[0]: "#2388ff", labels[1]: "#ff762e"}

    # 图 01：上半部分比较归一化音色形状，下半部分显示“参考 - 用户”的频段差值。
    fig, axes = plt.subplots(2, 1, figsize=(12, 11), facecolor="#0c1117", constrained_layout=True)
    ax = axes[0]
    set_style(ax)
    spectra = {}
    for label, record in records.items():
        grid, values = smooth_relative_spectrum(record["_segment_mono"], record["_sr"])
        spectra[label] = (grid, values)
        ax.semilogx(grid, values, lw=2.2, color=colors[label], label=label)
    ax.axvspan(150, 300, color=colors[labels[0]], alpha=0.10)
    ax.axvspan(8000, 14000, color=colors[labels[1]], alpha=0.10)
    ax.set_xlim(50, 18000)
    ax.set_ylim(-40, 22)
    ax.set_title(confidence_title("图 1A｜高能量片段长期频谱（1 kHz 附近归一化）", "中", "歌手、音域与音素影响"), color="white", fontproperties=fp(16, "bold"))
    ax.set_xlabel("频率（Hz）", color="white", fontproperties=fp(12))
    ax.set_ylabel("相对电平（dB）", color="white", fontproperties=fp(12))
    ax.legend(prop=fp(12), facecolor="#111820", labelcolor="white")

    ax = axes[1]
    set_style(ax)
    band_labels, differences = [], []
    user_bands = records[labels[0]]["selected_segment_relative_band_energy_db"]
    ref_bands = records[labels[1]]["selected_segment_relative_band_energy_db"]
    for key in user_bands:
        if key in ref_bands and not key.startswith("20-"):
            band_labels.append(format_band_label(key))
            differences.append(ref_bands[key] - user_bands[key])
    x = np.arange(len(band_labels))
    bar_colors = [colors[labels[0]] if value < 0 else colors[labels[1]] for value in differences]
    bars = ax.bar(x, differences, color=bar_colors, alpha=0.9)
    ax.axhline(0, color="white", lw=1)
    ax.set_xticks(x, band_labels, fontproperties=fp(9))
    low = min(-3, min(differences) - 1.5)
    high = max(3, max(differences) + 1.5)
    ax.set_ylim(low, high)
    ax.set_title(confidence_title(f"图 1B｜各频段差值：{labels[1]} − {labels[0]}", "中", "只比较归一化音色形状，不是 EQ 处方"), color="white", fontproperties=fp(16, "bold"))
    ax.set_xlabel("频段（Hz）", color="white", fontproperties=fp(12))
    ax.set_ylabel("差值（dB）", color="white", fontproperties=fp(12))
    for bar, value in zip(bars, differences):
        ax.text(bar.get_x() + bar.get_width() / 2, value + (0.2 if value >= 0 else -0.4), f"{value:+.1f}",
                ha="center", va="bottom" if value >= 0 else "top", color="white", fontproperties=fp(10))
    fig.savefig(out_dir / "01_spectrum_and_bands.png", dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)

    # 图 02：两份声谱图共用相同频率范围和颜色动态，避免视觉尺度不同造成误判。
    specs = []
    for record in records.values():
        f, t, magnitude = signal.spectrogram(record["_segment_mono"], record["_sr"], window="hann", nperseg=2048, noverlap=1792, mode="magnitude")
        values = 20 * np.log10(magnitude + 1e-10)
        specs.append((f, t, values - np.max(values)))
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), facecolor="#0c1117", constrained_layout=True)
    for ax, label, record, (freqs, times, values) in zip(axes, labels, records.values(), specs):
        keep = freqs <= 16000
        ax.pcolormesh(times, freqs[keep] / 1000, values[keep], shading="auto", cmap="magma", vmin=-65, vmax=-8)
        ax.set_ylim(0, 16)
        ax.set_facecolor("#0c1117")
        ax.tick_params(colors="white")
        start = record["selected_segment"]["start_seconds"]
        ax.set_title(confidence_title(f"{label}｜片段从原文件约 {start:.1f}s 开始", "中", "纹理可见，但处理与分离伪影成因重叠"), color="white", fontproperties=fp(14, "bold"))
        ax.set_xlabel("片段内时间（秒）", color="white", fontproperties=fp(12))
        ax.set_ylabel("频率（kHz）", color="white", fontproperties=fp(12))
    fig.suptitle("图 2｜时间—频率声谱图：横线为谐波，竖带多为辅音/瞬态", color="white", fontproperties=fp(20, "bold"))
    fig.savefig(out_dir / "02_time_frequency_spectrogram.png", dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)

    # 图 03：同时呈现动态分布、核心/句尾宽度变化和高频谐波集中指标。
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.5), facecolor="#0c1117", constrained_layout=True)
    ax = axes[0]
    set_style(ax)
    for label, record in records.items():
        active = moving_rms(record["_mono"], record["_sr"])
        values = 20 * np.log10(active + 1e-20)
        values = values[values > -45]
        ax.hist(values, bins=np.arange(-45, -5, 0.5), density=True, histtype="step", lw=2.2, color=colors[label], label=label)
    ax.set_title(confidence_title("3A｜50 ms 活跃电平分布", "高", "直接短时电平统计"), color="white", fontproperties=fp(13, "bold"))
    ax.set_xlabel("RMS（dBFS）", color="white", fontproperties=fp(11))
    ax.set_ylabel("概率密度", color="white", fontproperties=fp(11))
    ax.legend(prop=fp(9), facecolor="#111820", labelcolor="white")

    ax = axes[1]
    set_style(ax)
    stage_keys = ["core", "body", "tails", "very_quiet"]
    stage_labels = ["核心", "主体", "句尾", "极弱部分"]
    for label, record in records.items():
        values = [record["width_by_stage"].get(key) for key in stage_keys]
        ax.plot(range(4), values, "o-", lw=2.2, color=colors[label], label=label)
    ax.set_xticks(range(4), stage_labels, fontproperties=fp(10))
    ax.set_ylim(-26, 3)
    ax.axhline(0, color="white", lw=1, alpha=0.5)
    ax.set_title(confidence_title("3B｜空间宽度随音量变化", "中", "分离残留会夸大弱段 Side"), color="white", fontproperties=fp(13, "bold"))
    ax.set_xlabel("人声阶段", color="white", fontproperties=fp(11))
    ax.set_ylabel("Side / Mid（dB，越高越宽）", color="white", fontproperties=fp(11))
    ax.legend(prop=fp(9), facecolor="#111820", labelcolor="white")

    ax = axes[2]
    set_style(ax)
    harmonic_values = []
    for label in labels:
        value = records[label]["selected_segment_harmonics"]["high_band_harmonic_fraction_median"]
        harmonic_values.append(0 if value is None else 100 * value)
    bars = ax.bar(labels, harmonic_values, color=[colors[label] for label in labels])
    ax.set_ylim(0, 100)
    ax.set_title(confidence_title("3C｜5–10 kHz 谐波集中指标", "中低", "音高跟踪、颤音与分离影响"), color="white", fontproperties=fp(13, "bold"))
    ax.set_ylabel("靠近整数倍谐波的能量（%）", color="white", fontproperties=fp(11))
    ax.set_xticks(range(2), labels, fontproperties=fp(10))
    for bar, value in zip(bars, harmonic_values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2.5, f"{value:.0f}%", ha="center", color="white", fontproperties=fp(13, "bold"))
    fig.suptitle("图 3｜动态、空间和高频谐波结构", color="white", fontproperties=fp(20, "bold"))
    fig.savefig(out_dir / "03_dynamics_space_grain.png", dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)

    # 图 04：分频段 Side/Mid 对比，用于定位宽化主要发生在哪些频率。
    fig, ax = plt.subplots(figsize=(12, 6.5), facecolor="#0c1117", constrained_layout=True)
    set_style(ax)
    width = 0.36
    common = [key for key in records[labels[0]]["selected_segment_stereo_by_band"] if key in records[labels[1]]["selected_segment_stereo_by_band"]]
    x = np.arange(len(common))
    for index, label in enumerate(labels):
        values = [records[label]["selected_segment_stereo_by_band"][key] for key in common]
        ax.bar(x + (index - 0.5) * width, values, width, color=colors[label], label=label)
    ax.axhline(0, color="white", lw=1)
    ax.set_xticks(x, common, fontproperties=fp(9))
    ax.set_title(confidence_title("图 4｜所选片段各频段立体声宽度", "中", "宽度可测，来源类别不唯一"), color="white", fontproperties=fp(16, "bold"))
    ax.set_xlabel("频段（Hz）", color="white", fontproperties=fp(12))
    ax.set_ylabel("Side / Mid（dB，越接近 0 越宽）", color="white", fontproperties=fp(12))
    ax.legend(prop=fp(11), facecolor="#111820", labelcolor="white")
    fig.savefig(out_dir / "04_stereo_by_band.png", dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def draft_report(records, out_dir, reference_is_separated):
    """把关键测量整理成中文 Markdown 草稿，供最终试听判断和按图讲解继续完善。"""
    labels = list(records)
    user, ref = records[labels[0]], records[labels[1]]
    band_diffs = []
    for key, value in user["selected_segment_relative_band_energy_db"].items():
        if key in ref["selected_segment_relative_band_energy_db"]:
            band_diffs.append((key, ref["selected_segment_relative_band_energy_db"][key] - value))
    # 按绝对差值排序，只在草稿中优先展示最需要人工判断的频段。
    band_diffs.sort(key=lambda item: abs(item[1]), reverse=True)
    top_bands = "、".join(f"{key} Hz {value:+.1f} dB" for key, value in band_diffs[:4])
    user_h = user["selected_segment_harmonics"]["high_band_harmonic_fraction_median"]
    ref_h = ref["selected_segment_harmonics"]["high_band_harmonic_fraction_median"]
    caveat = "参考被标记为人声分离文件，必须把分离伪影与真实混音质感分开。" if reference_is_separated else "参考未被标记为分离文件，仍需确认它是干声、处理后独唱还是完整声部 stem。"
    text = f"""# 人声混音对比分析草稿

> 本文件由测量脚本自动生成。最终报告必须结合试听和图片检查，并把结论标为“测量 / 强推断 / 假设”。

## 输入有效性

- {labels[0]}：`{user['path']}`
- {labels[1]}：`{ref['path']}`
- 注意：{caveat}

## 核心测量

| 指标 | {labels[0]} | {labels[1]} |
|---|---:|---:|
| 综合响度 | {user['integrated_lufs']:.2f} LUFS | {ref['integrated_lufs']:.2f} LUFS |
| 样本峰值 | {user['sample_peak_dbfs']:.2f} dBFS | {ref['sample_peak_dbfs']:.2f} dBFS |
| Crest factor | {user['crest_factor_db']:.2f} dB | {ref['crest_factor_db']:.2f} dB |
| 活跃 50 ms 中位 RMS | {user['active_50ms_rms_db']['median']:.2f} dBFS | {ref['active_50ms_rms_db']['median']:.2f} dBFS |
| 立体声相关度 | {user['stereo_full_file']['correlation']:.3f} | {ref['stereo_full_file']['correlation']:.3f} |
| 频谱重心 | {user['spectral_centroid_hz']:.0f} Hz | {ref['spectral_centroid_hz']:.0f} Hz |
| 5–10 kHz 谐波集中指标 | {(user_h or 0)*100:.1f}% | {(ref_h or 0)*100:.1f}% |

最大相对频段差异（参考 − 用户）：{top_bands}。

## 图片

1. `01_spectrum_and_bands.png`：1A、1B 均为中可信；解释低中频遮蔽、Presence、空气区和为什么不能照抄 EQ 差值。
2. `02_time_frequency_spectrogram.png`：每个输入面板均为中可信；解释谐波横线、辅音竖带、上方雾状纹理与分离伪影。
3. `03_dynamics_space_grain.png`：3A 高、3B 中、3C 中低；解释是否需要更多压缩、核心/句尾宽度，以及颗粒指标的限制。
4. `04_stereo_by_band.png`：中可信；解释宽度集中在哪些频段，但不从宽度直接命名处理器。

## 最终报告待补

- [ ] 试听后写一句核心结论。
- [ ] 区分测量、强推断和假设。
- [ ] 给出 3–5 个优先操作及起始参数。
- [ ] 说明每一步的 A/B 方法、停止条件和失败征兆。
- [ ] 给出串联 / 并联 / 发送结构。
- [ ] 说明歌手、话筒、编曲、分离和编码限制。
"""
    (out_dir / "analysis-draft.md").write_text(text, encoding="utf-8")


def parse_args():
    """定义命令行参数；文件路径和输出目录必须由调用者明确提供。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mix", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--mix-label", default="你的混音")
    parser.add_argument("--reference-label", default="参考人声")
    parser.add_argument("--segment-seconds", type=float, default=12.0)
    parser.add_argument("--target-sr", type=int, default=44100)
    parser.add_argument("--font")
    parser.add_argument("--reference-is-separated", action="store_true")
    return parser.parse_args()


def main() -> int:
    """串联完整流程：校验输入、测量、写 JSON、出图、生成报告草稿。"""
    args = parse_args()
    if not args.mix.exists() or not args.reference.exists():
        raise SystemExit("--mix 和 --reference 都必须指向真实存在的音频文件。")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    font_path = find_cjk_font(args.font)
    records = {
        args.mix_label: analyze_file(args.mix_label, args.mix, args.segment_seconds, args.target_sr),
        args.reference_label: analyze_file(args.reference_label, args.reference, args.segment_seconds, args.target_sr),
    }
    metrics = {label: public_metrics(record) for label, record in records.items()}
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    make_charts(records, args.out_dir, font_path)
    draft_report(records, args.out_dir, args.reference_is_separated)
    manifest = {
        "metrics": "metrics.json",
        "draft_report": "analysis-draft.md",
        "charts": [
            "01_spectrum_and_bands.png",
            "02_time_frequency_spectrogram.png",
            "03_dynamics_space_grain.png",
            "04_stereo_by_band.png",
        ],
        "font": font_path,
    }
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
