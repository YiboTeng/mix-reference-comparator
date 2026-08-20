from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy import signal

EPS = 1e-12
TARGET_SR = 24000
BANDS = [
    (20, 80, "Sub 20–80"),
    (80, 150, "Bass 80–150"),
    (150, 300, "Low-mid 150–300"),
    (300, 600, "Body 300–600"),
    (600, 1200, "Mid 600–1.2k"),
    (1200, 2500, "Presence 1.2–2.5k"),
    (2500, 5000, "Clarity 2.5–5k"),
    (5000, 8000, "Harsh/Hat 5–8k"),
    (8000, 12000, "Air 8–12k"),
]
PALETTE = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#17becf"]
CONFIDENCE_ORDER = ["证据不足", "低", "中低", "中", "中高", "高"]


def configure_cjk_font() -> str:
    """显式注册中文字体，避免可信度行在 PNG 中显示为方框。"""
    candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]
    for path in candidates:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            name = font_manager.FontProperties(fname=str(path)).get_name()
            matplotlib.rcParams["font.family"] = name
            matplotlib.rcParams["axes.unicode_minus"] = False
            return name
    matplotlib.rcParams["axes.unicode_minus"] = False
    return "default"


FONT_NAME = configure_cjk_font()


def finite(value: Any) -> Any:
    """把 NumPy 类型转换成可写入 JSON 的有限 Python 类型。"""
    if isinstance(value, dict):
        return {key: finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(item) for item in value]
    if isinstance(value, np.ndarray):
        return finite(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(finite(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def read_audio(path: Path, sample_rate: int = TARGET_SR) -> tuple[np.ndarray, dict[str, Any]]:
    """读取并统一为双声道；重采样只发生在内存中，不修改源文件。"""
    info = sf.info(path)
    audio, source_sr = sf.read(path, always_2d=True, dtype="float64")
    if source_sr != sample_rate:
        audio = signal.resample_poly(audio, sample_rate, source_sr, axis=0)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]
    metadata = {
        "path": str(path),
        "source_sample_rate": source_sr,
        "analysis_sample_rate": sample_rate,
        "source_channels": info.channels,
        "duration_seconds": float(info.duration),
        "format": info.format,
        "subtype": info.subtype,
    }
    return audio, metadata


def mono(audio: np.ndarray) -> np.ndarray:
    return np.mean(audio, axis=1)


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)) + EPS))


def db(value: Any) -> Any:
    return 20.0 * np.log10(np.maximum(value, EPS))


def frame_view(values: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if len(values) < frame:
        values = np.pad(values, (0, frame - len(values)))
    count = 1 + (len(values) - frame) // hop
    shape = (count, frame)
    strides = (values.strides[0] * hop, values.strides[0])
    return np.lib.stride_tricks.as_strided(values, shape=shape, strides=strides).copy()


def frame_rms(values: np.ndarray, frame: int, hop: int) -> np.ndarray:
    return np.sqrt(np.mean(frame_view(values, frame, hop) ** 2, axis=1) + EPS)


def frame_peak(values: np.ndarray, frame: int, hop: int) -> np.ndarray:
    return np.max(np.abs(frame_view(values, frame, hop)), axis=1)


def integrated_loudness(audio: np.ndarray, sample_rate: int = TARGET_SR) -> float:
    """优先使用 EBU/ITU 门限；极短文件退化为 RMS 代理。"""
    meter = pyln.Meter(sample_rate)
    try:
        return float(meter.integrated_loudness(audio))
    except ValueError:
        return float(db(rms(audio)) - 0.7)


def loudness_stats(audio: np.ndarray, sample_rate: int = TARGET_SR) -> dict[str, Any]:
    """计算响度、峰均关系和限制器密度证据，不反推具体母带链。"""
    center = mono(audio)
    window = int(0.4 * sample_rate)
    hop = int(0.1 * sample_rate)
    momentary_proxy = db(frame_rms(center, window, hop))
    active = momentary_proxy[momentary_proxy > np.percentile(momentary_proxy, 15)]
    lra_proxy = float(np.percentile(active, 95) - np.percentile(active, 10)) if len(active) else 0.0
    # 4 倍过采样只用于估计采样间峰值，因此称为 True Peak Proxy。
    oversampled = signal.resample_poly(audio, 4, 1, axis=0)
    true_peak = float(db(np.max(np.abs(oversampled))))
    lufs = integrated_loudness(audio, sample_rate)
    peak_frames = db(frame_peak(center, int(0.05 * sample_rate), int(0.025 * sample_rate)))
    ceiling = np.percentile(peak_frames, 99)
    return {
        "lufs_i": lufs,
        "true_peak_proxy_dbtp": true_peak,
        "plr_db": true_peak - lufs,
        "crest_db": float(db(np.max(np.abs(center)) / rms(center))),
        "lra_proxy_db": lra_proxy,
        "limiter_density_pct": float(np.mean(peak_frames > ceiling - 0.35) * 100),
        "momentary_proxy_db": momentary_proxy,
        "momentary_times": np.arange(len(momentary_proxy)) * hop / sample_rate + window / sample_rate / 2,
    }


def stft_power(values: np.ndarray, sample_rate: int = TARGET_SR,
               n_fft: int = 2048, hop: int = 512) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frequencies, times, spectrum = signal.stft(
        values, fs=sample_rate, window="hann", nperseg=n_fft,
        noverlap=n_fft - hop, boundary=None, padded=False,
    )
    return frequencies, times, np.abs(spectrum) ** 2 + EPS


def band_frame_db(values: np.ndarray, sample_rate: int = TARGET_SR) -> tuple[np.ndarray, np.ndarray]:
    """把频谱汇总为混音决策常用宽频带，降低逐 FFT Bin 的偶然波动。"""
    frequencies, times, power = stft_power(values, sample_rate)
    rows = []
    for low, high, _ in BANDS:
        mask = (frequencies >= low) & (frequencies < min(high, sample_rate / 2))
        rows.append(10 * np.log10(np.mean(power[mask], axis=0) + EPS))
    return times, np.asarray(rows)


def tonal_stats(audio: np.ndarray, sample_rate: int = TARGET_SR) -> dict[str, Any]:
    _, frames = band_frame_db(mono(audio), sample_rate)
    median = np.median(frames, axis=1)
    # 用中间 80 Hz–8 kHz 的整体均值相对化，减小“更响=更亮”的偏见。
    relative = median - np.mean(median[1:8])
    dynamic = np.percentile(frames, 90, axis=1) - np.percentile(frames, 10, axis=1)
    return {
        "band_relative_db": relative,
        "band_dynamic_range_db": dynamic,
    }


def butter_filter(values: np.ndarray, low: float | None = None,
                  high: float | None = None, sample_rate: int = TARGET_SR,
                  order: int = 4) -> np.ndarray:
    if low is not None and high is not None:
        sos = signal.butter(order, [low, high], btype="bandpass", fs=sample_rate, output="sos")
    elif low is not None:
        sos = signal.butter(order, low, btype="highpass", fs=sample_rate, output="sos")
    else:
        sos = signal.butter(order, high, btype="lowpass", fs=sample_rate, output="sos")
    return signal.sosfiltfilt(sos, values, axis=0)


def stereo_stats(audio: np.ndarray, sample_rate: int = TARGET_SR) -> dict[str, Any]:
    """M/S 使用波形线性组合，再比较 RMS；负 dB 只表示 Side 小于 Mid。"""
    left, right = audio[:, 0], audio[:, 1]
    mid = (left + right) / 2
    side = (left - right) / 2
    stereo_rms = np.sqrt((np.mean(left ** 2) + np.mean(right ** 2)) / 2 + EPS)
    band_width = []
    for low, high, _ in BANDS:
        limited_high = min(high, sample_rate / 2 - 20)
        filtered_mid = butter_filter(mid, low, limited_high, sample_rate)
        filtered_side = butter_filter(side, low, limited_high, sample_rate)
        band_width.append(float(db(rms(filtered_side) / rms(filtered_mid))))
    low_audio = butter_filter(audio, high=150, sample_rate=sample_rate)
    low_mid = mono(low_audio)
    low_side = (low_audio[:, 0] - low_audio[:, 1]) / 2
    correlation = 1.0 if np.std(left) < EPS or np.std(right) < EPS else float(np.corrcoef(left, right)[0, 1])
    return {
        "side_mid_db": float(db(rms(side) / rms(mid))),
        "correlation": correlation,
        "mono_fold_loss_db": float(db(rms(mid) / stereo_rms)),
        "low_side_mid_db": float(db(rms(low_side) / rms(low_mid))),
        "band_side_mid_db": band_width,
    }


def spectral_centroid(values: np.ndarray, sample_rate: int = TARGET_SR) -> float:
    frequencies, _, power = stft_power(values, sample_rate)
    spectrum = np.mean(power, axis=1)
    return float(np.sum(frequencies * spectrum) / np.sum(spectrum))


def automatic_sections(audio: np.ndarray, sample_rate: int = TARGET_SR) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """用能量、频谱重心和宽度变化寻找结构候选；不猜 Verse/Chorus 名称。"""
    duration = len(audio) / sample_rate
    frame_seconds, hop_seconds = 0.8, 0.2
    frame, hop = int(frame_seconds * sample_rate), int(hop_seconds * sample_rate)
    center = mono(audio)
    energy = db(frame_rms(center, frame, hop))
    centroid, width = [], []
    for index in range(len(energy)):
        segment = audio[index * hop:index * hop + frame]
        centroid.append(spectral_centroid(mono(segment), sample_rate))
        # 结构检测只需要全频宽度变化；这里直接算 M/S，避免逐帧重复做九段滤波。
        left, right = segment[:, 0], segment[:, 1]
        mid, side = (left + right) / 2, (left - right) / 2
        width.append(float(db(rms(side) / rms(mid))))
    features = np.column_stack([energy, np.log1p(centroid), width])
    features = (features - np.median(features, axis=0)) / (np.std(features, axis=0) + EPS)
    novelty = np.r_[0.0, np.linalg.norm(np.diff(features, axis=0), axis=1)]
    novelty = signal.medfilt(novelty, 5)
    times = np.arange(len(novelty)) * hop_seconds + frame_seconds / 2
    peaks, properties = signal.find_peaks(
        novelty,
        distance=max(1, int(4.0 / hop_seconds)),
        prominence=max(0.05, float(np.percentile(novelty, 65) * 0.35)),
    )
    ranked = sorted(peaks, key=lambda index: properties["prominences"][list(peaks).index(index)], reverse=True)
    boundaries = sorted(float(times[index]) for index in ranked[:5] if 2.0 < times[index] < duration - 2.0)
    edges = [0.0, *boundaries, duration]
    sections = [
        {"name": f"auto-{index + 1}", "start": edges[index], "end": edges[index + 1], "automatic": True}
        for index in range(len(edges) - 1)
        if edges[index + 1] - edges[index] >= 1.0
    ]
    return sections, {"times": times, "novelty": novelty, "boundaries": boundaries}


def section_stats(audio: np.ndarray, sections: list[dict[str, Any]],
                  sample_rate: int = TARGET_SR) -> dict[str, Any]:
    result = {}
    for section in sections:
        start = max(0, int(section["start"] * sample_rate))
        end = min(len(audio), int(section["end"] * sample_rate))
        segment = audio[start:end]
        if len(segment) < sample_rate // 2:
            continue
        result[section["name"]] = {
            "start": section["start"],
            "end": section["end"],
            "lufs": integrated_loudness(segment, sample_rate),
            "rms_db": float(db(rms(segment))),
            "centroid_hz": spectral_centroid(mono(segment), sample_rate),
            "side_mid_db": stereo_stats(segment, sample_rate)["side_mid_db"],
        }
    return result


def envelope(audio: np.ndarray, sample_rate: int = TARGET_SR,
             frame_ms: float = 50, hop_ms: float = 25) -> np.ndarray:
    values = mono(audio) if audio.ndim == 2 else audio
    frame = max(16, int(sample_rate * frame_ms / 1000))
    hop = max(8, int(sample_rate * hop_ms / 1000))
    return frame_rms(values, frame, hop)


def transient_stats(audio: np.ndarray, sample_rate: int = TARGET_SR) -> dict[str, float]:
    """用正向 Spectral Flux 描述瞬态，不把频带代理冒充鼓件分类器。"""
    center = mono(audio)
    _, _, power = stft_power(center, sample_rate, 1024, 256)
    flux = np.maximum(0, np.diff(np.sqrt(power), axis=1)).sum(axis=0)
    median = np.median(flux)
    threshold = median + 2.5 * np.median(np.abs(flux - median))
    peaks, _ = signal.find_peaks(flux, height=threshold, distance=max(1, int(sample_rate / 256 * 0.08)))
    minutes = len(center) / sample_rate / 60
    return {
        "transient_density_per_min": float(len(peaks) / max(minutes, EPS)),
        "p95_flux": float(np.percentile(flux, 95)),
        "crest_db": float(db(np.max(np.abs(center)) / rms(center))),
    }


def source_confidence(origin: str) -> tuple[float, str, str]:
    table = {
        "original_stems": (0.95, "高", "工程原始 Stem，关系测量直接"),
        "official_stems": (0.85, "中高", "官方 Stem 可能已含总线处理"),
        "source_separated": (0.58, "中", "分离泄漏、伪影和共享残留会污染关系"),
        "master_only": (0.30, "证据不足", "没有可观测 Stem"),
    }
    return table.get(origin, (0.20, "低", "Stem 来源未知"))


def cap_confidence(level: str, cap: str) -> str:
    """把算法基础等级按最弱 Stem 来源上限降级。"""
    if level not in CONFIDENCE_ORDER or cap not in CONFIDENCE_ORDER:
        return "中低"
    return CONFIDENCE_ORDER[min(CONFIDENCE_ORDER.index(level), CONFIDENCE_ORDER.index(cap))]


def confidence_title(title: str, level: str, limitation: str) -> str:
    return f"{title}\n可信度：{level}｜{limitation}"


def style_axes(axis: plt.Axes) -> None:
    axis.grid(alpha=0.2)
    axis.tick_params(labelsize=8)


def save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=155, bbox_inches="tight")
    plt.close(figure)


def insufficient_figure(path: Path, titles: list[str], reason: str) -> None:
    """缺少 Stem 时输出证据不足面板，避免用零值伪装真实测量。"""
    columns = min(3, len(titles))
    rows = int(np.ceil(len(titles) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(5.2 * columns, 3.8 * rows))
    axes_array = np.atleast_1d(axes).ravel()
    for axis, title in zip(axes_array, titles):
        axis.set_title(confidence_title(title, "证据不足", reason))
        axis.text(0.5, 0.5, "缺少可用 Stem\n未生成零值假数据", ha="center", va="center", transform=axis.transAxes)
        axis.set_axis_off()
    for axis in axes_array[len(titles):]:
        axis.set_axis_off()
    save_figure(figure, path)
