# 完整混音项目清单

统一分析器只接收一个任务级 JSON 清单。相对路径以清单所在目录为基准；推荐写绝对路径。

```json
{
  "schema": "mix-reference-project-v1",
  "target": {
    "label": "My Mix",
    "mix": "C:\\Audio\\my-mix.wav",
    "bpm": 142,
    "stem_origin": "original_stems",
    "stems": {
      "vocal": "C:\\Audio\\stems\\vocal.wav",
      "drums": "C:\\Audio\\stems\\drums.wav",
      "bass": "C:\\Audio\\stems\\bass.wav",
      "lead": "C:\\Audio\\stems\\lead.wav",
      "pad": "C:\\Audio\\stems\\pad.wav"
    },
    "sections": [
      {"name": "verse", "start": 12.0, "end": 36.0},
      {"name": "chorus", "start": 36.0, "end": 60.0}
    ]
  },
  "references": [
    {
      "label": "Reference A",
      "mix": "C:\\Audio\\reference-a.wav",
      "stem_origin": "master_only"
    }
  ]
}
```

## 必填字段

- `schema`：固定为 `mix-reference-project-v1`。
- `target`：恰好一个待评估完整混音。
- `references`：至少一个参考完整混音。
- 每项的 `label` 和 `mix`。

标签必须唯一。`mix` 必须是包含人声和伴奏的完整 Mix/Master，不要填独唱干声。

## 可选元数据

- `bpm`：已知 Tempo 时填写正数。脚本只把它用于参考兼容性筛选；不提供时退化为音色与时长代理，不会从音频强猜 BPM。

## Stem 字段

`stems` 可省略。正式关系分析至少需要 `vocal`，并需要 `drums`、`bass`、`lead`、`pad` 中一个或多个用于组成 Instrumental。缺失的 Stem 只跳过相关角度，不以零值代替。

允许的 `stem_origin`：

- `original_stems`：工程原始分轨，来源可信度高。
- `official_stems`：官方 Stem，可能已带总线处理，来源可信度中高。
- `source_separated`：分离 Stem，泄漏和伪影使关系测量降为中或更低。
- `master_only`：没有 Stem，只运行 Master 层分析。

Source Separation 的多个 Stem 往往共享泄漏，不能把相互关系当作完全独立证据。

提供 `stems` 时必须同时填写真实的 `stem_origin`；没有任何 Stem 时应使用 `master_only`。字段与实际输入不一致时脚本会停止，避免错误的来源标签污染可信度。

## 段落字段

`sections` 可省略。提供时要求：

- `name` 非空。
- `0 <= start < end <= 文件时长`。
- 同一文件内不重叠。

未提供时脚本根据能量、Spectral Centroid 和 M/S 变化生成自动边界。自动边界只提供结构证据，不知道真实的 Verse/Chorus 语义。
