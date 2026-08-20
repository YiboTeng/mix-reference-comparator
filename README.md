# Mix Reference Comparator

面向 Codex 的混音参考对比插件。启动时可选择两种模式：

1. 干声/人声参考对比。
2. 包含人声与伴奏的完整 Mix/Master 参考对比。

插件以可重复测量和因果边界为核心，不根据渲染音频猜具体商业插件、Preset 或完整母带链。所有 PNG 的每个独立子图分别显示可信度和主要限制。

## 两种模式

### 干声对比

使用 `compare-vocal-references`：

- 输入一条 Target Vocal 和一个或多个 Reference Vocal。
- 支持干声、处理后独唱、Vocal Stem 和 Source Separation 人声。
- 分析 Levels、Timbre、Dynamics、Texture、Articulation、Sibilance、Harmonics 和高级立体声场。
- 可辅助区分点状 Doubler、Haas Delay、Microshift、Double Tracking、Modulation 和 Wide Reverb 的证据特征。

### 完整混音成品对比

使用 `compare-mix-references`：

- 输入一首 Target Mix/Master 和一个或多个完整混音参考。
- 只有 Master 时分析响度、峰值、动态、静态/动态音色、M/S、Correlation、Mono Translation、低频居中和段落结构。
- 提供可靠 Vocal/Drums/Bass/Lead/Pad Stems 后，进一步分析 Vocal-to-Instrument Ratio、频率冲突、空间关系、Stem 平衡、鼓瞬态、Bass、Lead/Pad 占位和编曲密度。
- 多参考模式统计 Median、IQR、Direction Consistency、Outlier 和 Reference Compatibility。
- `original_stems`、`official_stems`、`source_separated` 和 `master_only` 会传播为不同可信度上限。

受控实验没有稳定通过以下强因果推断，因此正式模式不会把它们写成结论：

- 用简单全带包络相关证明 Vocal-triggered Ducking。
- 用 Kick/Bass 包络相关证明 Sidechain Compression。

## 启动与模式选择

笼统启动插件时，使用 `select-reference-analysis-mode` 先询问：

```text
请选择分析模式：
1）干声/独唱/Vocal Stem 对比
2）包含人声与伴奏的完整混音成品对比
```

用户已经明确素材类型时直接进入对应模式，不重复询问。干声与完整 Mix 不应放在同一个参考统计集。

## 输入格式

音频支持 `WAV`、`MP3`、`FLAC` 等 `soundfile` 可解码格式。部分 `AAC/M4A` 可能需要先转换：

```powershell
ffmpeg -i "input.m4a" -c:a pcm_s24le "output.wav"
```

两种模式都需要一条 Target 和至少一条 Reference。参考不要求时间对齐；不同歌曲比较长期统计和各自段落，只有同一表演且可靠对齐时才适合 Null Test。

完整混音模式使用任务级 `mix-project.json`。示例和字段定义见：

- `skills/compare-mix-references/references/mix-input-schema.md`
- `skills/compare-mix-references/references/mix-measurement-and-confidence.md`

## 默认输出与对话内报告

如果用户没有指定输出目录，插件会在运行前明确说明实际路径。默认使用：

```text
C:\Projects\work\<本次任务子目录>
```

该目录不存在或不可写时，改用当前任务的独立可写目录。不会覆盖源音频。

分析完成后，Codex 必须先在回复中完整呈现报告正文和主要图片，再列出报告、指标、执行日志和 PNG 的绝对路径。用户不需要 Markdown 查看器也能阅读结果。

## 直接运行：干声模式

```powershell
Set-Location ".\skills\compare-vocal-references"
python .\scripts\bootstrap_deps.py --target "C:\Projects\work\vocal-reference-deps"
$env:PYTHONPATH = "C:\Projects\work\vocal-reference-deps"

python .\scripts\compare_reference_set.py `
  --target "My Vocal=C:\Audio\my-vocal.wav" `
  --reference "Reference A=C:\Audio\reference-a.wav" `
  --reference "Reference B=C:\Audio\reference-b.wav" `
  --separated-reference "Reference B" `
  --analysis-level full `
  --out-dir "C:\Projects\work\vocal-reference-set"
```

`--reference` 可以重复。`basic`、`detail`、`full` 分别增加细节与高级空间分析。

## 直接运行：完整混音模式

```powershell
Set-Location ".\skills\compare-mix-references"
python .\scripts\bootstrap_deps.py --target "C:\Projects\work\mix-reference-deps"
$env:PYTHONPATH = "C:\Projects\work\mix-reference-deps"

python .\scripts\compare_mix_references.py `
  --project "C:\Projects\work\my-analysis\mix-project.json" `
  --out-dir "C:\Projects\work\my-analysis"
```

完整混音模式输出：

```text
output/
├── mix-project.json
├── mix-reference-metrics.json
├── mix-reference-report.md
├── execution-log.json
├── M01_master_loudness_dynamics.png
├── M02_tonal_balance.png
├── M03_stereo_translation.png
├── M04_section_structure.png
├── M05_vocal_instrument_balance.png
├── M06_frequency_conflict.png
├── M07_fusion_components.png
├── M08_stem_balance_arrangement.png
├── M09_drums_bass.png
├── M10_lead_pad_occupancy.png
├── M11_reference_intervals.png
└── M12_reference_matching_confidence.png
```

缺少 Stems 时，`M05`–`M10` 会显示“证据不足”和所缺输入，不会用零值假数据代替。

## 可信度

子图等级只使用：`高`、`中高`、`中`、`中低`、`低`、`证据不足`。

报告结论分为：

1. **测量结果**：数据直接支持。
2. **强推断**：多个独立中高/高可信证据指向同一方向。
3. **假设**：合理但无法仅凭渲染音频确认。

少于三个参考不能称为稳定“参考共性”。Source Separation 会产生泄漏、Musical Noise、瞬态涂抹和虚假宽度，必须降级解释。

## 项目结构

```text
mix-reference-comparator/
├── .codex-plugin/
│   └── plugin.json
├── skills/
│   ├── select-reference-analysis-mode/
│   │   ├── SKILL.md
│   │   └── agents/
│   ├── compare-vocal-references/
│   │   ├── SKILL.md
│   │   ├── agents/
│   │   ├── references/
│   │   └── scripts/
│   └── compare-mix-references/
│       ├── SKILL.md
│       ├── agents/
│       ├── references/
│       └── scripts/
└── README.md
```

## Codex 个人安装

```powershell
git clone https://github.com/YiboTeng/mix-reference-comparator.git `
  "$env:USERPROFILE\.codex\plugins\mix-reference-comparator"
```

个人 Marketplace 条目中的 `name` 使用 `mix-reference-comparator`，本地路径使用：

```text
./.codex/plugins/mix-reference-comparator
```

安装命令：

```powershell
codex plugin add mix-reference-comparator@personal
```

本地开发源更新后，需要走 cachebuster 与重装流程；不要直接把开发目录当成安装缓存。

## 当前限制

- 不从 Master 无损拆出 Vocal、Drums、Bass、Lead 或 Pad。
- 不识别具体 Plugin、Preset、母带链或万能参数。
- 自动段落边界不知道 Verse/Chorus 语义。
- 不同曲风、编曲、年代、平台母版和 Codec 会降低可比性。
- 报告建议必须结合监听、等响度 A/B 和真实工程上下文。
