# 完整混音测量与可信度

## 正式纳入的方法

### Master-only 可运行

- Integrated Loudness、True Peak Proxy、PLR、Crest、短时响度、LRA Proxy、Limiter Density Evidence。
- 响度相对化的感知宽频带音色，以及各频带 P90–P10 动态范围。
- Global/Band Side-to-Mid、L/R Correlation、Mono Fold-down Loss、150 Hz 以下低频 Side-to-Mid。
- 自动结构 Novelty、段落响度和段落宽度；人工段落优先。
- 多参考 Median、IQR、Direction Consistency 和 Outlier Score。

### 需要可靠 Stem

- Vocal-to-Instrument Ratio 及段落分布。
- 1.2–5 kHz Vocal/Instrument Presence 冲突和 5–12 kHz Hat/Sibilance 冲突代理。
- Vocal 与 Instrumental 的 M/S 宽度差；它是空间关系，不是一个黑箱“融合度分数”。
- Stem 相对电平、活动 Stem 数、Dense-frame Probability 和 Spectral Occupancy。
- Drums 全带瞬态，以及低/中/高频 Kick/Snare/Hat 频带代理。
- Bass 频带、Crest、包络 IQR、M/S 与相对 Drums 电平。
- Lead-vs-Vocal Presence Margin、Pad-vs-Vocal Width Gap、Lead/Pad 相对电平。

## 不纳入强结论

受控实验没有稳定通过以下两项：

- 用全带 Vocal/Instrument 包络相关证明 Vocal-triggered Ducking。
- 用 Kick/Bass 包络相关证明 Sidechain Compression。

正式脚本不输出这两类强因果判断。若未来保留相关曲线，只能标为 `中低` 辅助证据，并明确“不能证明 Sidechain”。

也不从 Master 精确识别 Guitar/Piano/Synth、商业插件、Preset 或完整母带链。

## 可信度传播

子图基础等级还要受来源影响：

- `original_stems`：不降级。
- `official_stems`：最高 `中高`。
- `source_separated`：最高 `中`；高频、瞬态、立体声关系通常再谨慎一级。
- `master_only`：Stem 关系显示 `证据不足`。

等级只使用：`高`、`中高`、`中`、`中低`、`低`、`证据不足`。每个坐标轴必须单独写在标题第二行，不能只给整张图一个等级。

## 多参考边界

- 1 个参考：只做一对一差异，不称为“参考共性”。
- 2 个参考：可显示范围，可信度 `中低`。
- 3–4 个独立参考：可用 Median/IQR，可信度通常 `中`。
- 5 个以上可比参考：参考区间可到 `中高`，但仍不是跨曲风工业常模。

重复文件、同一母版不同编码、同一歌曲切片不得重复计权。

## 图表和报告

脚本生成：

- `M01_master_loudness_dynamics.png`
- `M02_tonal_balance.png`
- `M03_stereo_translation.png`
- `M04_section_structure.png`
- `M05_vocal_instrument_balance.png`
- `M06_frequency_conflict.png`
- `M07_fusion_components.png`
- `M08_stem_balance_arrangement.png`
- `M09_drums_bass.png`
- `M10_lead_pad_occupancy.png`
- `M11_reference_intervals.png`
- `M12_reference_matching_confidence.png`

没有足够 Stem 时，`M05`–`M10` 仍可生成“证据不足”面板，清楚说明缺什么，不画零值假数据。

报告必须说明：图上看到了什么、通常听起来怎样、可能原因、优先尝试什么，以及何时停止。音色和动态建议先做等响度 A/B Test。
