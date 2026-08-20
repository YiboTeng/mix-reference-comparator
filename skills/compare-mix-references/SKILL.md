---
name: compare-mix-references
description: 将一首待评估的完整混音成品与一个或多个完整混音参考进行可重复测量、中文 PNG 可视化和多参考诊断。适用于包含人声与伴奏的 Mix/Master，比较响度、动态、音色平衡、立体声、段落；有可靠 Stem 时进一步比较人声/伴奏平衡、遮蔽、鼓、Bass、Lead/Pad 与编曲密度。不用于独唱干声或仅有人声的 Stem。
---

# 对比完整混音参考

对完整 Mix/Master 做证据化比较。Master-only 测量与 Stem 关系测量必须分层呈现；没有 Stem 时不得假装能从最终混音无损拆出 Vocal、Drums、Bass 或 Lead。

## 必须阅读

运行前完整阅读：

- `references/mix-input-schema.md`：项目清单、Stem 字段、来源等级与命令格式。
- `references/mix-measurement-and-confidence.md`：正式纳入的测量、可信度、失败实验和报告规则。

## 模式边界

- 输入是干声、独唱、人声 Stem 或人声分离文件时，改用 `$compare-vocal-references`。
- 输入是包含 Vocal 与 Beats/Instrumental 的完整 Mix/Master 时使用本 skill。
- Target 与 Reference 不要求时间对齐；不同歌曲比较长期统计、归一化段落位置和稳健参考区间。
- 同一表演且可靠对齐时才考虑逐时间比较或 Null Test。

## 输入引导

用户已经要求分析但信息不完整时，一次性请求：

- 一首待评估完整混音的附件或绝对路径。
- 至少一首完整混音参考的附件或绝对路径。
- 可选：每首歌对应的 `vocal`、`drums`、`bass`、`lead`、`pad` Stem，以及来源是 `original_stems`、`official_stems` 还是 `source_separated`。
- 可选：人工确认的段落名称与起止时间。
- 可选：报告输出目录。

未指定输出目录时，先检查 `C:\Projects\work`，并明确告知实际落点：

> 如果不指定输出目录，我会把本次报告、图表、指标和项目清单保存到 `C:\Projects\work\<本次任务子目录>`。

只缺输出路径时无需等待确认，说明默认路径后继续。

## 工作流

1. 检查文件存在性、Codec、采样率、声道数、时长、是否为 Master、Stem 来源及参考独立性。
2. 按 `references/mix-input-schema.md` 在输出目录创建任务级 `mix-project.json`；不修改源音频。
3. 准备 Python 环境。缺少依赖时运行：

```powershell
python scripts/bootstrap_deps.py --target <当前任务依赖目录>
```

4. 运行统一分析器：

```powershell
python scripts/compare_mix_references.py `
  --project <输出目录\mix-project.json> `
  --out-dir <输出目录>
```

5. 确认生成 `mix-reference-metrics.json`、`mix-reference-report.md`、`execution-log.json` 和 `M01`–`M12` PNG。
6. 用图片查看工具打开全部 PNG。每个独立子图必须在标题第二行显示 `可信度：等级｜主要限制`；检查中文字体、裁切、空图、坐标和图例。
7. 先把人工复核后的解释写回 `mix-reference-report.md`，再在 Codex 回复中完整呈现报告正文和图片，最后列绝对文件路径。

## 解释顺序

1. 输入有效性、Stem 可用性与 Source Separation 限制。
2. Integrated Loudness、True Peak、PLR、短时响度、LRA 与限制器密度证据。
3. 响度相对化后的静态/动态频谱平衡。
4. 全频与分频 M/S、Correlation、Mono Fold-down、低频居中。
5. 自动或人工段落的响度、宽度及结构差异。
6. 有 Stem 时再解释 Vocal/Instrument Ratio、遮蔽、空间关系、Stem 平衡、鼓/Bass、Lead/Pad 和编曲密度。
7. 多参考 Median、IQR、Direction Consistency、Outlier 与来源可信度。

不得把简单包络相关写成 Vocal Ducking、Kick–Bass Sidechain 或具体处理器的证明；这些方法在受控实验中没有稳定通过。

## 最终交付

以中文为主，区分“测量结果”“强推断”“假设”。必须先以 `## 报告正文` 完整呈现最终 Markdown 报告与绝对路径图片，再以 `## 输出文件` 列出项目清单、报告、指标、执行日志和主要 PNG。不得只给路径或摘要。

结尾给出 3–5 项最高优先级调整，并说明每项需要怎样做等响度 A/B Test。没有工程文件或可靠 Stem 时，不给精确单轨插件参数。
