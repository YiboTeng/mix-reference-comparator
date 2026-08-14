---
name: compare-vocal-references
description: 将一条待评估人声音轨与一个或多个参考人声音轨进行可重复的对比测量、因果诊断、中文 PNG 可视化和插件链建议。当用户提供 WAV、MP3、FLAC、M4A 独唱、处理后人声、Stem 或人声分离文件，并要求比较静态或动态音色、低中频遮蔽、咬字、宏微动态、谐波、颗粒、齿音、立体声、混响、延迟、Doubler、Microshift、双轨、点状或区域型声场、分离伪影，建立多参考正常区间、识别参考共性/个性，或让自己的混音接近参考时使用。
---

# 对比人声参考

以证据为基础对比人声，但不要假装能从渲染后的音频准确反推出原始插件名称。必须区分可测量差异、合理处理原因和录音源/分离算法限制。

## 必须阅读

分析音频前，完整阅读：

- `references/measurement-and-causality.md`：测量定义、因果置信度和伪影检查。
- `references/report-and-visualization.md`：交付物、图表读法和面向初学者的讲解方式。

进行完整对比，或用户要求细分音色、咬字、动态、压缩、谐波和颗粒时，完整阅读 `references/detail-diagnostics.md`。

输入包含多个参考，或需要建立参考区间、识别共性与离群参考时，完整阅读 `references/reference-set-comparison.md`。

如果用户还要求复现目标声音，完整阅读 `references/plugin-actions.md`。

如果用户要求区分点状副本与区域型声场，或辨别 Doubler、Haas Delay、Microshift、双轨、调制和宽混响，完整阅读 `references/spatial-field-diagnostics.md`。

## 工作流程

### 1. 判断输入是否适合比较

先明确一条“待评估人声”，其余均为参考。逐个确认：

- 原始干声、处理后独唱、人声 Stem（分轨）还是 Source Separation（人声分离）文件。
- 格式、采样率、位深/Codec（编码）、声道数、时长以及是否含母带总线处理。
- 歌手、音域、段落、编曲和静音比例是否可比。

把人声分离参考当成“经过处理且含伪影的证据”，不能当作原始录音室干声。明确说明人声分离可能产生 Musical Noise（音乐噪声）、瞬态涂抹、基频缺失、虚假立体声宽度和编码伪影。

不要求歌曲时间对齐。若演唱不一致，比较长期统计和各自独立选出的高能量片段。只有同一表演且可靠对齐时，才增加时间对齐或 Null Test（抵消测试）。

多个参考必须检查独立性。重复文件、同一母版的不同编码或同一歌曲切片不能重复计权。优先选择风格、音域、声部、年代和目标用途相近的参考。

### 2. 准备确定性的分析环境

选择输出目录时，若 `C:\Projects\work` 存在且可写，默认将其作为 `--out-dir`；为避免不同任务的同名文件互相覆盖，可在其中创建清晰命名的任务子目录。若该路径不存在或不可写，再使用当前任务的独立输出目录。绝不覆盖源音频。

寻找 Python 和依赖。优先使用 Codex 工作区自带运行环境。如果缺少 `numpy`、`scipy`、`soundfile`、`matplotlib`、`pyloudnorm` 或 `librosa`，运行：

```powershell
python scripts/bootstrap_deps.py --target <当前任务的依赖目录>
```

如果联网安装需要用户批准，先请求批准。之后把依赖目录加入 `PYTHONPATH` 再运行分析器。

所有新任务优先运行统一编排器。一个参考时它生成完整一对一报告；多个参考时还生成参考区间、共性和离群参考汇总：

```powershell
python scripts/compare_reference_set.py `
  --target "待评估人声=<用户人声路径>" `
  --reference "参考 A=<参考路径 A>" `
  --reference "参考 B=<参考路径 B>" `
  --separated-reference "参考 B" `
  --analysis-level full `
  --out-dir <输出目录>
```

`--reference` 可重复任意次数。只有对应输入确为 Source Separation（人声分离）时才传 `--separated-reference`，标签必须完全一致。分析级别：`basic` 为基础；`detail` 为基础加细节；`full` 再加入高级空间诊断。完整报告默认使用 `full`。

编排器必须生成：

- `reference-set-metrics.json` 与 `reference-set-report.md`。
- `R1_levels_and_timbre.png`、`R2_dynamics_and_texture.png`、`R3_spatial_reference_field.png`、`R4_consensus_and_outliers.png`。
- `pairwise/<reference>/basic`：每个参考的原基础图、指标和草稿。
- `pairwise/<reference>/detail`：`detail` 或 `full` 时的细节诊断。
- `pairwise/<reference>/spatial`：`full` 时的高级空间诊断。
- `execution-log.json`：实际执行命令和结果。

仅在调试一对一内核时才直接运行以下脚本。基础诊断：

```powershell
python scripts/analyze_vocals.py --mix <用户人声> --reference <参考人声> --out-dir <输出目录> --mix-label "待评估人声" --reference-label "参考人声"
```

细节诊断：

```powershell
python scripts/detail_diagnostics.py `
  --input "用户人声=<用户人声路径>" `
  --input "参考人声=<参考人声路径>" `
  --out-dir <输出目录> `
  --segment-seconds 35
```

细节脚本恰好接收两个输入，分别独立寻找高能量片段。编排器会为每个参考自动调用。它生成：

- `metrics.json`：动态音色、咬字、宏/微动态、分频动态、谐波与辅助音高数据。
- `experiment_report.md`：细分诊断草稿与逐项可靠性。
- `D1_timbre_detail.png`：动态频段、音量相关音色、持续共振候选及 Presence/Air/齿音。
- `D2_articulation_time.png`：音节包络、字头后 Presence、长句尾条件检测及辅音/元音代理。
- `D3_dynamics_detail.png`：宏/微动态、峰值密度、分频动态及削波行为代理。
- `D4_texture_pitch.png`：谐波衰减、高阶份额、噪声平坦度、响度相关纹理和低可信音高辅助项。

不要把 D4 的音高平台、颤音或共振峰代理写成 Auto-Tune、Formant 处理或具体插件识别。没有足够合格长句尾时，D2 必须显示证据不足。

当用户追问声场为何呈“左中右几个点”或“连续区域”，或要求区分 Doubler、Microshift、双轨、短延迟、调制与宽混响时，再运行高级空间诊断：

```powershell
python scripts/spatial_diagnostics.py `
  --input "用户人声=<用户人声路径>" `
  --input "参考人声=<参考人声路径>" `
  --out-dir <输出目录> `
  --segment-seconds 20
```

若输入是人声分离文件，追加 `--separated-label "<对应标签>"`。标签必须与 `--input` 中的标签完全一致。脚本生成：

- `spatial_metrics.json`：相干度、GCC-PHAT 延迟峰、时频声像分布、Mid/Side 包络及启发式证据。
- `experiment_report.md`：高级空间诊断草稿。
- `01_spatial_distribution.png`：ILD/IPD 时频声像分布。
- `02_coherence_and_delay.png`：分频段相干度和非零延迟峰。
- `03_stage_width_and_tails.png`：宽度随音量变化及停顿后 Side 尾部。

不要把启发式分数解释为具体插件识别。若用户只关心音色或动态，可用 `--analysis-level detail` 跳过高级空间诊断。

### 3. 检查每张图

用图片查看工具打开全部 PNG，确认：

- 每个独立子图都在自己的标题第二行显示 `可信度：等级｜主要限制`；不能只给整张 PNG 一个总等级。
- 等级只能使用 `高`、`中高`、`中`、`中低`、`低`、`证据不足`，并与 `references/report-and-visualization.md` 的定义一致。
- 中文及其他 Unicode 字符正常显示。
- 坐标、图例、标签和标注没有重叠。
- 曲线没有跑出绘图区。
- 图中确实有真实数据，而不是空白或被裁切的图。

多个参考时还要确认：参考区间不是把所有曲线简单平均；R4A 和 R4B 的可信度随参考数量变化；少于三个参考时不得把差异写成“参考共性”。

发现问题就修复脚本或换用可用中文字体后重跑。不要交付没有检查过的图。

报告解释必须沿用图内可信度。不得把标为“低”“证据不足”的子图写成强推断；同一结论若由多个独立中高/高可信子图共同支持，才可升级为强推断。

### 4. 分层诊断

按此顺序进行，避免响度偏见影响音色判断：

1. 文件来源、有效性和人声分离伪影。
2. 绝对响度、峰值、活跃电平分布和 Crest Factor（峰均比）。
3. 响度形状归一化后的音色平衡和频段差异。
4. 时间—频率谐波结构、齿音、噪声和伪影纹理。
5. 动态频段分布、强弱声音色、低中频遮蔽及 Presence/Air/齿音拆分。
6. 字头/主体、音节包络、音节一致性、宏/微动态和分频动态。
7. 谐波衰减、高阶谐波份额、频谱平坦度和响度相关纹理。
8. 各频段及各音量阶段的 Mid/Side（中/侧）变化。
9. 仅把音高平台、颤音和共振峰代理作为低可信辅助证据。
10. 需要时检查时频声像分布、分频相干度、非零延迟峰和 Side 尾部。
11. 可能的处理链及可控制的验证实验。

给每个结论标明：

- **测量结果**：由数据或图直接支持。
- **强推断**：多项独立观察共同指向同一原因。
- **假设**：合理但无法仅凭成品音频确认。

不能仅凭听感判断具体商业插件。先说处理器类别，再给可选插件示例和 DAW 自带/免费替代。

### 5. 把“颗粒感”拆成多个来源

不要把颗粒感等同于噪声或某一个失真插件。至少检查：

- 低中频遮蔽：150–500 Hz 过多会盖住上方谐波。
- 谐波延伸：饱和、削波、明亮录音、Formant（共振峰）和音高校正可能延伸规则谐波。
- 包络密度：快速压缩和削波会改变音节边缘。
- 齿音结构：5–12 kHz 可能包含有声音高谐波、无声辅音、激励或分离残留。
- 人声分离：水声、砂砾感和相位化残留可能听起来像人为质感。

分析器给出的高频谐波集中百分比只能用于输入之间的相对比较，不能当成物理真值，更不能证明使用了某款插件。

### 6. 转换为复现方案

按优先级给出操作，并说明：

- 每个处理器的目的。
- 频率/范围、Ratio（压缩比）、Attack（启动）、Release（释放）、压缩量、Mix（混合比）和发送量起点。
- 这是串联、Parallel（并联）还是 Send/Return（发送/返回）。
- 调整时听什么。
- 何时停止以及常见失败表现。
- 不知道用户插件清单时，补充 DAW 自带或免费替代。

优先设计可撤回的 A/B Test（开关对比），每次比较 EQ、压缩、饱和或削波都要保持等响度。

### 7. 用中文交付图文报告

最终回复以中文为主。只保留必要的标准英文术语、插件名称、参数名和文件名，并在首次出现时给出中文解释。不要写连续大段英文。

使用绝对路径直接渲染 PNG；不能只给交互式可视化指令或裸文件链接。每张图后紧跟中文讲解。

结尾必须包含：

- 3–5 项最高优先级调整。
- 一条处理链结构图。
- 歌手、话筒、编曲、人声分离和编码带来的限制。
- 如果还不知道 DAW 和插件清单，邀请用户补充以便给出精确设置。

## 边界与原则

- 只读源音频或使用副本。
- 不声称人声分离文件就是原始独唱轨。
- 不把不同歌手之间的大幅频段差值直接照抄成静态 EQ 增益。
- 不把“更响”描述成“更好”；主观判断前必须等响度比较。
- 除非用户明确追求分离伪影美学，否则不要主动制造这种伪影。
