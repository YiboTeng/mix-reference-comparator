# Vocal Reference Comparator

一个面向 Codex 的人声参考对比插件。它将一条待评估人声与一个或多个参考人声进行可重复测量，生成中文报告、PNG 图表、参考区间、离群项判断和可执行的混音调整建议。

插件重点不是判断“用了哪一款插件”，而是区分：

- 可直接测量的音频差异；
- 多项证据支持的处理方向；
- 仅凭渲染音频无法确认的假设；
- Source Separation（人声分离）、编码、歌手和录音条件带来的限制。

## 主要能力

- 支持一条 Target Vocal（待评估人声）对一个或多个 Reference Vocal（参考人声）。
- 对比 Levels、Timbre、Dynamics、Texture、Articulation、Sibilance 和 Harmonics。
- 分析 Mid/Side、ILD、IPD、Coherence、Delay Peak 和 Side Tail 等空间证据。
- 辅助判断点状 Doubler、Haas Delay、Microshift、Double Tracking、Modulation 和 Wide Reverb 的差异特征。
- 多参考模式下统计 Median、IQR、Direction Consistency 和 Outlier Score。
- 为每个独立子图分别标注可信度及主要限制。
- 输出中文 Markdown 报告、JSON 指标、PNG 图表和 Execution Log。

## 适用输入

可以分析以下人声素材：

- 原始或已处理的独唱人声；
- Vocal Stem；
- Source Separation 得到的人声；
- `WAV`、`MP3`、`FLAC` 等 `soundfile` 可解码格式。

部分 `AAC/M4A` 文件可能无法被当前音频后端直接解码。遇到此情况，请先用 `ffmpeg` 转成 `WAV`：

```powershell
ffmpeg -i "input.m4a" -c:a pcm_s24le "output.wav"
```

参考文件不需要与待评估人声时间对齐。不同歌曲或不同演唱会分别选取高能量片段，并比较长期统计；只有同一表演且可靠对齐时才适合进行 Null Test。

## Codex 个人安装

### 1. Clone 仓库

```powershell
git clone https://github.com/YiboTeng/vocal-reference-comparator.git `
  "$env:USERPROFILE\.codex\plugins\vocal-reference-comparator"
```

### 2. 注册到 Personal Marketplace

个人市场文件位于：

```text
%USERPROFILE%\.agents\plugins\marketplace.json
```

如果文件已经存在，只需将下面的对象加入现有 `plugins[]`，不要覆盖其他插件：

```json
{
  "name": "vocal-reference-comparator",
  "source": {
    "source": "local",
    "path": "./.codex/plugins/vocal-reference-comparator"
  },
  "policy": {
    "installation": "AVAILABLE",
    "authentication": "ON_INSTALL"
  },
  "category": "Creative"
}
```

如果还没有个人市场文件，可以使用以下完整结构：

```json
{
  "name": "personal",
  "interface": {
    "displayName": "Personal"
  },
  "plugins": [
    {
      "name": "vocal-reference-comparator",
      "source": {
        "source": "local",
        "path": "./.codex/plugins/vocal-reference-comparator"
      },
      "policy": {
        "installation": "AVAILABLE",
        "authentication": "ON_INSTALL"
      },
      "category": "Creative"
    }
  ]
}
```

### 3. 安装插件

完全退出并重新打开 Codex，然后从 Plugins Directory 的 `Personal` 来源安装；也可以使用 CLI：

```powershell
codex plugin add vocal-reference-comparator@personal
```

安装或更新后建议新建 Codex 任务，确保新任务加载最新 Skill。

## 在 Codex 中使用

向 Codex 提供一条待评估人声和一个或多个参考文件，例如：

```text
请用 Vocal Reference Comparator 分析：
待评估人声：C:\Audio\my-vocal.wav
参考 A：C:\Audio\reference-a.wav
参考 B：C:\Audio\reference-b.wav
其中参考 B 是 Source Separation 文件。请进行 full 分析并给出优先调整方案。
```

也可以直接询问某个问题：

```text
判断我的 Doubler 为什么听起来像左、中、右三个点，而参考人声为什么像连续扩散区域。
```

### 默认输出路径与对话内报告

如果没有指定输出路径，插件会在运行前明确告知实际保存位置。默认情况下，若 `C:\Projects\work` 存在且可写，会在其中创建本次任务的独立子目录；否则使用当前任务可写的独立输出目录。

分析完成后，插件会先把最终报告的完整正文、表格、结论和主要图片直接呈现在 Codex 回复中，再列出 `reference-set-report.md` 等输出文件的绝对路径。因此没有合适的 Markdown 查看器也能直接阅读报告。

## 直接运行分析脚本

插件以 `compare_reference_set.py` 作为统一 Orchestrator。先进入 Skill 目录：

```powershell
Set-Location ".\skills\compare-vocal-references"
```

### 安装依赖

推荐将依赖安装到独立目录：

```powershell
python .\scripts\bootstrap_deps.py --target "C:\Projects\work\vocal-reference-deps"
$env:PYTHONPATH = "C:\Projects\work\vocal-reference-deps"
```

依赖包括：

- `numpy`
- `scipy`
- `soundfile`
- `matplotlib`
- `pyloudnorm`
- `librosa`

### 单参考

```powershell
python .\scripts\compare_reference_set.py `
  --target "My Vocal=C:\Audio\my-vocal.wav" `
  --reference "Reference A=C:\Audio\reference-a.wav" `
  --analysis-level full `
  --out-dir "C:\Projects\work\vocal-comparison"
```

### 多参考与分离参考

```powershell
python .\scripts\compare_reference_set.py `
  --target "My Vocal=C:\Audio\my-vocal.wav" `
  --reference "Reference A=C:\Audio\reference-a.wav" `
  --reference "Reference B=C:\Audio\reference-b.wav" `
  --separated-reference "Reference B" `
  --analysis-level full `
  --out-dir "C:\Projects\work\vocal-reference-set"
```

`--reference` 可以重复。`--separated-reference` 的标签必须与对应 `--reference` 标签完全一致。

## Analysis Level

| Value | 内容 |
| --- | --- |
| `basic` | 电平、基础音色、动态、纹理和基础空间指标 |
| `detail` | `basic` 加动态频段、咬字、宏微动态、谐波和颗粒诊断 |
| `full` | `detail` 加高级空间分布、相干度、延迟峰和 Side 尾部诊断 |

完整报告默认使用 `full`。

## 输出文件

统一 Orchestrator 会生成：

```text
output/
├── reference-set-metrics.json
├── reference-set-report.md
├── R1_levels_and_timbre.png
├── R2_dynamics_and_texture.png
├── R3_spatial_reference_field.png
├── R4_consensus_and_outliers.png
├── execution-log.json
└── pairwise/
    └── <reference>/
        ├── basic/
        ├── detail/
        └── spatial/
```

其中：

- `reference-set-report.md`：面向实际混音决策的中文汇总报告；
- `reference-set-metrics.json`：可复用的结构化测量结果；
- `R1–R4`：多参考总览图；
- `pairwise/`：待评估人声与每个参考的一对一完整证据；
- `execution-log.json`：实际执行的命令和结果，便于复现与排错。

## 可信度与解释边界

每个独立子图使用以下可信度等级：

- `高`
- `中高`
- `中`
- `中低`
- `低`
- `证据不足`

报告把结论分为三层：

1. **测量结果**：由图表或指标直接支持。
2. **强推断**：多项相互独立的证据指向同一处理方向。
3. **假设**：合理，但无法仅凭最终渲染音频确认。

请注意：

- 启发式指标不能证明使用了某个具体商业插件。
- Source Separation 可能制造 Musical Noise、瞬态涂抹、虚假宽度和相位残留。
- 不同歌手、音域、话筒、编曲和母带处理会降低直接可比性。
- 音色判断前应先控制响度，避免把“更响”误判为“更好”。
- 不应把不同歌手之间的频段差值直接照抄成 Static EQ 增益。

## 项目结构

```text
vocal-reference-comparator/
├── .codex-plugin/
│   └── plugin.json
├── skills/
│   └── compare-vocal-references/
│       ├── SKILL.md
│       ├── agents/
│       ├── references/
│       └── scripts/
└── README.md
```

- `.codex-plugin/plugin.json`：Plugin Manifest 和 Codex 展示信息。
- `SKILL.md`：触发条件、分析工作流和交付规则。
- `references/`：测量定义、因果边界、报告规范和空间诊断方法。
- `scripts/`：确定性音频分析、可视化和多参考汇总脚本。

## 更新本地安装

拉取新版本后重新安装插件：

```powershell
git -C "$env:USERPROFILE\.codex\plugins\vocal-reference-comparator" pull
codex plugin add vocal-reference-comparator@personal
```

随后重启 Codex，并在新任务中验证更新。

## 当前限制

- 不能从成品音频准确反推出具体 Plugin、Preset 或完整参数。
- 不同演唱内容通常只能比较统计分布，不能进行逐采样 Null Test。
- 少量参考不足以建立稳定的风格共性；少于三个参考时应谨慎解释 Consensus。
- 音高平台、Vibrato 和 Formant Proxy 仅作为低可信辅助项。
- 自动报告仍需要结合监听、等响度 A/B Test 和实际工程上下文进行判断。
