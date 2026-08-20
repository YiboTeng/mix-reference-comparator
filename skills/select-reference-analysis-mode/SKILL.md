---
name: select-reference-analysis-mode
description: 在用户启动或笼统调用 Mix Reference Comparator、但尚未说明素材类型时，先在“干声/人声参考对比”和“完整混音成品参考对比”之间选择模式。用户已经明确素材类型或直接调用具体分析 skill 时不要重复询问。
---

# 选择参考分析模式

当用户只说“使用 Mix Reference Comparator”“开始分析”或提供的文件类型不明确时，先用一句简短问题让用户选择：

> 请选择分析模式：1）干声/独唱/Vocal Stem 对比；2）包含人声与伴奏的完整混音成品对比。

不要仅凭文件名猜测。用户已经明确选择时直接继续：

- 干声、处理后独唱、人声 Stem 或人声分离文件：使用 `$compare-vocal-references`。
- 包含 Vocal 与 Beats/Instrumental 的完整 Mix/Master：使用 `$compare-mix-references`。

选择后由对应 skill 一次性收集 Target、一个或多个 Reference、来源信息和可选输出目录。未指定输出目录时，先说明将保存到 `C:\Projects\work\<本次任务子目录>`；该目录不可用时再选择其他任务独立目录。

如果用户混合提供了干声和完整 Mix，不要把两类文件放进同一个统计参考集；分别运行两个模式并分别报告。
