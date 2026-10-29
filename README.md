# DLM Take-home：TinyGSM 上 MDLM 的 SFT 与 RL

在 TinyGSM 上预训练好的掩码扩散语言模型 MDLM（170M）基础上做 SFT 和 RL，用 GSM8K 评测，对比原始、SFT 后、RL 后的模型。

> 项目进行中。运行说明会随代码完成逐步补充。

## 目录

| 路径 | 内容 |
|---|---|
| `docs/` | 题目、任务分析、概念讲解、决策记录、RL 设计、开销预算、进度 |
| `code/` | 代码。第一次提交是上游 [jdeschena/s-flm](https://github.com/jdeschena/s-flm) 的原样源码（commit `30def79`，Apache-2.0），之后的改动都在后续提交里 |
| `data/splits/` | 固定的数据划分（GSM8K train → train / dev） |
| `runs/` | 每次实验的配置、指标和日志 |
| `CLAUDE.md` | 给 Claude Code 的项目规则 |

## 从哪里开始读

1. [`docs/06-进度.md`](docs/06-进度.md)：当前进度和下一步
2. [`docs/00-任务分析.md`](docs/00-任务分析.md)：总体分析与方案
3. [`docs/02-决策记录.md`](docs/02-决策记录.md)：所有决策及其理由

## 致谢

模型 checkpoint 和基础代码来自 Deschenaux & Gulcehre, *Language Modeling with Hyperspherical Flows*（arXiv 2605.11125），[代码](https://github.com/jdeschena/s-flm)，[checkpoint](https://huggingface.co/jdeschena/s-flm)。
