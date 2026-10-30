# 13-gram 污染检查：TinyGSM ↔ GSM8K（10-30）

脚本 `code/scripts/contamination.py`，结果 `metrics.json`，日志 `../1030-contamination-13gram.log`。CPU 运行，约 95 秒（17 进程）。

## 方法

- 预训练数据：TinyGSM 全量 17 个分片，11,846,109 条（hf-mirror 下载，分片 0 与之前单独下载的逐字节相同）。`question` 和 `code` 两个字段都查（代码的 docstring 里会抄题目）。
- GSM8K：test 1319、dev 1000、train 其余 6473，按 D08 的固定划分。
- 归一化：转小写，弯撇号换成直撇号，只保留 `[a-z0-9]+` 的词；取连续 13 个词作 13-gram。
- 为什么用 13-gram：这是常见的污染检查做法（GPT-3 论文用的就是 13-gram）；13 个词足够长，偶然共用的套话（"how much money does she have left"）一般不到这个长度，命中基本说明有长片段重合。
- 两个指标：
  1. 题目中**任一** 13-gram 在 TinyGSM 任何一条里出现过；
  2. 单条 TinyGSM 覆盖这道题 13-gram 的**最大比例**（只看最接近的那一条），≥0.5 记为近重复。

## 结果

| GSM8K | 题数 | 含任一 13-gram | 近重复（≥0.5） | 覆盖 ≥0.8 | 平均最大覆盖 |
|---|---|---|---|---|---|
| test | 1319 | 29（2.2%） | 1（0.08%） | 0 | 0.3% |
| dev | 1000 | 553（55.3%） | 130（13.0%） | 64（6.4%） | 18.0% |
| train 其余 | 6473 | 3588（55.5%） | 653（10.1%） | 344（5.3%） | 16.1% |

- **test 基本干净**。覆盖最高的一题（gsm8k_index 602，0.54）只是同一模板的变体：GSM8K "A plane travels 1200 miles in 3 hours… additional 2000 miles"，TinyGSM "A car travels 120 miles in 3 hours… additional 240 miles"，数字和对象都不同。没有原文拷贝。
- **dev 和 train 一样被大量覆盖**，而且有**原文拷贝**：例如 dev 4247（Patricia 剪辑篮球视频）在 TinyGSM 里一字不差；dev 6330 出现在一条带生成提示残留的 TinyGSM 题里（"Current question and program provided as a reference… Current question: Ezekiel hikes…"）。这和 TinyGSM 以 GSM8K train 为种子生成一致（04 §3.5），说明生成时有一部分直接抄了种子题。

## 对 base 准确率的影响（dev）

用 D1 扫描里同一题在 5 个步数（64–1024）上的平均对错，比较近重复的 130 题和其余 870 题（bootstrap 4000 次）：

| 温度 | 近重复 | 其余 | 差 | 95% CI |
|---|---|---|---|---|
| T=1 | 18.2% | 14.7% | +3.5pp | [-1.0, +8.0] |
| T=0.1 | 39.8% | 36.3% | +3.6pp | [-2.5, +10.0] |

- 近重复题略高，但不显著。即使取上限 +8–10pp，这部分只占 dev 的 13%，对 dev 总准确率的影响也约 1pp 以内。
- 和 D1 观察一致：1024 步 T=1 下 dev 17.5% vs test 15.9%，差 1.6pp，不显著。
- 无法区分"见过所以会"和"这些题本来就更简单"，所以 +3.5pp 只是上界性质的估计。

## 结论（写进报告）

1. 最终评测用的 test 与预训练数据没有实质重合（1 道模板变体，无拷贝），test 上的数字不受预训练污染影响。
2. dev 有 13% 的题在预训练数据里有近重复（6.4% 覆盖 ≥80%，有原文拷贝）。dev 只用于相对比较（D08），对 base 的绝对准确率影响估计在 1pp 以内。
3. SFT / RL 用的 GSM8K train 同样被覆盖（10% 近重复），这不构成污染（训练集本来就可以见），但意味着 SFT / RL 的部分题目模型在预训练时见过变体或原题。
