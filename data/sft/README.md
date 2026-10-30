# SFT 数据：DeepSeek 改写的 GSM8K train（v1）

`deepseek_v1.jsonl`：GSM8K train 去掉 dev 后的 6473 题，每题一个由 DeepSeek 写的 TinyGSM 风格 Python 解，附带全部检查结果。

- **生成**：2026-10-30，`deepseek-flash`（DeepSeek-V4.1-Flash），思考模式关，temperature 0.7，每题生成 1 次。
- **规模和开销**：用时 5.8 分钟（32 并发）；输入 268 万 token（其中 160 万命中缓存），输出 103 万 token。花费按 README 里的单价**估计**约 ¥5–10，以 DeepSeek 后台账单为准。
- 生成和检查脚本：[`code/scripts/sft_data.py`](../../code/scripts/sft_data.py)。prompt 也在里面（`SYSTEM`、`USER_TEMPLATE`），用 `python code/scripts/sft_data.py system` 可以打印出来。

## 每行的字段

```json
{
  "gsm8k_index": 0,                 // GSM8K main/train 中的行号（data/splits/gsm8k_dev_ids.json 里的不会出现）
  "question": "...",                // GSM8K 原题
  "code": "def simple_math_problem() -> int: ...",
  "gold_answer": "72",              // GSM8K 标注答案（#### 之后的部分）
  "hard_pass": true,                // 通过全部硬检查 → 进 SFT 主数据
  "suspicious": false,              // 通过硬检查，但触发了软检查 → 可疑，单独做消融
  "checks": {
    "format_ok": true,              // def simple_math_problem() -> int: + docstring + result = ... + return result
    "exec_correct": true,           // 用评测同款 sandbox（code/sandbox_gsm8k.py）执行，返回值 == gold
    "returns_literal": false,       // result 直接（或经变量传递）等于一个数字字面量 → 答案是写进去的，不是算出来的
    "gold_in_body": false,          // gold 数值以字面量出现在代码里，而题目里没有（软）
    "numbers_ok": true,             // 代码里的数字都来自题目或常见换算常数（软）
    "foreign_numbers": [],          // 不满足上一条的数字
    "intermediate_match": 1.0,      // GSM8K 人写解答里 <<…=v>> 的中间值，有多大比例出现在代码的局部变量里（软）
    "num_tokens": 120,              // [BOS] question \n code [EOS] 的 token 数（SmolLM-135M tokenizer）
    "length_ok": true               // ≤ 512，与预训练的长度上限一致
  }
}
```

## 怎么用

| 用途 | 选法 | 条数 |
|---|---|---|
| SFT 主数据（D09 方案 A） | `hard_pass == true` | 6203 |
| D10 消融：不过滤 | `checks.length_ok == true`（只去掉超长的） | 6464 |
| 可选消融：去掉可疑样本 | `hard_pass and not suspicious` | 5770 |

```python
import json
rows = [json.loads(l) for l in open('data/sft/deepseek_v1.jsonl')]
sft = [r for r in rows if r['hard_pass']]
# 训练格式与预训练一致：[BOS] question \n code [EOS]，pad 到 512，只在 code 部分算 loss
```

## 检查结果（6473 题）

| 检查 | 通过 | 未通过 | 说明 |
|---|---|---|---|
| 格式 | 6473 | 0 | |
| 执行结果 == gold | 6213 | **260（4.0%）** | 见下方「被拒样本」 |
| result 不是字面量 | 6468 | 5 | |
| 长度 ≤ 512 | 6464 | 9 | 平均 230 token，p95 355 |
| **硬检查全部通过** | **6203（95.8%）** | 270 | |
| 软：数字都有出处 | 6328 | 145 | 多数是常识数字，见下 |
| 软：中间值与 GSM8K 全部对上 | 4417 / 6390 | — | 有 83 题的 GSM8K 解答里没有 `<<>>` 标注 |
| 通过硬检查但可疑 | — | 433 | 软检查任意一项触发 |

## 这份数据能保证正确吗？

**不能保证"一定"正确。** 理由和对策见 [docs/08-问答记录.md](../../docs/08-问答记录.md) Q11，这里只写实测结果。

### 1. 硬检查能保证什么，不能保证什么

- **能保证**：程序在评测同款 sandbox 里执行，最终数字和 GSM8K 标注一致，格式和长度与预训练数据一致，答案不是直接写成常数的。
- **不能保证**：
  - 推理过程正确：可能错两步、正好抵消；
  - GSM8K 标注本身正确，见下方第 3 点。

### 2. 人工抽查（2026-10-30，Claude 逐条阅读，对照题目判断推理是否正确）

| 样本 | 抽多少 | 推理有误 | 说明 |
|---|---|---|---|
| `hard_pass` 随机抽 | 50 | **0** | 推理错误率 95% 置信上界约 6%（0/50，按 3/n 规则） |
| `suspicious` 随机抽 | 20 | **0** | 都是"拆步方式和 GSM8K 不同"（例如把每周 5 天的工作日写成常数、先算总量再分摊），推理都对 |

抽到的题号存在 `/root/autodl-tmp/sft/review_*_ids.json`，抽样种子是 2026 和 7，可以复现。

**这说明什么**：
- 在约 6% 的分辨率内，没有发现"答案对、推理错"的样本。
- 软检查的精确率很低：抽查的 20 条可疑样本没有一条真有问题。所以它只用来做消融，不用来过滤。
- **这是一个人（Claude）的判断，不是多人标注**，报告里要这样写。

### 3. 被拒的 260 条：多数是 DeepSeek 错了，但也有 GSM8K 标注错的

随机抽 24 条被拒样本，逐条判断：

| 类别 | 条数 | 例子（`gsm8k_index`） |
|---|---|---|
| DeepSeek 推理错 | 19 | 6474：方向反了，算成 A−B，正确是 B−A；6714：没按题意找最优组合；4555：一周按 7 天算，题目说的是上学日；5280：每托鸡蛋编成了 30 个，题目没给，gold 用的是 36 |
| 题目本身有歧义 | 2 | 3489："2 132 contracts" 是 2 份各 132 页，DeepSeek 读成了 2132；4509：studio 住几人，题目没给 |
| **GSM8K 标注错，DeepSeek 对** | **5** | 见下表 |

GSM8K 标注错的 5 条（都手算核对过）：

| `gsm8k_index` | 题意 | GSM8K 的 gold | 正确答案 | GSM8K 错在哪 |
|---|---|---|---|---|
| 7472 | Anika 30 岁，是 Maddie 的 4/3 倍，15 年后平均年龄 | 50 | 41.25 | 把 Maddie 算成 30×4/3 |
| 1081 | 第二个蜂箱的蜂少 20%，每只多产 40% 的蜜 | 2460 | 1060 | 每只产量算成 1000/500，把除数和被除数弄反了 |
| 4099 | 镍占 10%、720 吨，求铜 | 360 | 2160 | 把 720 当成 60% 去算 |
| 5598 | Frances 的毛巾重多少盎司 | 128 | 192 | 用 4 条代替了 6 条 |
| 356 | 答错一题后的总分 | 66（解答正文算出的是 64） | 64 | 正文和 `####` 不一致 |

24 条里有 5 条（约 21%）是 GSM8K 标注错，按比例推到全部 260 条，**估计**被拒的样本里有约 30–80 条其实是 GSM8K 标注错了。换算到整个训练集，约 0.5–1.2% 的 gold 有问题。这个推算基于 24 条小样本，误差很大。

对 SFT 的影响：
- 这些题会因为 DeepSeek 答对、gold 错而被过滤掉，所以**只损失数据，不引入错误**。
- RL 的奖励也用 gold，这些题上奖励信号是错的。量不大，但报告里要提。
- 评测用的 GSM8K test 也会有类似比例的标注错误，所有模型受的影响相同，不影响模型之间的比较。

## 样例

### 通过（`hard_pass`，典型）

```python
# gsm8k_index 6156: A man drives 60 mph for 3 hours. How fast would he have to drive
# over the next 2 hours to get an average speed of 70 mph?   gold 85
def simple_math_problem() -> int:
    """
    A man drives 60 mph for 3 hours.
    How fast would he have to drive over the next 2 hours to get an average speed of 70 mph?
    """
    first_speed = 60
    first_time = 3
    second_time = 2
    target_average_speed = 70
    total_time = first_time + second_time
    first_distance = first_speed * first_time
    total_distance_needed = target_average_speed * total_time
    second_distance_needed = total_distance_needed - first_distance
    second_speed = second_distance_needed / second_time
    result = second_speed
    return result
```

### 可疑但推理正确（`suspicious`，中间值和 GSM8K 只对上 40%）

```python
# gsm8k_index 2919: 2 classes on weekdays, 4 on each weekend day, 5 people per class, 3 weeks.  gold 270
    weekday_classes = 2
    weekend_classes_per_day = 4
    people_per_class = 5
    weeks = 3
    weekday_class_days = 5          # 题目没写"5"，是常识
    weekend_class_days = 2
    classes_per_week = weekday_classes * weekday_class_days + weekend_classes_per_day * weekend_class_days
    total_classes = classes_per_week * weeks
    result = total_classes * people_per_class
    return result
```

GSM8K 的拆法是"每周人数 90 → ×3"，DeepSeek 是"总课数 54 → ×5"。中间值对不上，推理是对的。

### 被拒：DeepSeek 推理错

```python
# gsm8k_index 4555: Maddy eats two eggs each day after school. 40 eggs, how many weeks?  gold 4
    days_per_week = 7               # 错：题目说"after school"，一周 5 个上学日
```

### 被拒：result 是写死的常数

```python
# gsm8k_index 807（Yuan / David 年龄题）gold 7
    age_difference = 7
    # Yuan = David + 7 and Yuan = 2 * David, so David = 7
    david_age = age_difference      # 在注释里心算出答案，再把恰好相等的输入当结果返回
    result = david_age
```

数值对，过程等于没算。这条被 `returns_literal` 抓到，严格说属于"答案对、推理没写出来"。

### 被拒：GSM8K 标注错

```python
# gsm8k_index 1081: 第一个蜂箱 1000 只蜂产 500 升；第二个少 20% 的蜂、每只多产 40%。
# GSM8K gold 2460（每只产量算成 1000/500=2 升）；正确答案 1060
    honey_per_bee_first = first_hive_honey / first_hive_bees            # 0.5
    second_hive_bees = first_hive_bees * (1 - 20 / 100)                 # 800
    honey_per_bee_second = honey_per_bee_first * (1 + 40 / 100)         # 0.7
    total_honey = first_hive_honey + second_hive_bees * honey_per_bee_second   # 1060
```

## 怎么重新生成

见 `code/scripts/sft_data.py` 文件开头的三条命令（tasks → DeepSeek batch → build）。DeepSeek 的输出有随机性，重新生成不会逐字相同，统计数字应该接近。原始 API 输出（含 token 用量）在服务器的 `/root/autodl-tmp/sft/raw.jsonl`，不进 git。
