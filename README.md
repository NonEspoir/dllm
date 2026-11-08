# DLM Take-home：TinyGSM 上 MDLM 的 SFT 与 RL

在 TinyGSM 上预训练的掩码扩散语言模型 MDLM（170M，上游 s-flm 仓库的 checkpoint）上做 SFT 和 GRPO 强化学习，用 GSM8K 评测，对比原始、SFT 后、RL 后的模型。

## 主要结果（GSM8K test，1319 题，每题一个 seed）

| 模型 | 512 步 / T=1 | 512 步 / T=0.1 | 1024 步 / T=1（论文设置） |
|---|---|---|---|
| 原始 checkpoint | 15.2 | 32.4 | 16.0（论文 18.0） |
| SFT（lr 1e-5，epoch 1） | 15.2 | 33.5 | 15.3 |
| RL，轨迹 log-prob（收敛轮第 400 步） | 27.4 | 39.5 | 29.4 |
| **RL，ELBO log-prob（收敛轮第 400 步，定稿）** | **42.8** | **45.4** | **43.1** |

- SFT 没有提升；RL 显著提升，且超过免费的降温技巧。
- 定稿先是轨迹法，后来 ELBO 对照臂在 dev 上更好，按事先定的规则改为 ELBO，所以 test 被两个 RL 模型各用过一次。选择只看 dev。
- **口径提醒**：交付后审查发现 ELBO 臂的 loss/KL 存在权重广播放大（约 120 倍，报告 §6.3.1），比较还差着 KL、选点规则和训练/评测步数。准确率数字有效，但不能据此断言"ELBO 估计量本身更优"。
- 完整报告见 [`docs/11-报告.md`](docs/11-报告.md)，决策与理由见 [`docs/02-决策记录.md`](docs/02-决策记录.md)。

## 目录

| 路径 | 内容 |
|---|---|
| `docs/` | 题目、任务分析、决策记录、RL 设计、进度、问答记录、阶段复查、报告；图在 `docs/figures/` |
| `code/` | 上游 [jdeschena/s-flm](https://github.com/jdeschena/s-flm)（commit `30def79`，Apache-2.0）加我们的新增，上游原有代码路径不改 |
| `code/fast_sampler.py` | 新增 `FastAncestralSampler`（配置名 `fast_ancestral`），约 2 倍加速 |
| `code/rl/` | rollout、轨迹 / ELBO log-prob、判分、GRPO loss |
| `code/scripts/` | 评测、SFT、GRPO、汇总、画图、demo、各阶段流水线 |
| `code/tests/` | 我们自己的测试（不要用 `code/test/`，上游 `.gitignore` 会忽略它） |
| `data/splits/` | 固定的 dev 划分 id；`data/sft/` 是 SFT 数据与说明 |
| `runs/` | 每次运行的配置、git commit、指标 JSON、日志和汇总表 |

## 环境

PyTorch 2.8.0 / Python 3.12 / CUDA 12.x，单张 A800 80GB。flash-attn 用预编译包，numpy 固定 1.26.4。详细步骤和踩过的坑见 [`docs/07-服务器环境配置.md`](docs/07-服务器环境配置.md)。

```bash
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1   # torch >= 2.6 加载这些 checkpoint 需要
export HF_ENDPOINT=https://hf-mirror.com    # 按网络情况选择
export OMP_NUM_THREADS=8                    # 有的平台默认 0，libgomp 会报错

huggingface-cli download jdeschena/s-flm tinygsm/mdlm.ckpt --local-dir /root/autodl-tmp/checkpoints
```

以下命令默认在 `code/` 目录下执行。权重、数据缓存放在 `/root/autodl-tmp/`，不进 git。

## 1. 数据划分

```bash
# 先把 GSM8K 下到 --gsm8k_dir（默认 /root/autodl-tmp/hf/gsm8k/main）
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download('openai/gsm8k', repo_type='dataset',
                  allow_patterns=['main/*'], local_dir='/root/autodl-tmp/hf/gsm8k')
EOF
python scripts/prepare_gsm8k.py   # 输出 /root/autodl-tmp/data/gsm8k_{test,dev,dev200,train}.json
```

train 6473 / dev 1000 / test 1319；dev 的 id 固定在 `data/splits/gsm8k_dev_ids.json`，重跑只校验、不覆盖。dev 的前 200 题（dev200）用于 RL 中途选 checkpoint，后 800 题无偏报告。

## 2. 评测

```bash
# 标准协议：新采样器，512 步，skip_unchanged，T=1，batch 200
SPLIT=dev SAMPLER=fast_ancestral STEPS=512 TEMP=1.0 BATCH=200 RUN_NAME=my-eval \
  CKPT_PATH=/root/autodl-tmp/checkpoints/tinygsm/mdlm.ckpt \
  bash scripts/eval_gsm8k.sh sampler.skip_unchanged=true

# 对齐论文：原版采样器，test，1024 步，T=1
SPLIT=test SAMPLER=ancestral STEPS=1024 TEMP=1.0 BATCH=64 RUN_NAME=my-repro \
  bash scripts/eval_gsm8k.sh
```

- `SPLIT` 取 `test`、`dev`、`dev200`；结果写到 `runs/<RUN_NAME>/`：`run_info.txt`（配置、commit、耗时、显存）、`metrics.json`（准确率、CI、逐题对错）、`samples/results.json`（生成内容，不进 git）。
- 画 slides 需要 `pip install python-pptx`。
- 评测 SFT 或 RL 的 checkpoint：设置 `CKPT_PATH`。
- 配对检验（精确 McNemar）：`python scripts/summarize_runs.py compare ../runs/<a> ../runs/<b>`；表格：`summarize_runs.py table ...`。
- 多数投票 / pass@k（同一 checkpoint 用 8 个不同 seed 评测）：`python scripts/majority_vote.py ../runs/<run1> ../runs/<run2> ...`。
- **test 不参与任何选择**：用于复现原始模型一次，以及 base / SFT / 两个 RL 候选（轨迹法原定稿、ELBO 定稿）的最终评测。选步数、温度、checkpoint 都只看 dev。
- 最终评测的完整流水线：`scripts/pipeline_d6_test.sh`（test 三模型 × 三设置，加 dev 上的步数鲁棒性与投票）。

## 3. 采样器等价性测试

```bash
python -m tests.test_fast_sampler --ckpt /root/autodl-tmp/checkpoints/tinygsm/mdlm.ckpt   # 层次 ①：逐字一致
```

层次 ② 统计一致（dev1000，原版 vs 新版）与层次 ③ 耗时对比见 [`runs/1030-D1-summary.md`](runs/1030-D1-summary.md) 和 `scripts/pipeline_d1.sh`。`skip_unchanged` 与默认路径同分布但不逐位相同（TF32 下 cuBLAS 选 kernel 与子 batch 大小有关），所以配对比较都另跑同协议的 base。

## 4. SFT

数据：`data/sft/deepseek_v1.jsonl`（GSM8K train 去掉 dev 后 6473 题，DeepSeek 改写成 TinyGSM 风格 Python，附逐条检查结果；生成与检查见 `scripts/sft_data.py` 头部和 [`data/sft/README.md`](data/sft/README.md)）。

```bash
python -m scripts.sft --run_name my-sft --select sft_v1b --lr 1e-5   # 默认 5 个 epoch，每个 epoch 存一个 checkpoint
```

- 自写 PyTorch 循环，复用上游 `MDLM._loss`，从 EMA 权重出发；原因见 `scripts/sft.py` 头部。
- **分隔符是字面量 `\n`（反斜杠加 n），不是换行**，脚本里有断言。
- 学习率 × epoch 扫描和数据过滤消融：`scripts/pipeline_d2_sft.sh`，汇总见 `runs/1030-D2-sft-summary.md`。

## 5. RL（GRPO）

定稿配置（ELBO log-prob，无 KL，cosine 学习率衰减，400 步）：

```bash
python -u -m scripts.grpo --run_name my-grpo-elbo --logprob elbo \
  --lr 3e-5 --updates 4 --no_std --format_reward 0.1 \
  --spike_ratio 10 --lr_schedule cosine --lr_min_frac 0.1 \
  --steps 400 --eval_every 25 --keep_ckpts last --stop_ratio 2.0 --stop_window 10
```

- 每步 32 题 × G=8，rollout 128 步 T=1；奖励 = 执行正确，另给"能跑但答错"0.1 的格式分；每批 rollout 做 4 次更新（PPO clip，clip-higher）。
- 每 25 步在 dev200 上评测（两温度）；`--stop_ratio` 是 log-prob 下降刹车。checkpoint 存到 `/root/autodl-tmp/rl_ckpt/<run_name>/`。
- 轨迹 log-prob 版本（默认 `--logprob trajectory`，收敛轮另加 `--kl 0.04 --kl_clip 20 --kl_max 10`）：`scripts/pipeline_d5f_conv.sh`。ELBO 版本：`scripts/pipeline_d7_elbo.sh`（150 步）、`scripts/pipeline_d8_elbo_conv.sh`（400 步）。
- ELBO 下不要照搬 KL 系数：日志里的 KL 量级大约 100 倍（其中约 120 倍来自 §6.3.1 的归一化问题），β=0.04 会压住训练（见报告 §6.3.1）。
- 按事先定的规则选 checkpoint：`python scripts/select_grpo.py pick ../runs/<arm> [single|smooth3|last]`。
- 训练曲线转 TensorBoard：`python scripts/runs_to_tb.py <run_name>`。

RL 部分的测试：

```bash
python -m tests.test_rl_logprob      # 轨迹 log-prob：画布还原、与采样器逐位一致（需要 GPU）
python -m tests.test_rl_reward       # 判分与上游评测一致（读已有评测输出）
python -m tests.test_grpo_direction  # 一次更新的方向正确
```

## 6. 图

```bash
python scripts/make_figures.py                   # 全部 8 张，输出到 docs/figures/，不需要 GPU
python scripts/make_figures.py fig_dev200_curves # 只画一张
python scripts/make_slides.py                    # 演示用 slides：docs/slides.pptx（需要 python-pptx）
```

每张图都从 `runs/` 里的指标重新生成。

## 7. Demo

```bash
python -m scripts.demo --split dev --ids 3 17 42 88 137 \
  --ckpts base:/root/autodl-tmp/checkpoints/tinygsm/mdlm.ckpt \
          rl:/root/autodl-tmp/rl_ckpt/1103-grpo-elbo-conv-fmt0.1-nokl/step400.ckpt \
  --temperature 0.1 --out ../runs/demo.json
```

用同一个 seed 和评测同款解码，对每题打印题目、标准答案、各模型的代码、执行结果和判定。题目按固定编号选取，不是挑出来的。已生成的卡片见 [`runs/1103-demo-final.txt`](runs/1103-demo-final.txt)（base 2/5 对，定稿 3/5 对，5 题只作样例）。

## 复现注意

- **每次 GPU 运行都记录在 `runs/<日期>-<名称>/`**，进度与卡时在 [`docs/06-进度.md`](docs/06-进度.md) 的"运行记录"。
- 随机性：同配置同 seed 的两次 RL 在 dev200 上每点差 1–5pp，选中 checkpoint 的 dev1000 差异不显著（报告 §5.4）；只有单个题序 seed，没有多 seed。
- 大文件（`*.ckpt`、数据缓存）不进 git；checkpoint 需要自己重训。

## 阅读顺序

1. [`docs/11-报告.md`](docs/11-报告.md)：结果与分析
2. [`docs/02-决策记录.md`](docs/02-决策记录.md)：所有决策、理由、被更正的判断
3. [`docs/06-进度.md`](docs/06-进度.md)：进度、运行记录与开销
4. [`docs/13-外部审查.md`](docs/13-外部审查.md)：R50 交付前审查的发现与处理
5. [`docs/14-面试口径卡.md`](docs/14-面试口径卡.md)：一页纸"可以说 / 不能说"

## 致谢

模型 checkpoint 和基础代码来自 Deschenaux & Gulcehre, *Language Modeling with Hyperspherical Flows*（arXiv 2605.11125），[代码](https://github.com/jdeschena/s-flm)，[checkpoint](https://huggingface.co/jdeschena/s-flm)。
