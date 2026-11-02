#!/bin/bash
# R45c: one unattended chain, then shut down.
#   1. D6 test evaluation (pipeline_d6_test.sh, SHUTDOWN=0 so this script owns it)
#   2. ELBO control arm (D12): identical to the D5 format arm -- lr 3e-5, 4
#      updates/batch, no std, format reward 0.1, 150 steps, brake -- except
#      --logprob elbo (one fully masked forward per trajectory instead of the
#      exact trajectory probability). Single-variable comparison with
#      runs/1030-grpo-base-d5-fmt0.1.
#   3. dev1000 on the ELBO arm's picked checkpoint (smooth3 rule, D35)
#   4. demo cards for the report
#   5. summary, commit, push, shutdown unless /root/autodl-tmp/NO_SHUTDOWN
#
#   D=1103 SHUTDOWN=1 nohup setsid bash code/scripts/pipeline_d7_elbo.sh &

set -uo pipefail
CODE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "${CODE_DIR}/.." && pwd)"
RUNS="${REPO_ROOT}/runs"
D="${D:?}"
SHUTDOWN="${SHUTDOWN:-0}"
LOG="${RUNS}/${D}-pipeline-d7.log"
RL_CKPT="/root/autodl-tmp/rl_ckpt/1101-grpo-conv-fmt0.1-kl0.04/step400.ckpt"
BASE_CKPT="/root/autodl-tmp/checkpoints/tinygsm/mdlm.ckpt"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
exec > >(tee -a "${LOG}") 2>&1
cd "${CODE_DIR}"
echo "=== $(date -Is) pipeline d7 start"

# 1. D6 (it commits and pushes its own results; SHUTDOWN=0 keeps the machine up)
echo "=== $(date -Is) D6 start"
D="${D}" SHUTDOWN=0 timeout 18000 bash scripts/pipeline_d6_test.sh
echo "=== $(date -Is) D6 end exit=$?"

# 2. ELBO control arm
ELBO="${D}-grpo-elbo-fmt0.1"
echo "=== $(date -Is) start ${ELBO}"
mkdir -p "${RUNS}/${ELBO}"
timeout 18000 python -u -m scripts.grpo --run_name "${ELBO}" \
  --logprob elbo --lr 3e-5 --updates 4 --no_std --format_reward 0.1 \
  --steps 150 --eval_every 25 --keep_ckpts all \
  --stop_ratio 2.0 --stop_window 10 > "${RUNS}/${ELBO}/stdout.txt" 2>&1
echo "=== $(date -Is) end ${ELBO} exit=$?"
grep -E '^EVAL|^STOP' "${RUNS}/${ELBO}/stdout.txt"

if read -r step ckpt < <(python scripts/select_grpo.py pick "${RUNS}/${ELBO}" smooth3); then
  echo "=== ${ELBO}: selected step ${step} (${ckpt})"
  for T in 1.0 0.1; do
    run="${ELBO}-best-s${step}-dev-e512-T${T}"
    echo "=== $(date -Is) start ${run}"
    SPLIT=dev SAMPLER=fast_ancestral STEPS=512 TEMP="${T}" BATCH=200 \
      CKPT_PATH="${ckpt}" RUN_NAME="${run}" \
      timeout 3600 bash scripts/eval_gsm8k.sh sampler.skip_unchanged=true > /dev/null 2>&1
    echo "=== $(date -Is) end ${run} exit=$?"
  done
else
  echo "=== ${ELBO}: no checkpoint to select"
fi

# 3. demo cards: two pairs, fixed ids chosen before looking at the outputs
#    (dev 3 and 17 come from the dev200 dev set; the rest of the ids are spread
#    over dev so the cards are not hand-picked successes)
echo "=== $(date -Is) demo"
python -m scripts.demo --split dev --ids 3 17 42 88 137 \
  --ckpts "base:${BASE_CKPT}" "rl:${RL_CKPT}" \
  --temperature 0.1 --out "${RUNS}/${D}-demo-cards.json" \
  > "${RUNS}/${D}-demo.txt" 2>&1
echo "=== demo exit=$?"
head -40 "${RUNS}/${D}-demo.txt"

# 4. summary
{
  echo "# ${D}: D6 结果、ELBO 对照、demo"
  echo
  echo "## ELBO 对照（D12）"
  echo
  echo "| | 训练时间/步（估） | 训练正确率 | T=1 | T=0.1 |"
  echo "|---|---|---|---|---|"
  python - <<PY
import json, os, sys
sys.path.insert(0, 'scripts')
from summarize_runs import mcnemar
R = os.path.join('${REPO_ROOT}', 'runs')
def load(r):
    return [json.loads(l) for l in open(os.path.join(R, r, 'train_log.jsonl'))]
def mean(xs, k):
    v = [x[k] for x in xs if x.get(k) is not None]
    return sum(v) / len(v) if v else float('nan')
elbo = load('${ELBO}')
fmt = load('1030-grpo-base-d5-fmt0.1')
import statistics
print(f"| ELBO | {statistics.mean(x['train_s'] for x in elbo):.1f} s | "
      f"{mean(elbo[-25:], 'correct'):.3f}（最后 25 步） | (见下) | (见下) |")
print(f"| 轨迹法（D5 格式分臂） | {statistics.mean(x['train_s'] for x in fmt):.4f} s | "
      f"{mean(fmt[-25:], 'correct'):.3f}（最后 25 步） | 31.6 | 46.4 |")
print()
print('| 臂 | T | dev1000 | p vs base | last 800 |')
print('|---|---|---|---|---|')
name = '${ELBO}-best-s${step}'
for T in ('1.0', '0.1'):
    d = os.path.join(R, f'{name}-dev-e512-T{T}', 'metrics.json')
    b = json.load(open(os.path.join(R, f'1030-eval-base-s512-T{T}', 'metrics.json')))['per_question_correct']
    if not os.path.exists(d):
        print(f'| ELBO | {T} | missing | | |'); continue
    pq = json.load(open(d))['per_question_correct']
    n01, n10, p = mcnemar(b, pq)
    print(f'| ELBO（step ${step}） | {T} | {100*pq.count("1")/1000:.1f} | {p:.3g} (+{n01}/-{n10}) '
          f'| {100*pq[200:].count("1")/800:.1f} |')
PY
  echo
  echo "## 其余结果"
  echo
  echo "D6 见 ${D}-D6-test-summary.md；demo 卡片见 ${D}-demo.txt 与 ${D}-demo-cards.json。"
} > "${RUNS}/${D}-D7-summary.md" 2>&1
cat "${RUNS}/${D}-D7-summary.md"

cd "${REPO_ROOT}"
git add -A runs
git commit -q -m "runs: ELBO 对照臂（D12）与 demo 卡片；D6 test 评测

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
for i in 1 2 3; do timeout 120 git push -q origin main && break; sleep 30; done
echo "=== $(date -Is) pipeline done"

if [[ "${SHUTDOWN}" == 1 && ! -e /root/autodl-tmp/NO_SHUTDOWN ]]; then
  sync
  /usr/bin/shutdown
else
  echo "=== no shutdown (SHUTDOWN=${SHUTDOWN}, NO_SHUTDOWN $( [[ -e /root/autodl-tmp/NO_SHUTDOWN ]] && echo present || echo absent ))"
fi
