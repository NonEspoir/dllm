#!/bin/bash
# D5 round 2 (R41): format reward + KL to the init checkpoint (beta = 0.04),
# 300 steps with the divergence brake. Same as the D5 format arm except the KL
# term. Then pick the checkpoint by the fixed rule, confirm on dev1000 (full and
# the last 800 questions), write the summary, commit, push, shut down unless
# /root/autodl-tmp/NO_SHUTDOWN exists.
#
#   D=1031 SHUTDOWN=1 nohup setsid bash code/scripts/pipeline_d5d_grpo_kl.sh &

set -uo pipefail
CODE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "${CODE_DIR}/.." && pwd)"
RUNS="${REPO_ROOT}/runs"
D="${D:?}"
SHUTDOWN="${SHUTDOWN:-0}"
STEPS="${STEPS:-300}"
BETA="${BETA:-0.04}"
NAME="${D}-grpo-base-fmt0.1-kl0.04"
LOG="${RUNS}/${D}-pipeline-d5d-grpo-kl.log"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
exec > >(tee -a "${LOG}") 2>&1
cd "${CODE_DIR}"
echo "=== $(date -Is) pipeline d5d start: ${NAME}, ${STEPS} steps, kl ${BETA}"

echo "=== $(date -Is) start ${NAME}"
mkdir -p "${RUNS}/${NAME}"
timeout 36000 python -u -m scripts.grpo --run_name "${NAME}" --lr 3e-5 --updates 4 \
  --no_std --format_reward 0.1 --kl "${BETA}" --steps "${STEPS}" --eval_every 25 \
  --keep_ckpts best_last --stop_ratio 2.0 --stop_window 10 \
  > "${RUNS}/${NAME}/stdout.txt" 2>&1
echo "=== $(date -Is) end ${NAME} exit=$?"
grep -E '^EVAL|^STOP' "${RUNS}/${NAME}/stdout.txt"

if read -r step ckpt < <(python scripts/select_grpo.py pick "${RUNS}/${NAME}"); then
  echo "=== ${NAME}: selected step ${step} (${ckpt})"
  for T in 1.0 0.1; do
    run="${NAME}-best-s${step}-dev-e512-T${T}"
    echo "=== $(date -Is) start ${run}"
    SPLIT=dev SAMPLER=fast_ancestral STEPS=512 TEMP="${T}" BATCH=200 \
      CKPT_PATH="${ckpt}" RUN_NAME="${run}" \
      timeout 3600 bash scripts/eval_gsm8k.sh sampler.skip_unchanged=true > /dev/null 2>&1
    echo "=== $(date -Is) end ${run} exit=$?"
  done
else
  echo "=== ${NAME}: no checkpoint to select"
fi

{
  python scripts/summarize_grpo.py "${RUNS}/${NAME}"
  echo
  echo "## Selected checkpoint on dev1000 (selection on dev200 = first 200; last 800 unbiased)"
  echo
  python scripts/select_grpo.py report "${RUNS}/${NAME}"
} > "${RUNS}/${D}-D5d-grpo-kl-summary.md" 2>&1
cat "${RUNS}/${D}-D5d-grpo-kl-summary.md"

cd "${REPO_ROOT}"
git add -A runs
git commit -q -m "runs: D5 第二轮 GRPO（格式分 + KL β=0.04，${STEPS} 步）与 dev1000 确认

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
for i in 1 2 3; do timeout 120 git push -q origin main && break; sleep 30; done
echo "=== $(date -Is) pipeline done"

if [[ "${SHUTDOWN}" == 1 && ! -e /root/autodl-tmp/NO_SHUTDOWN ]]; then
  sync
  /usr/bin/shutdown
else
  echo "=== no shutdown (SHUTDOWN=${SHUTDOWN}, NO_SHUTDOWN $( [[ -e /root/autodl-tmp/NO_SHUTDOWN ]] && echo present || echo absent ))"
fi
