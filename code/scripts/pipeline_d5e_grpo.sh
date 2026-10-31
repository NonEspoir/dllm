#!/bin/bash
# D5 round 3 (R43), two arms in sequence, both with the format-reward arm's
# settings (lr 3e-5, 4 updates per batch, no std, format reward 0.1, no KL,
# brake at 2x the starting log-prob):
#   1. SFT -> RL (D13 control): start from the SFT checkpoint ${SFT_CKPT}
#      (lr 1e-5, epoch ${SFT_EPOCH}), 150 steps, same as the base-start format arm;
#   2. base -> RL, format arm run to 300 steps: does the format arm diverge
#      after step 150 (08 Q42)? Its first 150 steps also replicate the D5
#      format arm (same seed; GPU kernels are not bit-deterministic).
#      Skipped if /root/autodl-tmp/SKIP_ARM2 exists when arm 1 ends.
# Each arm: pick the checkpoint by the fixed rule, confirm on dev1000. Then
# summary, commit, push, shut down unless /root/autodl-tmp/NO_SHUTDOWN exists.
#
#   D=1031 SFT_EPOCH=1 SHUTDOWN=1 nohup setsid bash code/scripts/pipeline_d5e_grpo.sh &

set -uo pipefail
CODE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "${CODE_DIR}/.." && pwd)"
RUNS="${REPO_ROOT}/runs"
D="${D:?}"
SFT_EPOCH="${SFT_EPOCH:?}"
SHUTDOWN="${SHUTDOWN:-0}"
SFT_CKPT="/root/autodl-tmp/sft_ckpt/1030-sft-v1b-lr1e-5/epoch${SFT_EPOCH}.ckpt"
ARM1="${D}-grpo-sft-e${SFT_EPOCH}-fmt0.1"
ARM2="${D}-grpo-base-fmt0.1-s300"
LOG="${RUNS}/${D}-pipeline-d5e-grpo.log"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
exec > >(tee -a "${LOG}") 2>&1
cd "${CODE_DIR}"
echo "=== $(date -Is) pipeline d5e start: ${ARM1} (init ${SFT_CKPT}), then ${ARM2}"
test -f "${SFT_CKPT}" || { echo "missing ${SFT_CKPT}"; exit 1; }

confirm() {  # name
  local name="$1" step ckpt
  grep -E '^EVAL|^STOP' "${RUNS}/${name}/stdout.txt"
  if read -r step ckpt < <(python scripts/select_grpo.py pick "${RUNS}/${name}"); then
    echo "=== ${name}: selected step ${step} (${ckpt})"
    for T in 1.0 0.1; do
      run="${name}-best-s${step}-dev-e512-T${T}"
      echo "=== $(date -Is) start ${run}"
      SPLIT=dev SAMPLER=fast_ancestral STEPS=512 TEMP="${T}" BATCH=200 \
        CKPT_PATH="${ckpt}" RUN_NAME="${run}" \
        timeout 3600 bash scripts/eval_gsm8k.sh sampler.skip_unchanged=true > /dev/null 2>&1
      echo "=== $(date -Is) end ${run} exit=$?"
    done
  else
    echo "=== ${name}: no checkpoint to select"
  fi
}

echo "=== $(date -Is) start ${ARM1}"
mkdir -p "${RUNS}/${ARM1}"
timeout 21600 python -u -m scripts.grpo --run_name "${ARM1}" --init_ckpt "${SFT_CKPT}" \
  --lr 3e-5 --updates 4 --no_std --format_reward 0.1 --steps 150 --eval_every 25 \
  --keep_ckpts best_last --stop_ratio 2.0 --stop_window 10 \
  > "${RUNS}/${ARM1}/stdout.txt" 2>&1
echo "=== $(date -Is) end ${ARM1} exit=$?"
confirm "${ARM1}"
DONE=("${RUNS}/${ARM1}")

if [[ -e /root/autodl-tmp/SKIP_ARM2 ]]; then
  echo "=== SKIP_ARM2 present: ${ARM2} not run"
else
  echo "=== $(date -Is) start ${ARM2}"
  mkdir -p "${RUNS}/${ARM2}"
  timeout 36000 python -u -m scripts.grpo --run_name "${ARM2}" \
    --lr 3e-5 --updates 4 --no_std --format_reward 0.1 --steps 300 --eval_every 25 \
    --keep_ckpts best_last --stop_ratio 2.0 --stop_window 10 \
    > "${RUNS}/${ARM2}/stdout.txt" 2>&1
  echo "=== $(date -Is) end ${ARM2} exit=$?"
  confirm "${ARM2}"
  DONE+=("${RUNS}/${ARM2}")
fi

{
  python scripts/summarize_grpo.py "${DONE[@]}"
  echo
  echo "## Selected checkpoints on dev1000 (selection on dev200 = first 200; last 800 unbiased within each arm)"
  echo
  python scripts/select_grpo.py report "${RUNS}/1030-grpo-base-d5-binary" \
    "${RUNS}/1030-grpo-base-d5-fmt0.1" "${RUNS}/1031-grpo-base-fmt0.1-kl0.04" "${DONE[@]}"
} > "${RUNS}/${D}-D5e-grpo-summary.md" 2>&1
cat "${RUNS}/${D}-D5e-grpo-summary.md"

cd "${REPO_ROOT}"
git add -A runs
git commit -q -m "runs: D5 第三轮 GRPO（SFT e${SFT_EPOCH} → RL 格式分 150 步；base 格式分臂 300 步）与 dev1000 确认

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
for i in 1 2 3; do timeout 120 git push -q origin main && break; sleep 30; done
echo "=== $(date -Is) pipeline done"

if [[ "${SHUTDOWN}" == 1 && ! -e /root/autodl-tmp/NO_SHUTDOWN ]]; then
  sync
  /usr/bin/shutdown
else
  echo "=== no shutdown (SHUTDOWN=${SHUTDOWN}, NO_SHUTDOWN $( [[ -e /root/autodl-tmp/NO_SHUTDOWN ]] && echo present || echo absent ))"
fi
