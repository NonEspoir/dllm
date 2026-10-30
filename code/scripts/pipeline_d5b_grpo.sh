#!/bin/bash
# D5 resumed (R39). The binary arm of pipeline_d5_grpo.sh diverged after
# about step 150 and was stopped at step 264 (its best checkpoint by the fixed
# rule, step 150, is kept). This runs the format-reward arm with the same
# settings plus a brake and a 150-step cap, then confirms both arms' selected
# checkpoints on dev1000 (full and the last 800 questions), summary, commit,
# push; shuts down unless /root/autodl-tmp/NO_SHUTDOWN exists.
#
#   D=1030 SHUTDOWN=1 nohup setsid bash code/scripts/pipeline_d5b_grpo.sh &

set -uo pipefail
CODE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "${CODE_DIR}/.." && pwd)"
RUNS="${REPO_ROOT}/runs"
D="${D:?}"
SHUTDOWN="${SHUTDOWN:-0}"
LOG="${RUNS}/${D}-pipeline-d5-grpo.log"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
exec > >(tee -a "${LOG}") 2>&1
cd "${CODE_DIR}"
echo "=== $(date -Is) pipeline d5b start (binary arm stopped at step 264 after diverging)"

BIN="${D}-grpo-base-d5-binary"
FMT="${D}-grpo-base-d5-fmt0.1"
echo "=== $(date -Is) start ${FMT}"
mkdir -p "${RUNS}/${FMT}"
timeout 28800 python -u -m scripts.grpo --run_name "${FMT}" --lr 3e-5 --updates 4 \
  --no_std --format_reward 0.1 --steps 150 --eval_every 25 --keep_ckpts best_last \
  --stop_ratio 2.0 --stop_window 10 > "${RUNS}/${FMT}/stdout.txt" 2>&1
echo "=== $(date -Is) end ${FMT} exit=$?"
grep -E '^EVAL|^STOP' "${RUNS}/${FMT}/stdout.txt"

for name in "${BIN}" "${FMT}"; do
  read -r step ckpt < <(python scripts/select_grpo.py pick "${RUNS}/${name}")
  echo "=== ${name}: selected step ${step} (${ckpt})"
  for T in 1.0 0.1; do
    run="${name}-best-s${step}-dev-e512-T${T}"
    echo "=== $(date -Is) start ${run}"
    SPLIT=dev SAMPLER=fast_ancestral STEPS=512 TEMP="${T}" BATCH=200 \
      CKPT_PATH="${ckpt}" RUN_NAME="${run}" \
      timeout 3600 bash scripts/eval_gsm8k.sh sampler.skip_unchanged=true > /dev/null 2>&1
    echo "=== $(date -Is) end ${run} exit=$?"
  done
done

{
  python scripts/summarize_grpo.py "${RUNS}/${BIN}" "${RUNS}/${FMT}"
  echo
  echo "## Selected checkpoints on dev1000 (selection on dev200 = first 200; last 800 unbiased)"
  echo
  python scripts/select_grpo.py report "${RUNS}/${BIN}" "${RUNS}/${FMT}"
} > "${RUNS}/${D}-D5-grpo-summary.md" 2>&1
cat "${RUNS}/${D}-D5-grpo-summary.md"

cd "${REPO_ROOT}"
git add runs
git commit -q -m "runs: D5 正式 GRPO（二值臂第 150 步后发散；格式分臂加刹车、上限 150 步）与 dev1000 确认

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
for i in 1 2 3; do timeout 120 git push -q origin main && break; sleep 30; done
echo "=== $(date -Is) pipeline done"

if [[ "${SHUTDOWN}" == 1 && ! -e /root/autodl-tmp/NO_SHUTDOWN ]]; then
  sync
  /usr/bin/shutdown
else
  echo "=== no shutdown (SHUTDOWN=${SHUTDOWN}, NO_SHUTDOWN $( [[ -e /root/autodl-tmp/NO_SHUTDOWN ]] && echo present || echo absent ))"
fi
