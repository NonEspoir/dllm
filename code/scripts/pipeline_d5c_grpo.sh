#!/bin/bash
# D5, second try of the format-reward arm (R39). pipeline_d5b_grpo.sh crashed
# on the arm's first step (variable name clash in the brake code, fixed and
# smoke-tested in 1031-grpo-smoke-brake2) but went on to confirm the binary
# arm's step-150 checkpoint on dev1000. This waits for it, moves the crashed
# run and the junk evals it made for the missing arm aside, runs the arm with
# D33 settings, confirms it on dev1000, writes the summary, commits, pushes and
# shuts down. NO_SHUTDOWN was created only to stop d5b from shutting down; it
# is removed after d5b ends, so creating it again later still works.
#
#   WAIT_PID=<d5b pid> D=1030 SHUTDOWN=1 nohup setsid bash code/scripts/pipeline_d5c_grpo.sh &

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
echo "=== $(date -Is) pipeline d5c start (waiting for ${WAIT_PID:-nothing})"
if [[ -n "${WAIT_PID:-}" ]]; then
  while kill -0 "${WAIT_PID}" 2>/dev/null; do sleep 15; done
  echo "=== $(date -Is) process ${WAIT_PID} finished"
fi
rm -f /root/autodl-tmp/NO_SHUTDOWN

BIN="${D}-grpo-base-d5-binary"
FMT="${D}-grpo-base-d5-fmt0.1"
cd "${REPO_ROOT}"
if [[ -d "runs/${FMT}" ]]; then
  git mv "runs/${FMT}" "runs/${FMT}-crashed" 2>/dev/null || mv "runs/${FMT}" "runs/${FMT}-crashed"
fi
for j in runs/"${FMT}"-best-s-dev-e512-T*; do
  [[ -e "$j" ]] && { git rm -r -q --cached "$j" 2>/dev/null; rm -rf "$j"; echo "=== removed junk eval $j"; }
done
rm -rf "/root/autodl-tmp/rl_ckpt/${FMT}"
cd "${CODE_DIR}"

echo "=== $(date -Is) start ${FMT}"
mkdir -p "${RUNS}/${FMT}"
timeout 28800 python -u -m scripts.grpo --run_name "${FMT}" --lr 3e-5 --updates 4 \
  --no_std --format_reward 0.1 --steps 150 --eval_every 25 --keep_ckpts best_last \
  --stop_ratio 2.0 --stop_window 10 > "${RUNS}/${FMT}/stdout.txt" 2>&1
echo "=== $(date -Is) end ${FMT} exit=$?"
grep -E '^EVAL|^STOP' "${RUNS}/${FMT}/stdout.txt"

if read -r step ckpt < <(python scripts/select_grpo.py pick "${RUNS}/${FMT}"); then
  echo "=== ${FMT}: selected step ${step} (${ckpt})"
  for T in 1.0 0.1; do
    run="${FMT}-best-s${step}-dev-e512-T${T}"
    echo "=== $(date -Is) start ${run}"
    SPLIT=dev SAMPLER=fast_ancestral STEPS=512 TEMP="${T}" BATCH=200 \
      CKPT_PATH="${ckpt}" RUN_NAME="${run}" \
      timeout 3600 bash scripts/eval_gsm8k.sh sampler.skip_unchanged=true > /dev/null 2>&1
    echo "=== $(date -Is) end ${run} exit=$?"
  done
else
  echo "=== ${FMT}: no checkpoint to select"
fi

{
  python scripts/summarize_grpo.py "${RUNS}/${BIN}" "${RUNS}/${FMT}"
  echo
  echo "## Selected checkpoints on dev1000 (selection on dev200 = first 200; last 800 unbiased)"
  echo
  python scripts/select_grpo.py report "${RUNS}/${BIN}" "${RUNS}/${FMT}"
} > "${RUNS}/${D}-D5-grpo-summary.md" 2>&1
cat "${RUNS}/${D}-D5-grpo-summary.md"

cd "${REPO_ROOT}"
git add -A runs
git commit -q -m "runs: D5 格式分臂（D33：同配置 + 刹车，上限 150 步）与两臂 dev1000 确认；首次尝试因刹车代码变量名冲突崩溃，已移到 -crashed

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
for i in 1 2 3; do timeout 120 git push -q origin main && break; sleep 30; done
echo "=== $(date -Is) pipeline done"

if [[ "${SHUTDOWN}" == 1 && ! -e /root/autodl-tmp/NO_SHUTDOWN ]]; then
  sync
  /usr/bin/shutdown
else
  echo "=== no shutdown (SHUTDOWN=${SHUTDOWN}, NO_SHUTDOWN $( [[ -e /root/autodl-tmp/NO_SHUTDOWN ]] && echo present || echo absent ))"
fi
