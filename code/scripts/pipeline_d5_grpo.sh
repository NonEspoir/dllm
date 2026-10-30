#!/bin/bash
# D5 formal GRPO run (D32): base -> GRPO, two arms that differ only in the
# reward (binary vs format reward 0.1), both without group-std normalisation,
# lr 3e-5, 4 updates per rollout batch, 300 steps, dev200 eval every 25 steps.
# Then per arm: pick the checkpoint by the fixed rule (scripts/select_grpo.py),
# evaluate it on dev1000 at both temperatures, report against base (full dev
# and the last 800 questions, which were not used for selection).
# Summary, commit, push; shuts down unless /root/autodl-tmp/NO_SHUTDOWN exists.
#
#   WAIT_PID=<pid or empty> D=1031 SHUTDOWN=1 nohup setsid bash code/scripts/pipeline_d5_grpo.sh &

set -uo pipefail
CODE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "${CODE_DIR}/.." && pwd)"
RUNS="${REPO_ROOT}/runs"
D="${D:?}"
SHUTDOWN="${SHUTDOWN:-0}"
STEPS="${STEPS:-300}"
LOG="${RUNS}/${D}-pipeline-d5-grpo.log"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
exec > >(tee -a "${LOG}") 2>&1
cd "${CODE_DIR}"
echo "=== $(date -Is) pipeline start: STEPS=${STEPS} SHUTDOWN=${SHUTDOWN}"

if [[ -n "${WAIT_PID:-}" ]]; then
  while kill -0 "${WAIT_PID}" 2>/dev/null; do sleep 15; done
  echo "=== $(date -Is) process ${WAIT_PID} finished"
fi

COMMON=(--lr 3e-5 --updates 4 --no_std --steps "${STEPS}" --eval_every 25
        --keep_ckpts best_last)
ARMS=("${D}-grpo-base-d5-binary" "${D}-grpo-base-d5-fmt0.1")
ARGS=("" "--format_reward 0.1")

for i in 0 1; do
  name="${ARMS[$i]}"
  echo "=== $(date -Is) start ${name}"
  mkdir -p "${RUNS}/${name}"
  # shellcheck disable=SC2086
  timeout 28800 python -u -m scripts.grpo --run_name "${name}" "${COMMON[@]}" ${ARGS[$i]} \
    > "${RUNS}/${name}/stdout.txt" 2>&1
  echo "=== $(date -Is) end ${name} exit=$?"
  grep '^EVAL' "${RUNS}/${name}/stdout.txt"
done

for name in "${ARMS[@]}"; do
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
  python scripts/summarize_grpo.py "${RUNS}/${ARMS[0]}" "${RUNS}/${ARMS[1]}"
  echo
  echo "## Selected checkpoints on dev1000 (selection on dev200 = first 200; last 800 unbiased)"
  echo
  python scripts/select_grpo.py report "${RUNS}/${ARMS[0]}" "${RUNS}/${ARMS[1]}"
} > "${RUNS}/${D}-D5-grpo-summary.md" 2>&1
cat "${RUNS}/${D}-D5-grpo-summary.md"

cd "${REPO_ROOT}"
git add runs
git commit -q -m "runs: D5 正式 GRPO（二值 vs 格式分 0.1，lr 3e-5，每批 4 次更新，${STEPS} 步）与 dev1000 确认

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
for i in 1 2 3; do timeout 120 git push -q origin main && break; sleep 30; done
echo "=== $(date -Is) pipeline done"

rm -f /root/autodl-tmp/KEEP_ALIVE
if [[ "${SHUTDOWN}" == 1 && ! -e /root/autodl-tmp/NO_SHUTDOWN ]]; then
  sync
  /usr/bin/shutdown
else
  echo "=== no shutdown (SHUTDOWN=${SHUTDOWN}, NO_SHUTDOWN $( [[ -e /root/autodl-tmp/NO_SHUTDOWN ]] && echo present || echo absent ))"
fi
