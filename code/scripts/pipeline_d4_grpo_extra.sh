#!/bin/bash
# D4 extra GRPO runs (R36): reward-design ablations at lr 3e-6, after
# pipeline_d4_grpo.sh (the 3-lr main-line sweep) has finished.
#   1. binary reward, no group-std normalisation (Dr. GRPO)
#   2. format reward 0.1 for code that runs and returns a number, no std norm
# 1 vs 2 differs only in the format reward; 1 vs the main-line 3e-6 run differs
# only in the std normalisation.
# KEEP_ALIVE was created so that the main-line pipeline does not shut down;
# this script removes it at the end and shuts down unless
# /root/autodl-tmp/NO_SHUTDOWN exists.
#
#   WAIT_PID=<pid of pipeline_d4_grpo.sh> D=1030 SHUTDOWN=1 \
#     nohup setsid bash code/scripts/pipeline_d4_grpo_extra.sh &

set -uo pipefail
CODE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "${CODE_DIR}/.." && pwd)"
RUNS="${REPO_ROOT}/runs"
D="${D:?}"
SHUTDOWN="${SHUTDOWN:-0}"
LOG="${RUNS}/${D}-pipeline-d4-grpo.log"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
exec > >(tee -a "${LOG}") 2>&1
cd "${CODE_DIR}"
echo "=== $(date -Is) extra pipeline start (waiting for ${WAIT_PID:-nothing})"

if [[ -n "${WAIT_PID:-}" ]]; then
  while kill -0 "${WAIT_PID}" 2>/dev/null; do sleep 15; done
  echo "=== $(date -Is) process ${WAIT_PID} finished"
fi

run() {  # run <name> <args...>
  local name=$1; shift
  echo "=== $(date -Is) start ${name}"
  mkdir -p "${RUNS}/${name}"
  timeout 10800 python -u -m scripts.grpo --run_name "${name}" "$@" \
    > "${RUNS}/${name}/stdout.txt" 2>&1
  echo "=== $(date -Is) end ${name} exit=$?"
  grep '^EVAL' "${RUNS}/${name}/stdout.txt"
}

run "${D}-grpo-base-lr3e-6-nostd" --lr 3e-6 --no_std
run "${D}-grpo-base-lr3e-6-nostd-fmt0.1" --lr 3e-6 --no_std --format_reward 0.1

dirs=()
for n in lr1e-6 lr3e-6 lr1e-5 lr3e-6-nostd lr3e-6-nostd-fmt0.1; do
  dirs+=("${RUNS}/${D}-grpo-base-${n}")
done
python scripts/summarize_grpo.py "${dirs[@]}" > "${RUNS}/${D}-D4-grpo-summary.md" 2>&1
cat "${RUNS}/${D}-D4-grpo-summary.md"

cd "${REPO_ROOT}"
git add runs
git commit -q -m "runs: D4 GRPO 奖励设计消融（lr 3e-6：去 std 归一化、格式分 0.1）与 5 组汇总

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
for i in 1 2 3; do timeout 120 git push -q origin main && break; sleep 30; done
echo "=== $(date -Is) extra pipeline done"

rm -f /root/autodl-tmp/KEEP_ALIVE
if [[ "${SHUTDOWN}" == 1 && ! -e /root/autodl-tmp/NO_SHUTDOWN ]]; then
  sync
  /usr/bin/shutdown
else
  echo "=== no shutdown (SHUTDOWN=${SHUTDOWN}, NO_SHUTDOWN $( [[ -e /root/autodl-tmp/NO_SHUTDOWN ]] && echo present || echo absent ))"
fi
