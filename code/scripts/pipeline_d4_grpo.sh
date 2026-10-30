#!/bin/bash
# D4 GRPO debug runs (D31): base -> GRPO, 3 learning rates x 50 steps, each
# evaluated on dev200 every 10 steps (512 steps + skip, T in {1, 0.1}).
# Waits for a running process first (the D2 resume pipeline) so the two do not
# share the GPU. Then summary, commit, push; shuts down only if SHUTDOWN=1 and
# /root/autodl-tmp/KEEP_ALIVE does not exist.
#
#   WAIT_PID=<pid> D=1030 SHUTDOWN=1 nohup setsid bash code/scripts/pipeline_d4_grpo.sh &

set -uo pipefail
CODE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "${CODE_DIR}/.." && pwd)"
RUNS="${REPO_ROOT}/runs"
D="${D:-$(date +%m%d)}"
LRS="${LRS:-1e-6 3e-6 1e-5}"
SHUTDOWN="${SHUTDOWN:-0}"
KEEP_ALIVE=/root/autodl-tmp/KEEP_ALIVE
LOG="${RUNS}/${D}-pipeline-d4-grpo.log"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
exec > >(tee -a "${LOG}") 2>&1
cd "${CODE_DIR}"
echo "=== $(date -Is) pipeline start: LRS='${LRS}' SHUTDOWN=${SHUTDOWN}"

if [[ -n "${WAIT_PID:-}" ]]; then
  while kill -0 "${WAIT_PID}" 2>/dev/null; do sleep 15; done
  echo "=== $(date -Is) process ${WAIT_PID} finished"
fi

for LR in ${LRS}; do
  name="${D}-grpo-base-lr${LR}"
  echo "=== $(date -Is) start ${name}"
  mkdir -p "${RUNS}/${name}"
  timeout 10800 python -u -m scripts.grpo --run_name "${name}" --lr "${LR}" \
    > "${RUNS}/${name}/stdout.txt" 2>&1
  echo "=== $(date -Is) end ${name} exit=$?"
  grep '^EVAL' "${RUNS}/${name}/stdout.txt"
done

dirs=()
for LR in ${LRS}; do dirs+=("${RUNS}/${D}-grpo-base-lr${LR}"); done
python scripts/summarize_grpo.py "${dirs[@]}" > "${RUNS}/${D}-D4-grpo-summary.md" 2>&1
cat "${RUNS}/${D}-D4-grpo-summary.md"

cd "${REPO_ROOT}"
git add runs
git commit -q -m "runs: D4 GRPO 调试（base 起点，lr ${LRS// / / }，各 50 步，dev200 每 10 步评测）

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
for i in 1 2 3; do timeout 120 git push -q origin main && break; sleep 30; done
echo "=== $(date -Is) pipeline done"

if [[ "${SHUTDOWN}" == 1 && ! -e "${KEEP_ALIVE}" ]]; then
  sync
  /usr/bin/shutdown
else
  echo "=== no shutdown (SHUTDOWN=${SHUTDOWN}, KEEP_ALIVE $( [[ -e ${KEEP_ALIVE} ]] && echo present || echo absent ))"
fi
