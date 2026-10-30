#!/bin/bash
# Resume of pipeline_d2_sft.sh after its wrapper is stopped (10-30, so that the
# instance can stay on for RL). Waits for a running training process if given,
# then runs only what is missing for the D10 ablation (unfiltered at the
# selected lr), writes the summary, commits and pushes.
# Shuts down only if SHUTDOWN=1 *and* /root/autodl-tmp/KEEP_ALIVE does not
# exist, so the decision can still be changed while it runs.
#
#   WAIT_PID=<pid of scripts.sft> STEPS=512 SKIP=true D=1030 SHUTDOWN=1 \
#     nohup setsid bash code/scripts/pipeline_d2_sft_resume.sh > /dev/null 2>&1 &

set -uo pipefail
CODE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "${CODE_DIR}/.." && pwd)"
RUNS="${REPO_ROOT}/runs"
CKPT_ROOT=/root/autodl-tmp/sft_ckpt
STEPS="${STEPS:?}"; SKIP="${SKIP:?}"; D="${D:?}"
EPOCHS="${EPOCHS:-5}"
SHUTDOWN="${SHUTDOWN:-0}"
KEEP_ALIVE=/root/autodl-tmp/KEEP_ALIVE
LOG="${RUNS}/${D}-pipeline-d2-sft.log"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=8
exec > >(tee -a "${LOG}") 2>&1
cd "${CODE_DIR}"
echo "=== $(date -Is) resume start (wrapper of pipeline_d2_sft.sh was stopped)"

if [[ -n "${WAIT_PID:-}" ]]; then
  while kill -0 "${WAIT_PID}" 2>/dev/null; do sleep 10; done
  echo "=== $(date -Is) process ${WAIT_PID} finished"
fi

SEL="${RUNS}/${D}-sft-lr-select.json"
LR_SEL=$(python -c "import json; print(json.load(open('${SEL}'))['lr'])")
name="${D}-sft-unfiltered-lr${LR_SEL}"
if [[ ! -e "${RUNS}/${name}/metrics.json" ]]; then
  echo "=== ${name} has no metrics.json (training did not finish); retraining"
  mkdir -p "${RUNS}/${name}"
  timeout 7200 python -u -m scripts.sft --run_name "${name}" --select unfiltered \
    --lr "${LR_SEL}" --epochs "${EPOCHS}" > "${RUNS}/${name}/stdout.txt" 2>&1 || \
    echo "=== train exit=$? (an incomplete earlier attempt may block it; see stdout.txt)"
fi

for ep in $(seq 1 "${EPOCHS}"); do
  for T in 1.0 0.1; do
    run="${D}-eval-sft-unfiltered-lr${LR_SEL}-e${ep}-s${STEPS}-T${T}"
    if [[ -e "${RUNS}/${run}/metrics.json" ]]; then echo "=== skip ${run} (done)"; continue; fi
    ck="${CKPT_ROOT}/${name}/epoch${ep}.ckpt"
    if [[ ! -e "${ck}" ]]; then echo "=== missing ${ck}, skip ${run}"; continue; fi
    rm -rf "${RUNS:?}/${run}"   # leftovers of an interrupted attempt
    echo "=== $(date -Is) start ${run}"
    SPLIT=dev SAMPLER=fast_ancestral STEPS="${STEPS}" TEMP="${T}" BATCH=200 \
      CKPT_PATH="${ck}" RUN_NAME="${run}" \
      timeout 3600 bash scripts/eval_gsm8k.sh sampler.skip_unchanged="${SKIP}" \
      > /dev/null 2>&1
    echo "=== $(date -Is) end ${run} exit=$?"
  done
done

python scripts/summarize_sft.py --prefix "${D}" --steps "${STEPS}" \
  --select_data v1b > "${RUNS}/${D}-D2-sft-summary.md" 2>&1
cat "${RUNS}/${D}-D2-sft-summary.md"
cd "${REPO_ROOT}"
git add runs
git commit -q -m "runs: D2 SFT 学习率扫描与 D10 消融（dev 1000，${STEPS} 步，T ∈ {1, 0.1}）

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
for i in 1 2 3; do timeout 120 git push -q origin main && break; sleep 30; done
echo "=== $(date -Is) pipeline done"

if [[ "${SHUTDOWN}" == 1 && ! -e "${KEEP_ALIVE}" ]]; then
  sync
  /usr/bin/shutdown
else
  echo "=== no shutdown (SHUTDOWN=${SHUTDOWN}, KEEP_ALIVE $( [[ -e ${KEEP_ALIVE} ]] && echo present || echo absent ))"
fi
