#!/bin/bash
# D2 SFT sweep (docs/06 "下一步" 4-5), unattended:
#   1. lr sweep on sft_v1b: lr in LRS, 5 epochs; every epoch checkpoint is
#      evaluated on dev 1000 at STEPS (D14) x T in {1.0, 0.1}, fast sampler,
#      batch 200, seed 1 (same as the D1 base sweep, so runs pair by question).
#   2. D10 ablation: `unfiltered` at the lr picked by scripts/summarize_sft.py
#      (rule fixed in advance, see there), same evals.
#   3. Summary, commit, push; shut down if SHUTDOWN=1.
# Every stage runs even if an earlier one failed.
#
#   STEPS=512 SKIP=true SHUTDOWN=1 nohup setsid bash code/scripts/pipeline_d2_sft.sh &

set -uo pipefail
CODE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "${CODE_DIR}/.." && pwd)"
RUNS="${REPO_ROOT}/runs"
CKPT_ROOT=/root/autodl-tmp/sft_ckpt
STEPS="${STEPS:?set STEPS (D14)}"
SKIP="${SKIP:?set SKIP=true|false}"
LRS="${LRS:-1e-5 3e-5 1e-4}"
EPOCHS="${EPOCHS:-5}"
D="${D:-$(date +%m%d)}"
SHUTDOWN="${SHUTDOWN:-0}"
LOG="${RUNS}/${D}-pipeline-d2-sft.log"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=8
mkdir -p "${RUNS}"
exec > >(tee -a "${LOG}") 2>&1
cd "${CODE_DIR}"
echo "=== $(date -Is) pipeline start: STEPS=${STEPS} SKIP=${SKIP} LRS='${LRS}' EPOCHS=${EPOCHS}"

train() {  # train <data> <lr>; run names use v1b for sft_v1b
  local name="${D}-sft-${1#sft_}-lr$2"
  echo "=== $(date -Is) start train ${name}"
  mkdir -p "${RUNS}/${name}"
  timeout 7200 python -u -m scripts.sft --run_name "${name}" --select "$1" \
    --lr "$2" --epochs "${EPOCHS}" > "${RUNS}/${name}/stdout.txt" 2>&1
  echo "=== $(date -Is) end train ${name} exit=$?"
}

evals() {  # evals <data> <lr>
  local name="${D}-sft-${1#sft_}-lr$2" ep T run
  for ep in $(seq 1 "${EPOCHS}"); do
    for T in 1.0 0.1; do
      run="${D}-eval-sft-${1#sft_}-lr$2-e${ep}-s${STEPS}-T${T}"
      if [[ ! -e "${CKPT_ROOT}/${name}/epoch${ep}.ckpt" ]]; then
        echo "=== missing ${CKPT_ROOT}/${name}/epoch${ep}.ckpt, skip ${run}"; continue
      fi
      echo "=== $(date -Is) start ${run}"
      SPLIT=dev SAMPLER=fast_ancestral STEPS="${STEPS}" TEMP="${T}" BATCH=200 \
        CKPT_PATH="${CKPT_ROOT}/${name}/epoch${ep}.ckpt" RUN_NAME="${run}" \
        timeout 3600 bash scripts/eval_gsm8k.sh sampler.skip_unchanged="${SKIP}" \
        > /dev/null 2>&1
      echo "=== $(date -Is) end ${run} exit=$?"
    done
  done
}

# 0. Base under exactly the same protocol (STEPS, skip flag, T, seed, batch).
#    skip_unchanged is not bit-identical to the default path under TF32 (see
#    docs/02), so the D1 sweep (no skip) is only a secondary reference.
for T in 1.0 0.1; do
  run="${D}-eval-base-s${STEPS}-T${T}"
  echo "=== $(date -Is) start ${run}"
  SPLIT=dev SAMPLER=fast_ancestral STEPS="${STEPS}" TEMP="${T}" BATCH=200 \
    RUN_NAME="${run}" timeout 3600 bash scripts/eval_gsm8k.sh \
    sampler.skip_unchanged="${SKIP}" > /dev/null 2>&1
  echo "=== $(date -Is) end ${run} exit=$?"
done

# 1. lr sweep on v1b.
for LR in ${LRS}; do
  train sft_v1b "${LR}"
  evals sft_v1b "${LR}"
done

# 2. D10 ablation at the selected lr.
SEL="${RUNS}/${D}-sft-lr-select.json"
python scripts/summarize_sft.py --prefix "${D}" --steps "${STEPS}" \
  --select_data v1b --select_out "${SEL}" > /dev/null
LR_SEL=$(python -c "import json; print(json.load(open('${SEL}'))['lr'])" 2>/dev/null)
if [[ -n "${LR_SEL}" ]]; then
  echo "=== selected lr ${LR_SEL} for the unfiltered ablation"
  train unfiltered "${LR_SEL}"
  evals unfiltered "${LR_SEL}"
else
  echo "=== lr selection failed, ablation skipped"
fi

# 3. Summary, commit, push.
python scripts/summarize_sft.py --prefix "${D}" --steps "${STEPS}" \
  --select_data v1b > "${RUNS}/${D}-D2-sft-summary.md" 2>&1
cat "${RUNS}/${D}-D2-sft-summary.md"
cd "${REPO_ROOT}"
git add runs
git commit -q -m "runs: D2 SFT 学习率扫描与 D10 消融（dev 1000，${STEPS} 步，T ∈ {1, 0.1}）

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
for i in 1 2 3; do timeout 120 git push -q origin main && break; sleep 30; done
echo "=== $(date -Is) pipeline done"

if [[ "${SHUTDOWN}" == 1 ]]; then
  sync
  /usr/bin/shutdown
fi
