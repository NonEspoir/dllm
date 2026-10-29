#!/bin/bash
# GSM8K eval for MDLM on a fixed split (test / dev / dev200), recorded under runs/.
#
# Usage: SPLIT=dev200 SAMPLER=fast_ancestral STEPS=256 RUN_NAME=xxx \
#          bash scripts/eval_gsm8k.sh [extra hydra overrides]
#
# Each split gets its own data cache dir: upstream caches the tokenized set
# under a name that depends only on the tokenizer, so sharing one cache dir
# across splits would silently reuse the wrong split.

set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

CODE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "${CODE_DIR}/.." && pwd)"
SPLIT="${SPLIT:?set SPLIT=test|dev|dev200}"
SAMPLER="${SAMPLER:-fast_ancestral}"
STEPS="${STEPS:-256}"
TEMP="${TEMP:-1.0}"
BATCH="${BATCH:-32}"
SEED="${SEED:-1}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/data}"
CKPT_PATH="${CKPT_PATH:-/root/autodl-tmp/checkpoints/tinygsm/mdlm.ckpt}"
RUN_NAME="${RUN_NAME:-$(date +%m%d)-${SPLIT}-${SAMPLER}-s${STEPS}-T${TEMP}}"
RUN_DIR="${REPO_ROOT}/runs/${RUN_NAME}"

if [[ -e "${RUN_DIR}/results.json" ]]; then
  echo "refusing to overwrite ${RUN_DIR}/results.json" >&2; exit 1
fi
mkdir -p "${RUN_DIR}"
{
  echo "git_commit: $(git -C "${REPO_ROOT}" rev-parse HEAD)"
  echo "git_dirty: $(git -C "${REPO_ROOT}" status --porcelain | wc -l)"
  echo "split: ${SPLIT}"; echo "sampler: ${SAMPLER}"; echo "steps: ${STEPS}"
  echo "temperature: ${TEMP}"; echo "batch: ${BATCH}"; echo "seed: ${SEED}"
  echo "ckpt: ${CKPT_PATH}"; echo "extra: $*"
  echo "gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  echo "start: $(date -Is)"
} > "${RUN_DIR}/run_info.txt"

cd "${CODE_DIR}"
START=$(date +%s)
python -u -m main \
    mode=gsm8k_eval \
    seed="${SEED}" \
    eval.checkpoint_path="${CKPT_PATH}" \
    data=gsm8k-test \
    data.data_path="${DATA_ROOT}/gsm8k_${SPLIT}.json" \
    data.cache_dir="/root/autodl-tmp/data_cache/gsm8k_${SPLIT}" \
    data.tokenizer_name_or_path=HuggingFaceTB/SmolLM-135M \
    model=small \
    model.length=512 \
    algo=mdlm \
    sampler="${SAMPLER}" \
    sampler.steps="${STEPS}" \
    sampler.temperature="${TEMP}" \
    loader.eval_batch_size="${BATCH}" \
    loader.num_workers=4 \
    trainer.num_nodes=1 \
    trainer.devices=1 \
    gsm8k.output_dir="${RUN_DIR}" \
    gsm8k.bootstrap_size=1000 \
    +wandb.offline=True \
    hydra.run.dir="${RUN_DIR}/hydra" \
    "$@" 2>&1 | tee "${RUN_DIR}/log.txt"
END=$(date +%s)
{
  echo "end: $(date -Is)"; echo "wall_seconds: $((END - START))"
} >> "${RUN_DIR}/run_info.txt"
rm -f "${RUN_DIR}"/rank_*.json
