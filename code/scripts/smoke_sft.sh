#!/bin/bash
# First thing to run on a GPU after writing scripts/sft.py:
#   1. a 20-step SFT run with lr=0 must leave the model unchanged: its dev200
#      accuracy must equal the base checkpoint's under the same sampler seed;
#   2. a 20-step run with a real lr must lower the training loss.
set -euo pipefail
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=8
cd "$(dirname "$0")/.."
python -m scripts.sft --run_name smoke-sft-lr0 --lr 0 --max_steps 20
python -m scripts.sft --run_name smoke-sft-lr1e-4 --lr 1e-4 --max_steps 20
for ck in base smoke-sft-lr0; do
  if [[ $ck == base ]]; then P=/root/autodl-tmp/checkpoints/tinygsm/mdlm.ckpt
  else P=/root/autodl-tmp/sft_ckpt/$ck/epoch1.ckpt; fi
  SPLIT=dev200 SAMPLER=fast_ancestral STEPS=64 BATCH=200 CKPT_PATH=$P \
    RUN_NAME=smoke-eval-$ck bash scripts/eval_gsm8k.sh > /dev/null 2>&1
done
python scripts/summarize_runs.py compare ../runs/smoke-eval-base ../runs/smoke-eval-smoke-sft-lr0
python - <<'P'
import json
for r in ['smoke-sft-lr0', 'smoke-sft-lr1e-4']:
  log = [json.loads(l) for l in open(f'../runs/{r}/train_log.jsonl')]
  print(r, 'loss first/last', round(log[0]['loss'], 4), round(log[-1]['loss'], 4))
P
