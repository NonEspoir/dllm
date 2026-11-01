#!/bin/bash
# D5 round 4 (R44): a run that is meant to converge, so the final model is not a
# checkpoint picked off a noise peak.
#
# Every previous base-start run drifted and was stopped by the brake (binary
# ~160, SFT->RL 138, format-only 174); the checkpoint rules then had to choose
# a step inside a short healthy window, and the dev200 maximum (binary step 150,
# format step 100) sat on a single-point fluctuation (docs/09 §4.1, 08 Q45).
# This run adds three things on top of the format + KL arm:
#   --kl_clip 20 --kl_max 10   the k3 coefficient is bounded (the first KL run
#                              spiked to a gradient norm of ~2500 at step 14)
#   --spike_ratio 10           an update whose gradient norm exceeds 10x the
#                              median of the last 50 is discarded
#   --lr_schedule cosine       the step size decays to 10% by the last step, so
#                              the run settles instead of drifting
# and keeps only the latest checkpoint (--keep_ckpts last). Selection rule fixed
# in advance: the LAST checkpoint (a convergence run needs no peaking); the
# summary also prints what the smooth3 rule would have picked, for transparency.
#
#   D=1101 SHUTDOWN=1 nohup setsid bash code/scripts/pipeline_d5f_conv.sh &

set -uo pipefail
CODE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "${CODE_DIR}/.." && pwd)"
RUNS="${REPO_ROOT}/runs"
D="${D:?}"
SHUTDOWN="${SHUTDOWN:-0}"
STEPS="${STEPS:-400}"
NAME="${D}-grpo-conv-fmt0.1-kl0.04"
LOG="${RUNS}/${D}-pipeline-d5f-conv.log"
ARGS=(--lr 3e-5 --updates 4 --no_std --format_reward 0.1 --kl 0.04
      --kl_clip 20 --kl_max 10 --spike_ratio 10 --lr_schedule cosine --lr_min_frac 0.1
      --steps "${STEPS}" --eval_every 25 --keep_ckpts last
      --stop_ratio 2.0 --stop_window 10)
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
exec > >(tee -a "${LOG}") 2>&1
cd "${CODE_DIR}"

# 0. smoke with the exact formal arguments (Q34: import checks are not enough)
echo "=== $(date -Is) smoke: 2 steps, formal arguments"
if ! timeout 900 python -u -m scripts.grpo --run_name "${D}-grpo-smoke-conv" "${ARGS[@]}" \
     --steps 2 --no_eval --keep_ckpts all > "${RUNS}/${D}-grpo-smoke-conv.log" 2>&1; then
  echo "=== smoke FAILED, aborting"; tail -30 "${RUNS}/${D}-grpo-smoke-conv.log"; exit 1
fi
tail -2 "${RUNS}/${D}-grpo-smoke-conv.log"
rm -rf "${RUNS}/${D}-grpo-smoke-conv" "${RUNS}/${D}-grpo-smoke-conv.log" \
       /root/autodl-tmp/rl_ckpt/"${D}-grpo-smoke-conv"
echo "=== $(date -Is) smoke ok"

echo "=== $(date -Is) start ${NAME} (${STEPS} steps)"
mkdir -p "${RUNS}/${NAME}"
timeout 54000 python -u -m scripts.grpo --run_name "${NAME}" "${ARGS[@]}" \
  > "${RUNS}/${NAME}/stdout.txt" 2>&1
echo "=== $(date -Is) end ${NAME} exit=$?"
grep -E '^EVAL|^STOP' "${RUNS}/${NAME}/stdout.txt"

# dev1000 on the LAST checkpoint (rule fixed in advance)
LAST_CKPT="/root/autodl-tmp/rl_ckpt/${NAME}/step$(tail -1 "${RUNS}/${NAME}/train_log.jsonl" | python -c 'import json,sys; print(json.load(sys.stdin)["step"])').ckpt"
echo "=== last checkpoint ${LAST_CKPT}"
for T in 1.0 0.1; do
  run="${NAME}-last-dev-e512-T${T}"
  echo "=== $(date -Is) start ${run}"
  SPLIT=dev SAMPLER=fast_ancestral STEPS=512 TEMP="${T}" BATCH=200 \
    CKPT_PATH="${LAST_CKPT}" RUN_NAME="${run}" \
    timeout 3600 bash scripts/eval_gsm8k.sh sampler.skip_unchanged=true > /dev/null 2>&1
  echo "=== $(date -Is) end ${run} exit=$?"
done

{
  python scripts/summarize_grpo.py "${RUNS}/${NAME}"
  echo
  echo "## Last checkpoint on dev1000 (rule fixed in advance for a convergence run)"
  echo
  echo "| arm | T | dev1000 | p vs base | last 800 | p vs base (last 800) |"
  echo "|---|---|---|---|---|---|"
  python - <<PY
import json, os, sys
sys.path.insert(0, 'scripts')
from summarize_runs import mcnemar
R = os.path.join('${REPO_ROOT}', 'runs')
name = '${NAME}'
last = json.loads(open(os.path.join(R, name, 'train_log.jsonl')).readlines()[-1])['step']
for T in ('1.0', '0.1'):
  b = json.load(open(os.path.join(R, f'1030-eval-base-s512-T{T}', 'metrics.json')))['per_question_correct']
  d = os.path.join(R, f'{name}-last-dev-e512-T{T}', 'metrics.json')
  if not os.path.exists(d):
    print(f'| {name} (step {last}) | {T} | missing | | | |'); continue
  pq = json.load(open(d))['per_question_correct']
  n01, n10, p = mcnemar(b, pq)
  m01, m10, q = mcnemar(b[200:], pq[200:])
  print(f'| {name} (step {last}) | {T} | {100*pq.count("1")/1000:.1f} | {p:.3g} (+{n01}/-{n10}) '
        f'| {100*pq[200:].count("1")/800:.1f} | {q:.3g} (+{m01}/-{m10}) |')
PY
  echo
  echo "Sanity check only (not used): what smooth3 would pick ->"
  python scripts/select_grpo.py pick "${RUNS}/${NAME}" smooth3 || true
} > "${RUNS}/${D}-D5f-conv-summary.md" 2>&1
cat "${RUNS}/${D}-D5f-conv-summary.md"

cd "${REPO_ROOT}"
git add -A runs
git commit -q -m "runs: D5 收敛轮（格式分 + 截断 KL + 尖峰跳过 + cosine LR，${STEPS} 步；定稿 = 最后一步）

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
for i in 1 2 3; do timeout 120 git push -q origin main && break; sleep 30; done
echo "=== $(date -Is) pipeline done"

if [[ "${SHUTDOWN}" == 1 && ! -e /root/autodl-tmp/NO_SHUTDOWN ]]; then
  sync
  /usr/bin/shutdown
else
  echo "=== no shutdown (SHUTDOWN=${SHUTDOWN}, NO_SHUTDOWN $( [[ -e /root/autodl-tmp/NO_SHUTDOWN ]] && echo present || echo absent ))"
fi
