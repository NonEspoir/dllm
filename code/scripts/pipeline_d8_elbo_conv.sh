#!/bin/bash
# R46 (D37): three pieces, unattended, then shut down (>= 3 h rule, R30).
#   1. base majority vote on dev: 8 different seeds x T in {1, 0.1}, 512 steps
#      + skip -- the same-budget control for the final model's vote (D16).
#   2. ELBO convergence run: exactly the D35 convergence recipe (format 0.1,
#      KL 0.04 with k3 clipped, spike skip 10x, cosine lr 3e-5 -> 3e-6, 400
#      steps, brake, keep only the latest checkpoint) except --logprob elbo.
#      Rule fixed in advance: its LAST checkpoint goes to dev1000.
#   3. Final model (D37 rule, user's choice in R46): among the candidates
#        convergence run step 400 (trajectory), ELBO arm step 150, ELBO
#        convergence run last step
#      pick the highest dev1000 mean over T=1 and T=0.1 (ties -> the earlier
#      decided one). If it is not the D36 model, run test once on it in the
#      D36 protocol (512/T=1, 512/T=0.1, 1024/T=1). Test is never used to choose.
#
#   D=1103 SHUTDOWN=1 nohup setsid bash code/scripts/pipeline_d8_elbo_conv.sh &

set -uo pipefail
CODE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "${CODE_DIR}/.." && pwd)"
RUNS="${REPO_ROOT}/runs"
D="${D:?}"
SHUTDOWN="${SHUTDOWN:-0}"
LOG="${RUNS}/${D}-pipeline-d8.log"
BASE_CKPT="/root/autodl-tmp/checkpoints/tinygsm/mdlm.ckpt"
NAME="${D}-grpo-elbo-conv-fmt0.1-kl0.04"
ARGS=(--logprob elbo --lr 3e-5 --updates 4 --no_std --format_reward 0.1 --kl 0.04
      --kl_clip 20 --kl_max 10 --spike_ratio 10 --lr_schedule cosine --lr_min_frac 0.1
      --steps 400 --eval_every 25 --keep_ckpts last --stop_ratio 2.0 --stop_window 10)
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
exec > >(tee -a "${LOG}") 2>&1
cd "${CODE_DIR}"
echo "=== $(date -Is) pipeline d8 start"

ev() {  # split ckpt steps temp name [seed]
  local split="$1" ckpt="$2" steps="$3" temp="$4" name="$5" seed="${6:-1}"
  if [[ -e "${RUNS}/${name}/metrics.json" ]]; then
    echo "=== skip ${name} (exists)"; return 0
  fi
  echo "=== $(date -Is) start ${name}"
  SPLIT="${split}" SAMPLER=fast_ancestral STEPS="${steps}" TEMP="${temp}" BATCH=200 \
    SEED="${seed}" CKPT_PATH="${ckpt}" RUN_NAME="${name}" \
    timeout 5400 bash scripts/eval_gsm8k.sh sampler.skip_unchanged=true > /dev/null 2>&1
  echo "=== $(date -Is) end ${name} exit=$?"
}

# 1. base majority vote
for T in 1.0 0.1; do
  for seed in 1 2 3 4 5 6 7 8; do
    ev dev "${BASE_CKPT}" 512 "${T}" "${D}-dev-base-mv-T${T}-seed${seed}" "${seed}"
  done
done

# 2. ELBO convergence run (smoke with the exact formal arguments first, Q34)
echo "=== $(date -Is) smoke: 2 steps, formal arguments"
if ! timeout 900 python -u -m scripts.grpo --run_name "${D}-grpo-smoke-elbo-conv" "${ARGS[@]}" \
     --steps 2 --no_eval --keep_ckpts all > "${RUNS}/${D}-grpo-smoke-elbo-conv.log" 2>&1; then
  echo "=== smoke FAILED"; tail -30 "${RUNS}/${D}-grpo-smoke-elbo-conv.log"
else
  echo "=== $(date -Is) smoke ok"
  rm -rf "${RUNS}/${D}-grpo-smoke-elbo-conv" "${RUNS}/${D}-grpo-smoke-elbo-conv.log" \
         "/root/autodl-tmp/rl_ckpt/${D}-grpo-smoke-elbo-conv"
  echo "=== $(date -Is) start ${NAME}"
  mkdir -p "${RUNS}/${NAME}"
  timeout 36000 python -u -m scripts.grpo --run_name "${NAME}" "${ARGS[@]}" \
    > "${RUNS}/${NAME}/stdout.txt" 2>&1
  echo "=== $(date -Is) end ${NAME} exit=$?"
  grep -E '^EVAL|^STOP' "${RUNS}/${NAME}/stdout.txt"
  last=$(tail -1 "${RUNS}/${NAME}/train_log.jsonl" | python -c 'import json,sys; print(json.load(sys.stdin)["step"])')
  LAST_CKPT="/root/autodl-tmp/rl_ckpt/${NAME}/step${last}.ckpt"
  echo "=== last checkpoint ${LAST_CKPT}"
  for T in 1.0 0.1; do
    ev dev "${LAST_CKPT}" 512 "${T}" "${NAME}-last-dev-e512-T${T}"
  done
fi

# 3. final model by dev1000 mean; test once on it if it changed
SEL=$(python - "${RUNS}" "${NAME}" <<'PY'
import json, os, sys
R, name = sys.argv[1:3]
cands = [  # (tag, dev1000 run prefix, checkpoint) in the order they were decided
  ('conv400', '1101-grpo-conv-fmt0.1-kl0.04-last',
   '/root/autodl-tmp/rl_ckpt/1101-grpo-conv-fmt0.1-kl0.04/step400.ckpt'),
  ('elbo150', '1103-grpo-elbo-fmt0.1-best-s150',
   '/root/autodl-tmp/rl_ckpt/1103-grpo-elbo-fmt0.1/step150.ckpt'),
]
p = os.path.join(R, name, 'train_log.jsonl')
if os.path.exists(p):
  last = json.loads(open(p).readlines()[-1])['step']
  cands.append(('elboconv', f'{name}-last', f'/root/autodl-tmp/rl_ckpt/{name}/step{last}.ckpt'))
best = None
for tag, pre, ck in cands:
  acc = []
  for T in ('1.0', '0.1'):
    m = os.path.join(R, f'{pre}-dev-e512-T{T}', 'metrics.json')
    if os.path.exists(m):
      acc.append(json.load(open(m))['accuracy'])
  if len(acc) == 2:
    mean = sum(acc) / 2
    print(f'# {tag} dev1000 T=1 {100*acc[0]:.1f} T=0.1 {100*acc[1]:.1f} mean {100*mean:.2f}', file=sys.stderr)
    if best is None or mean > best[1]:
      best = (tag, mean, ck)
print(best[0], best[2])
PY
)
echo "=== final selection: ${SEL}"
read -r FTAG FCKPT <<< "${SEL}"
if [[ "${FTAG}" != "conv400" && -f "${FCKPT}" ]]; then
  ev test "${FCKPT}" 512  1.0 "${D}-test-${FTAG}-s512-T1.0"
  ev test "${FCKPT}" 512  0.1 "${D}-test-${FTAG}-s512-T0.1"
  ev test "${FCKPT}" 1024 1.0 "${D}-test-${FTAG}-s1024-T1.0"
fi

# 4. summary
{
  echo "# ${D} D8: base 多数投票、ELBO 收敛轮、定稿"
  for T in 1.0 0.1; do
    echo; echo "## base 多数投票 dev T=${T}（8 个 seed）"; echo '```'
    dirs=(); for s in 1 2 3 4 5 6 7 8; do dirs+=("${RUNS}/${D}-dev-base-mv-T${T}-seed${s}"); done
    python scripts/majority_vote.py "${dirs[@]}" 2>/dev/null | grep -E "^[0-9]+ samples|^k=|no usable" || true
    echo '```'
  done
  echo; echo "## ELBO 收敛轮"; echo
  [[ -f "${RUNS}/${NAME}/train_log.jsonl" ]] && python scripts/summarize_grpo.py "${RUNS}/${NAME}"
  echo; echo "## 定稿选择（dev1000 两温度平均；test 不参与选择）"; echo
  echo "选中：${SEL}"
  python - "${RUNS}" "${D}" "${FTAG}" <<'PY2'
import json, os, sys
R, D, tag = sys.argv[1:4]
for steps, T in (('512', '1.0'), ('512', '0.1'), ('1024', '1.0')):
  m = os.path.join(R, f'{D}-test-{tag}-s{steps}-T{T}', 'metrics.json')
  if os.path.exists(m):
    x = json.load(open(m))
    print(f'test {tag} {steps} 步 T={T}: {100*x["accuracy"]:.2f}% '
          f'CI [{100*x["ci95"][0]:.1f}, {100*x["ci95"][1]:.1f}]')
PY2
} > "${RUNS}/${D}-D8-summary.md" 2>&1
cat "${RUNS}/${D}-D8-summary.md"

cd "${REPO_ROOT}"
git add -A runs
git commit -q -m "runs: D8（base 多数投票、ELBO 收敛轮、按 dev 定稿与 test）

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
for i in 1 2 3; do timeout 120 git push -q origin main && break; sleep 30; done
echo "=== $(date -Is) pipeline done"

if [[ "${SHUTDOWN}" == 1 && ! -e /root/autodl-tmp/NO_SHUTDOWN ]]; then
  sync
  /usr/bin/shutdown
else
  echo "=== no shutdown (SHUTDOWN=${SHUTDOWN}, NO_SHUTDOWN $( [[ -e /root/autodl-tmp/NO_SHUTDOWN ]] && echo present || echo absent ))"
fi
