#!/bin/bash
# D6 (R45): final evaluation on the test split, then dev-only analysis.
#
# Rule (D08): test is used only for the models that go into the final table, in
# one fixed protocol, one seed each:
#   test, 512 steps + skip_unchanged, T = 1.0 and 0.1   (our standard protocol)
#   test, 1024 steps + skip_unchanged, T = 1.0          (the paper's setting)
# Models: the final RL model (convergence run step 400, chosen by a rule fixed
# before its results existed, D35), base, and the SFT checkpoint picked by the
# D30 rule (lr 1e-5, epoch 1).
# Then on dev only:
#   step robustness of the final model (128 / 256 / 1024 steps, T=1),
#   majority vote / pass@k over 8 *different* seeds (D16 free control; the eval
#   script defaults to seed 1, which would make all 8 samples identical).
#
#   D=1102 SHUTDOWN=1 nohup setsid bash code/scripts/pipeline_d6_test.sh &

set -uo pipefail
CODE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "${CODE_DIR}/.." && pwd)"
RUNS="${REPO_ROOT}/runs"
D="${D:?}"
SHUTDOWN="${SHUTDOWN:-0}"
LOG="${RUNS}/${D}-pipeline-d6.log"
RL_CKPT="/root/autodl-tmp/rl_ckpt/1101-grpo-conv-fmt0.1-kl0.04/step400.ckpt"
BASE_CKPT="/root/autodl-tmp/checkpoints/tinygsm/mdlm.ckpt"
SFT_CKPT="/root/autodl-tmp/sft_ckpt/1030-sft-v1b-lr1e-5/epoch1.ckpt"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
exec > >(tee -a "${LOG}") 2>&1
cd "${CODE_DIR}"
echo "=== $(date -Is) pipeline d6 start"
for c in "${RL_CKPT}" "${BASE_CKPT}" "${SFT_CKPT}"; do
  test -f "$c" || { echo "missing $c"; exit 1; }
done

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

# 1. test: three models x three settings
for pair in "rl:${RL_CKPT}" "base:${BASE_CKPT}" "sft:${SFT_CKPT}"; do
  tag="${pair%%:*}"; ckpt="${pair#*:}"
  ev test "${ckpt}" 512  1.0 "${D}-test-${tag}-s512-T1.0"
  ev test "${ckpt}" 512  0.1 "${D}-test-${tag}-s512-T0.1"
  ev test "${ckpt}" 1024 1.0 "${D}-test-${tag}-s1024-T1.0"
done

# 2. final model on dev (the 512-step standard protocol; the others are robustness)
ev dev "${RL_CKPT}" 512 1.0 "${D}-dev-rl-s512-T1.0"
ev dev "${RL_CKPT}" 512 0.1 "${D}-dev-rl-s512-T0.1"
for steps in 128 256 1024; do
  ev dev "${RL_CKPT}" "${steps}" 1.0 "${D}-dev-rl-s${steps}-T1.0"
done

# 3. majority vote / pass@k on dev: 8 different seeds per temperature
for T in 1.0 0.1; do
  for seed in 1 2 3 4 5 6 7 8; do
    ev dev "${RL_CKPT}" 512 "${T}" "${D}-dev-rl-mv-T${T}-seed${seed}" "${seed}"
  done
done

# 4. summary
{
  echo "# D6 final evaluation (${D})"
  echo
  echo "## test set, three models x three settings"
  echo
  echo "| model | steps | T | accuracy | 95% CI | n | time |"
  echo "|---|---|---|---|---|---|---|"
  for tag in rl base sft; do
    for setting in "512 1.0" "512 0.1" "1024 1.0"; do
      set -- ${setting}
      python - "$D" "$tag" "$1" "$2" <<'PY'
import json, os, sys
D, tag, steps, T = sys.argv[1:5]
p = os.path.join('/root/autodl-tmp/dllm/runs', f'{D}-test-{tag}-s{steps}-T{T}', 'metrics.json')
if not os.path.exists(p):
    print(f'| {tag} | {steps} | {T} | missing | | | |'); raise SystemExit
m = json.load(open(p))
print(f"| {tag} | {steps} | {T} | {100*m['accuracy']:.1f} | "
      f"[{100*m['ci95'][0]:.1f}, {100*m['ci95'][1]:.1f}] | {m['num_total']} | {m['wall_seconds']} s |")
PY
    done
  done
  echo
  echo "## final model on dev: step robustness (T=1)"
  echo
  echo "| steps | accuracy |"
  echo "|---|---|"
  for steps in 128 256 512 1024; do
    python - "$D" "$steps" <<'PY'
import json, os, sys
D, steps = sys.argv[1:3]
p = os.path.join('/root/autodl-tmp/dllm/runs', f'{D}-dev-rl-s{steps}-T1.0', 'metrics.json')
print(f"| {steps} | {100*json.load(open(p))['accuracy']:.1f} |" if os.path.exists(p) else f'| {steps} | missing |')
PY
  done
  echo
  echo "## majority vote / pass@k on dev (8 different seeds)"
  for T in 1.0 0.1; do
    echo
    echo "### T=$T"
    echo '```'
    dirs=()
    for seed in 1 2 3 4 5 6 7 8; do dirs+=("${RUNS}/${D}-dev-rl-mv-T${T}-seed${seed}"); done
    python scripts/majority_vote.py "${dirs[@]}" 2>&1 || true
    echo '```'
  done
} > "${RUNS}/${D}-D6-test-summary.md" 2>&1
cat "${RUNS}/${D}-D6-test-summary.md"

cd "${REPO_ROOT}"
git add -A runs
git commit -q -m "runs: D6 test 评测（定稿 RL / base / SFT，512 与 1024 步）+ 步数鲁棒性 + 多数投票

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
for i in 1 2 3; do timeout 120 git push -q origin main && break; sleep 30; done
echo "=== $(date -Is) pipeline done"

if [[ "${SHUTDOWN}" == 1 && ! -e /root/autodl-tmp/NO_SHUTDOWN ]]; then
  sync
  /usr/bin/shutdown
else
  echo "=== no shutdown (SHUTDOWN=${SHUTDOWN}, NO_SHUTDOWN $( [[ -e /root/autodl-tmp/NO_SHUTDOWN ]] && echo present || echo absent ))"
fi
