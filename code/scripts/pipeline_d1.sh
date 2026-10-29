#!/bin/bash
# D1 overnight pipeline: sampler check (layers 2/3), paper reproduction on
# test, step x temperature sweep on dev. Then summarise, commit, push and
# shut the instance down. Every stage runs even if an earlier one failed.
#
#   tmux new -d -s d1 'bash code/scripts/pipeline_d1.sh'

set -uo pipefail
CODE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "${CODE_DIR}/.." && pwd)"
RUNS="${REPO_ROOT}/runs"
LOG="${RUNS}/1030-pipeline-d1.log"
SHUTDOWN="${SHUTDOWN:-1}"
mkdir -p "${RUNS}"
exec > >(tee -a "${LOG}") 2>&1
cd "${CODE_DIR}"

stage() {  # stage <run_name> <timeout_s> VAR=... ; runs eval_gsm8k.sh
  local name=$1 limit=$2; shift 2
  echo "=== $(date -Is) start ${name}"
  env "$@" RUN_NAME="${name}" timeout "${limit}" bash scripts/eval_gsm8k.sh \
    > /dev/null 2>&1
  echo "=== $(date -Is) end ${name} exit=$?"
}

# 1. Layers 2/3 (docs/04 §2.4): dev200, 256 steps, T=1, same batch size.
stage 1030-L2-ancestral-dev200-s256 3600 SPLIT=dev200 SAMPLER=ancestral STEPS=256 BATCH=64
stage 1030-L2-fast-dev200-s256      3600 SPLIT=dev200 SAMPLER=fast_ancestral STEPS=256 BATCH=64
stage 1030-L3-fast-dev200-s256-b200 3600 SPLIT=dev200 SAMPLER=fast_ancestral STEPS=256 BATCH=200

# 2. Paper reproduction (the one allowed use of test on D1): upstream
#    sampler, 1024 steps, T=1.
stage 1030-repro-ancestral-test-s1024 14400 SPLIT=test SAMPLER=ancestral STEPS=1024 BATCH=64

# 3. Step x temperature sweep on dev 1000 with the fast sampler (D14).
for T in 1.0 0.1; do
  for S in 64 128 256 512 1024; do
    stage "1030-sweep-fast-dev-s${S}-T${T}" 7200 \
      SPLIT=dev SAMPLER=fast_ancestral STEPS=${S} TEMP=${T} BATCH=200
  done
done

# 4. Summary, commit, push.
{
  echo "# D1 pipeline summary ($(date -Is))"
  echo
  echo "## Sampler check (dev200, 256 steps, T=1)"
  echo
  python scripts/summarize_runs.py table \
    "${RUNS}"/1030-L2-ancestral-dev200-s256 "${RUNS}"/1030-L2-fast-dev200-s256 \
    "${RUNS}"/1030-L3-fast-dev200-s256-b200
  echo
  python scripts/summarize_runs.py compare \
    "${RUNS}"/1030-L2-ancestral-dev200-s256 "${RUNS}"/1030-L2-fast-dev200-s256
  echo
  echo "## Reproduction (test, upstream sampler, 1024 steps, T=1; paper: 18.0)"
  echo
  python scripts/summarize_runs.py table "${RUNS}"/1030-repro-ancestral-test-s1024
  echo
  echo "## Sweep (dev 1000, fast sampler)"
  echo
  python scripts/summarize_runs.py table "${RUNS}"/1030-sweep-fast-dev-s*-T*
} > "${RUNS}/1030-D1-summary.md" 2>&1
cat "${RUNS}/1030-D1-summary.md"

cd "${REPO_ROOT}"
git add runs
git commit -q -m "runs: D1 流水线结果（采样器层次 ②③、test 复现、dev 步数 × 温度扫描）

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
for i in 1 2 3; do timeout 120 git push -q origin main && break; sleep 30; done
echo "=== $(date -Is) pipeline done"

if [[ "${SHUTDOWN}" == 1 ]]; then
  sync
  /usr/bin/shutdown
fi
