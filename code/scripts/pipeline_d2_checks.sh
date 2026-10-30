#!/bin/bash
# D2 checks before the SFT sweep (docs/06 "下一步" 2-3):
#   1. skip_unchanged speed: dev200, batch 200, steps {256, 512, 1024} x skip
#      {false, true}; decides D14 by the rule in docs/08 Q8 (512 steps with
#      skip costing <= ~1.5x 256 steps without skip -> evaluate at 512).
#      Same seed, so skip on/off must also give identical per-question results.
#   2. Layer 2 on dev 1000: upstream sampler, 256 steps, T=1, seed 1, paired
#      with runs/1030-sweep-fast-dev-s256-T1.0 (McNemar).
# Runs sequentially so the timings are not disturbed by each other.

set -uo pipefail
CODE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "${CODE_DIR}/.." && pwd)"
RUNS="${REPO_ROOT}/runs"
D="${D:-$(date +%m%d)}"
cd "${CODE_DIR}"

stage() {  # stage <run_name> <timeout_s> VAR=... [-- hydra overrides]
  local name=$1 limit=$2; shift 2
  local envs=()
  while [[ $# -gt 0 && $1 != -- ]]; do envs+=("$1"); shift; done
  [[ $# -gt 0 ]] && shift
  echo "=== $(date -Is) start ${name}"
  env "${envs[@]}" RUN_NAME="${name}" timeout "${limit}" \
    bash scripts/eval_gsm8k.sh "$@" > /dev/null 2>&1
  echo "=== $(date -Is) end ${name} exit=$?"
}

for S in 256 512 1024; do
  for K in false true; do
    stage "${D}-speed-skip${K}-dev200-s${S}" 3600 \
      SPLIT=dev200 SAMPLER=fast_ancestral STEPS=${S} BATCH=200 \
      -- sampler.skip_unchanged=${K}
  done
done

# Layer 2 on dev 1000 (run separately on 10-30, kept here for reruns):
#   SPLIT=dev SAMPLER=ancestral STEPS=256 BATCH=64 \
#     RUN_NAME=<mmdd>-L2-ancestral-dev-s256-T1.0 bash scripts/eval_gsm8k.sh
L2="${RUNS}/${D}-L2-ancestral-dev-s256-T1.0"

{
  echo "## skip_unchanged speed (dev200, batch 200, T=1)"
  echo
  python scripts/summarize_runs.py table "${RUNS}/${D}"-speed-skip*-dev200-s*
  echo
  for S in 256 512 1024; do
    python scripts/summarize_runs.py compare \
      "${RUNS}/${D}-speed-skipfalse-dev200-s${S}" "${RUNS}/${D}-speed-skiptrue-dev200-s${S}"
  done
  python - "${RUNS}" "${D}" <<'P'
import json, sys
runs, d = sys.argv[1:]
m = {(k, s): json.load(open(f'{runs}/{d}-speed-skip{k}-dev200-s{s}/metrics.json'))
     for k in ('false', 'true') for s in (256, 512, 1024)}
base = m['false', 256]['sample_seconds']
print('\nsample_seconds relative to 256 steps without skip:')
for (k, s), v in sorted(m.items(), key=lambda x: (x[0][1], x[0][0])):
  same = v['per_question_correct'] == m['false', s]['per_question_correct']
  print(f'  steps {s:4d} skip {k:5s}: {v["sample_seconds"]:4d} s  '
        f'x{v["sample_seconds"] / base:.2f}  same-as-noskip={same}')
P
  echo
  echo "## Layer 2 on dev 1000 (256 steps, T=1)"
  echo
  python scripts/summarize_runs.py table "${L2}" "${RUNS}/1030-sweep-fast-dev-s256-T1.0"
  python scripts/summarize_runs.py compare "${L2}" "${RUNS}/1030-sweep-fast-dev-s256-T1.0"
} > "${RUNS}/${D}-D2-checks-summary.md" 2>&1
cat "${RUNS}/${D}-D2-checks-summary.md"
echo "=== $(date -Is) done"
