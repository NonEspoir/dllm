"""Final step of the formal RL run (D32): pick each arm's checkpoint by the rule
fixed in advance, then report its dev1000 accuracy against base.

  python scripts/select_grpo.py pick ../runs/<arm> [rule]   -> "<step> <ckpt>"
  python scripts/select_grpo.py report ../runs/<arm> ...    -> markdown
  rule = single (default, D32) | smooth3 (R44) | last (convergence runs)

Rule (--rule): the original D32 rule is 'single' (highest mean dev200 accuracy
over T=1 and T=0.1; ties -> earlier step). R44 added:
  smooth3  highest 3-point mean of that quantity (the point and its two
           neighbours on the eval grid); ties -> *later* step. A single dev200
           point has a standard error of about 2.5-3 points, so the raw maximum
           is a coin flip: the D5 format arm's step 100 (0.41, neighbours 0.39)
           and the binary arm's step 150 (chosen right where the drift started)
           are both artefacts of it.
  last     the last evaluated checkpoint (convergence runs, R44).
dev200 is the first 200 questions of dev1000, so the dev1000 number is
optimistic for the selected checkpoint; the last 800 questions were not used
for within-arm selection and give an unbiased comparison. Base reference: the
same-protocol dev1000 base runs (runs/1030-eval-base-s512-T*).
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.summarize_runs import mcnemar  # noqa: E402

RUNS = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../runs'))


def load_evals(run):
  """metrics.json, or the EVAL lines of stdout.txt for a run that was stopped
  before writing it (D5 binary arm, stopped at step 264 after diverging)."""
  path = os.path.join(run, 'metrics.json')
  if os.path.exists(path):
    with open(path) as f:
      return json.load(f)['evals']
  with open(os.path.join(run, 'stdout.txt')) as f:
    return [json.loads(l[len('EVAL '):]) for l in f if l.startswith('EVAL ')]


def pick(run, rule='single'):
  evals = load_evals(run)
  ok = [e for e in evals if e.get('acc_T1.0') is not None and e.get('acc_T0.1') is not None]
  if not ok:
    raise SystemExit(f'no complete eval in {run}')
  mean = [(e['acc_T1.0'] + e['acc_T0.1']) / 2 for e in ok]
  if rule == 'last':
    return ok[-1]
  if rule == 'smooth3':
    # R44: the point and its two neighbours on the eval grid; ties -> later step
    smooth = [sum(mean[max(0, i - 1):i + 2]) / len(mean[max(0, i - 1):i + 2])
              for i in range(len(ok))]
    best = max(range(len(ok)), key=lambda i: (smooth[i], i))
    return ok[best]
  # 'single' (D32): highest mean; ties -> earlier step
  best = max(range(len(ok)), key=lambda i: (mean[i], -ok[i]['step']))
  return ok[best]


def report(runs, rule='single'):
  print('| arm | step | T | dev1000 | p vs base | last 800 | p vs base (last 800) |')
  print('|---|---|---|---|---|---|---|')
  for T in ('1.0', '0.1'):
    with open(os.path.join(RUNS, f'1030-eval-base-s512-T{T}', 'metrics.json')) as f:
      b = json.load(f)['per_question_correct']
    print(f"| base | — | {T} | {100 * b.count('1') / 1000:.1f} | — "
          f"| {100 * b[200:].count('1') / 800:.1f} | — |")
  for run in runs:
    name = os.path.basename(run.rstrip('/'))
    best = pick(run, rule)
    for T in ('1.0', '0.1'):
      d = os.path.join(RUNS, f"{name}-best-s{best['step']}-dev-e512-T{T}", 'metrics.json')
      if not os.path.exists(d):
        print(f"| {name} | {best['step']} | {T} | missing | | | |")
        continue
      with open(d) as f:
        pq = json.load(f)['per_question_correct']
      with open(os.path.join(RUNS, f'1030-eval-base-s512-T{T}', 'metrics.json')) as f:
        b = json.load(f)['per_question_correct']
      n01, n10, p = mcnemar(b, pq)
      m01, m10, q = mcnemar(b[200:], pq[200:])
      print(f"| {name} | {best['step']} | {T} | {100 * pq.count('1') / 1000:.1f} "
            f"| {p:.3g} (+{n01}/-{n10}) | {100 * pq[200:].count('1') / 800:.1f} "
            f"| {q:.3g} (+{m01}/-{m10}) |")


if __name__ == '__main__':
  if sys.argv[1] == 'pick':
    rule = sys.argv[3] if len(sys.argv) > 3 else 'single'
    b = pick(sys.argv[2], rule)
    print(b['step'], b['ckpt'])
  else:
    report(sys.argv[2:])
