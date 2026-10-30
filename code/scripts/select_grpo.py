"""Final step of the formal RL run (D32): pick each arm's checkpoint by the rule
fixed in advance, then report its dev1000 accuracy against base.

  python scripts/select_grpo.py pick ../runs/<arm>        -> prints "<step> <ckpt>"
  python scripts/select_grpo.py report ../runs/<arm> ...  -> markdown

Rule: highest mean dev200 accuracy over T=1 and T=0.1; ties -> earlier step.
dev200 is the first 200 questions of dev1000, so the dev1000 number is
optimistic for the selected checkpoint; the last 800 questions were not used
for selection and give an unbiased comparison. Base reference: the
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


def pick(run):
  evals = load_evals(run)
  ok = [e for e in evals if e.get('acc_T1.0') is not None and e.get('acc_T0.1') is not None]
  best = max(ok, key=lambda e: ((e['acc_T1.0'] + e['acc_T0.1']) / 2, -e['step']))
  return best


def report(runs):
  print('| arm | step | T | dev1000 | p vs base | last 800 | p vs base (last 800) |')
  print('|---|---|---|---|---|---|---|')
  for T in ('1.0', '0.1'):
    with open(os.path.join(RUNS, f'1030-eval-base-s512-T{T}', 'metrics.json')) as f:
      b = json.load(f)['per_question_correct']
    print(f"| base | — | {T} | {100 * b.count('1') / 1000:.1f} | — "
          f"| {100 * b[200:].count('1') / 800:.1f} | — |")
  for run in runs:
    name = os.path.basename(run.rstrip('/'))
    best = pick(run)
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
    b = pick(sys.argv[2])
    print(b['step'], b['ckpt'])
  else:
    report(sys.argv[2:])
