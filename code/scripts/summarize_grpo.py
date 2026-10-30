"""Summary of GRPO runs: dev accuracy over training steps, paired McNemar against
base under the same protocol, and training curves (reward, kept rows, action
log-prob) averaged per 10 steps.

  python scripts/summarize_grpo.py ../runs/1030-grpo-base-lr1e-6 ...

Base reference on dev200 = the first 200 questions of the same-protocol dev1000
base run (first batch is identical: checked 200/200 identical outputs).
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.summarize_runs import mcnemar  # noqa: E402

RUNS = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../runs'))


def base_ref(split, steps, T):
  n = {'dev200': 200, 'dev': 1000}[split]
  with open(os.path.join(RUNS, f'1030-eval-base-s{steps}-T{T}', 'metrics.json')) as f:
    return json.load(f)['per_question_correct'][:n]


def main():
  for run in sys.argv[1:]:
    name = os.path.basename(run.rstrip('/'))
    with open(os.path.join(run, 'run_info.json')) as f:
      args = json.load(f)['args']
    print(f'\n## {name}  (lr {args["lr"]}, {args["questions"]} q x G={args["group"]}, '
          f'rollout {args["rollout_steps"]} steps)\n')
    log = []
    if os.path.exists(os.path.join(run, 'train_log.jsonl')):
      log = [json.loads(l) for l in open(os.path.join(run, 'train_log.jsonl'))]
    print('| steps | reward | solved any | kept rows | action logp | grad norm | s/step |')
    print('|---|---|---|---|---|---|---|')
    for s in range(0, len(log), 10):
      w = log[s:s + 10]
      lp = [r['action_logp'] for r in w if r.get('action_logp') is not None]
      print(f"| {w[0]['step']}-{w[-1]['step']} | {sum(r['reward'] for r in w) / len(w):.3f} "
            f"| {sum(r['solved_any'] for r in w) / len(w):.2f} "
            f"| {sum(r['kept_rows'] for r in w) / len(w):.0f} "
            f"| {sum(lp) / max(len(lp), 1):.4f} "
            f"| {sum(r['grad_norm'] for r in w) / len(w):.3f} "
            f"| {sum(r['rollout_s'] + r['train_s'] for r in w) / len(w):.0f} |")
    split, steps = args['eval_split'], args['eval_steps']
    print(f'\n| step | T=1 | p vs base | T=0.1 | p vs base |')
    print('|---|---|---|---|---|')
    refs = {T: base_ref(split, steps, T) for T in ('1.0', '0.1')}
    print(f"| base | {100 * refs['1.0'].count('1') / len(refs['1.0']):.1f} | — "
          f"| {100 * refs['0.1'].count('1') / len(refs['0.1']):.1f} | — |")
    s = args['eval_every']
    while s <= args['steps']:
      cells = []
      for T in ('1.0', '0.1'):
        d = os.path.join(RUNS, f'{name}-s{s}-{split}-e{steps}-T{T}', 'metrics.json')
        if not os.path.exists(d):
          cells += ['—', '—']
          continue
        with open(d) as f:
          pq = json.load(f)['per_question_correct']
        n01, n10, p = mcnemar(refs[T], pq)
        cells += [f"{100 * pq.count('1') / len(pq):.1f}", f'{p:.3g} (+{n01}/-{n10})']
      print(f'| {s} | ' + ' | '.join(cells) + ' |')
      s += args['eval_every']


if __name__ == '__main__':
  main()
