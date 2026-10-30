"""rl.reward.grade must agree with the upstream scorer on what is correct.

  cd code && python -m tests.test_rl_reward

Runs both on saved eval generations (base and an SFT checkpoint, both
temperatures, 4000 responses) and requires grade(...) == 'correct' exactly
when sandbox_gsm8k.evaluate_samples(...) is True. Also prints the category
mix, which is what the format reward would see.
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from rl import rollout  # noqa: E402

RUNS = os.path.join(os.path.dirname(__file__), '../../runs')
SOURCES = ['1030-eval-base-s512-T1.0', '1030-eval-base-s512-T0.1',
           '1030-eval-sft-v1b-lr3e-5-e2-s512-T1.0', '1030-eval-sft-v1b-lr3e-5-e2-s512-T0.1']


def main():
  ok = True
  for src in SOURCES:
    with open(os.path.join(RUNS, src, 'samples', 'results.json')) as f:
      recs = json.load(f)['records']
    resp = [r['response'] for r in recs]
    gold = [r['response_ground_truth'] for r in recs]
    scores = rollout.score(resp, gold, procs=16)
    cats = rollout.grade(resp, gold, procs=16)
    mism = sum((c == 'correct') != bool(s) for c, s in zip(cats, scores))
    print(f'{src}: {len(recs)} responses, mismatches {mism}, categories '
          f'{dict(collections.Counter(cats))}')
    ok &= mism == 0
  print('ALL PASSED' if ok else 'FAILED')
  sys.exit(0 if ok else 1)


if __name__ == '__main__':
  main()
