"""Does pre-training overlap (scripts/contamination.py) help base on dev?
Compares per-question accuracy, averaged over the D1 step sweep, between dev
questions with a near-duplicate in TinyGSM (best-item 13-gram coverage >= 0.5)
and the rest. Bootstrap CI over questions.

  python scripts/contamination_effect.py
"""
import json
import os
import random

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
RUNS = os.path.join(REPO, 'runs')


def main():
  with open('/root/autodl-tmp/data/gsm8k_dev.json') as f:
    dev = json.load(f)
  with open(os.path.join(RUNS, '1030-contamination-13gram/metrics.json')) as f:
    near = set(json.load(f)['per_split']['dev']['near_duplicate_ids'])
  g = [i for i, d in enumerate(dev) if d['gsm8k_index'] in near]
  o = [i for i, d in enumerate(dev) if d['gsm8k_index'] not in near]
  for T in ('1.0', '0.1'):
    pq = []
    for s in (64, 128, 256, 512, 1024):
      run = os.path.join(RUNS, f'1030-sweep-fast-dev-s{s}-T{T}')
      with open(os.path.join(run, 'metrics.json')) as f:
        pq.append(json.load(f)['per_question_correct'])
      # per_question_correct follows the order of gsm8k_dev.json
      with open(os.path.join(run, 'samples/results.json')) as f:
        recs = json.load(f)['records']
      assert all(r['prompt'].strip() == d['prompt'].strip() for r, d in zip(recs, dev))
    p = [sum(int(x[i]) for x in pq) / len(pq) for i in range(len(dev))]
    mean = lambda idx: sum(p[i] for i in idx) / len(idx)
    rng = random.Random(0)
    bs = sorted(mean([rng.choice(g) for _ in g]) - mean([rng.choice(o) for _ in o])
                for _ in range(4000))
    print(f'T={T}: near-dup {mean(g):.3f} (n={len(g)}) vs rest {mean(o):.3f} (n={len(o)}), '
          f'diff {mean(g) - mean(o):+.3f}, 95% CI [{bs[100]:+.3f}, {bs[3899]:+.3f}]')


if __name__ == '__main__':
  main()
