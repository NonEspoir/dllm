"""Majority vote / pass@k over k samples per question (D16 free control).

  python scripts/majority_vote.py ../runs/<run1> ../runs/<run2> ...

Each run must contain samples/results.json for the same split in the same order
(e.g. the same checkpoint evaluated k times with different seeds). For every
question: value = the number the code returns (or prints, like the scorer);
majority vote over the values that are not None, ties go to the value from the
earliest sample; all None -> wrong. Reports:
  - accuracy of the first sample (single-shot baseline),
  - majority-vote accuracy over all k,
  - majority accuracy for k = 1..K (first k samples),
  - pass@k: fraction of questions with at least one correct answer in the k samples.
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import sandbox_gsm8k as sb  # noqa: E402


def value(sample, timeout_s=5.0):
  """The number the scorer would compare, or None."""
  code = sb._extract_code(sample)
  try:
    with sb._time_limit(timeout_s):
      ns = sb._safe_exec_no_timer(code)
      fn = ns.get('simple_math_problem', None)
      if fn is None:
        return None
      out_buf = __import__('io').StringIO()
      import contextlib as _c
      with _c.redirect_stdout(out_buf), _c.redirect_stderr(__import__('io').StringIO()):
        out = fn()
  except (sb._Timeout, Exception):
    return None
  v = sb._to_number(out)
  if v is None:
    v = sb._extract_text_answer(out_buf.getvalue())
  return v


def correct(v, gold):
  return sb._numbers_equal(v, sb._extract_gold_answer(gold))


def vote(vals):
  """Most common non-None value; ties -> earliest sample; all None -> None."""
  counts = collections.Counter(v for v in vals if v is not None)
  if not counts:
    return None
  best = max(counts.values())
  for v in vals:
    if v is not None and counts[v] == best:
      return v
  return None


def main():
  runs = sys.argv[1:]
  data = []
  for r in runs:
    with open(os.path.join(r, 'samples', 'results.json')) as f:
      data.append(json.load(f)['records'])
  n = len(data[0])
  assert all(len(d) == n for d in data), 'runs must cover the same questions'
  golds = [d['response_ground_truth'] for d in data[0]]
  vals = [[value(d[i]['response']) for d in data] for i in range(n)]
  print(f'{len(runs)} samples x {n} questions')
  for k in range(1, len(runs) + 1):
    acc = sum(correct(vote(v[:k]), g) for v, g in zip(vals, golds)) / n
    p = sum(any(correct(v, g) for v in vv[:k]) for vv, g in zip(vals, golds)) / n
    print(f'k={k}: majority {100 * acc:.1f}%   pass@{k} {100 * p:.1f}%')
  none = sum(all(v is None for v in vv) for vv in vals)
  print(f'questions with no usable answer in any sample: {none} ({100 * none / n:.1f}%)')


if __name__ == '__main__':
  main()
