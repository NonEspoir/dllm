"""Error categories of GSM8K generations, using the upstream scorer's own
extraction and execution (sandbox_gsm8k), so 'correct' matches metrics.json.

  python scripts/error_breakdown.py ../runs/<a> [../runs/<b> ...]

Categories: no_fn (no simple_math_problem after extraction), syntax (does not
compile), runtime (exception or timeout), wrong (runs, wrong number), correct.
Also: docstring quote style and mean response length (characters).
"""
import contextlib
import io
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import sandbox_gsm8k as sb  # noqa: E402


def category(sample, answer, timeout_s=5.0):
  code = sb._extract_code(sample)
  try:
    compile(code, '<sample>', 'exec')
  except SyntaxError:
    return 'syntax'
  try:
    with sb._time_limit(timeout_s):
      ns = sb._safe_exec_no_timer(code)
      fn = ns.get('simple_math_problem')
      if fn is None:
        return 'no_fn'
      out_buf = io.StringIO()
      with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(io.StringIO()):
        out = fn()
  except (sb._Timeout, Exception):
    return 'runtime'
  gold = sb._extract_gold_answer(answer)
  out = sb._to_number(out)
  if out is None:
    out = sb._extract_text_answer(out_buf.getvalue())
  return 'correct' if sb._numbers_equal(out, gold) else 'wrong'


def main():
  cats = ['correct', 'wrong', 'runtime', 'syntax', 'no_fn']
  print('| run | ' + ' | '.join(cats) + ' | `"""` | `\'\'\'` | mean chars |')
  print('|---' * (len(cats) + 4) + '|')
  for run in sys.argv[1:]:
    with open(os.path.join(run, 'samples', 'results.json')) as f:
      recs = json.load(f)['records']
    c = {k: 0 for k in cats}
    for r in recs:
      c[category(r['response'], r['response_ground_truth'])] += 1
    dq = sum('"""' in r['response'] for r in recs)
    sq = sum("'''" in r['response'] for r in recs)
    ln = sum(len(r['response']) for r in recs) / len(recs)
    print(f'| {os.path.basename(run.rstrip("/"))} | ' + ' | '.join(str(c[k]) for k in cats)
          + f' | {dq} | {sq} | {ln:.0f} |')


if __name__ == '__main__':
  main()
