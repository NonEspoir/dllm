"""Diagnostics for the SFT result: why does T=1 get worse, and what kind of
questions flip. CPU only.

  python scripts/diagnose_sft.py --pairs base:../runs/1030-eval-base-s512-T1.0 ...

For each run: re-extracts and executes the code exactly like the scorer, then
records the exception class, the line where it failed (relative to the code the
scorer extracted), whether the first line matches the SFT template, and the
quote style. Also dumps the flipped questions of a chosen pair for eyeballing.
"""
import argparse
import collections
import contextlib
import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import sandbox_gsm8k as sb  # noqa: E402

TEMPLATE = re.compile(r'^\s*def\s+simple_math_problem\s*\(\s*\)\s*(->\s*int\s*)?:')


def run_one(sample, answer, timeout_s=5.0):
  code = sb._extract_code(sample)
  first = next((l for l in sample.splitlines() if l.strip()), '')
  rec = {'tmpl': bool(TEMPLATE.match(first)),
         'dq': '"""' in sample, 'sq': "'''" in sample,
         'lines': len(code.splitlines())}
  try:
    compile(code, '<sample>', 'exec')
  except SyntaxError as e:
    rec['cat'], rec['err'] = 'syntax', type(e).__name__
    rec['at'] = (e.lineno or 0) / max(1, rec['lines'])
    return rec
  try:
    with sb._time_limit(timeout_s):
      ns = sb._safe_exec_no_timer(code)
      fn = ns.get('simple_math_problem')
      if fn is None:
        rec['cat'], rec['err'] = 'no_fn', ''
        return rec
      buf = io.StringIO()
      with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        out = fn()
  except sb._Timeout:
    rec['cat'], rec['err'] = 'runtime', 'Timeout'
    return rec
  except Exception as e:  # noqa: BLE001 - this is the point: classify it
    rec['cat'], rec['err'] = 'runtime', type(e).__name__
    if e.__traceback__ is not None:
      tb = e.__traceback__
      while tb.tb_next is not None:
        tb = tb.tb_next
      rec['at'] = tb.tb_lineno / max(1, rec['lines'])
    return rec
  gold = sb._extract_gold_answer(answer)
  val = sb._to_number(out)
  if val is None:
    val = sb._extract_text_answer(buf.getvalue())
  rec['cat'] = 'correct' if sb._numbers_equal(val, gold) else 'wrong'
  rec['err'] = ''
  return rec


def load(run):
  with open(os.path.join(run, 'samples', 'results.json')) as f:
    recs = json.load(f)['records']
  out = [run_one(r['response'], r['response_ground_truth']) for r in recs]
  return recs, out


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--pairs', action='append', required=True,
                 help='name=run_a,run_b (b is the treatment)')
  p.add_argument('--flips', default='',
                 help='name=run_a,run_b; dump the flipped questions of this pair')
  p.add_argument('--n_show', type=int, default=6)
  args = p.parse_args()

  data = {}
  for spec in args.pairs + ([args.flips] if args.flips else []):
    name, runs = spec.split('=')
    a, b = runs.split(',')
    for r in (a, b):
      if r not in data:
        data[r] = load(r)
    (ra, oa), (rb, ob) = data[a], data[b]
    print(f'\n### {name}   A={os.path.basename(a)}   B={os.path.basename(b)}')
    for label, recs, outs in (('A', ra, oa), ('B', rb, ob)):
      cats = collections.Counter(o['cat'] for o in outs)
      errs = collections.Counter(o['err'] for o in outs if o['err'])
      at = [o['at'] for o in outs if 'at' in o]
      print(f'{label}: ' + ' '.join(f'{k}={cats[k]}' for k in
                                    ('correct', 'wrong', 'runtime', 'syntax', 'no_fn'))
            + f'  template {sum(o["tmpl"] for o in outs)}/1000'
            + f'  dq {sum(o["dq"] for o in outs)}  sq {sum(o["sq"] for o in outs)}'
            + f'  mean lines {sum(o["lines"] for o in outs) / len(outs):.1f}'
            + (f'  fail-line quantiles {sorted(round(x, 2) for x in at)[::max(1, len(at) // 5)]}'
               if at else ''))
      print(f'   exceptions: {dict(errs.most_common(8))}')

    # flips
    flip = [(i, ra[i], rb[i], oa[i], ob[i]) for i in range(min(len(ra), len(rb)))
            if oa[i]['cat'] != ob[i]['cat']
            and {oa[i]['cat'], ob[i]['cat']} & {'correct'}]
    if flip:
      kinds = collections.Counter(f'{x[3]["cat"]}->{x[4]["cat"]}' for x in flip)
      print(f'   flips involving "correct": {len(flip)}  {dict(kinds)}')
      b_only = [x for x in flip if x[3]['cat'] != 'correct']
      a_only = [x for x in flip if x[4]['cat'] != 'correct']
      if b_only:
        print('   base-correct -> B failed: ' + str(collections.Counter(
          f'{x[4]["cat"]}/{x[4]["err"]}' for x in b_only).most_common(6)))
      if a_only:
        print('   A failed -> B correct: ' + str(collections.Counter(
          f'{x[3]["cat"]}/{x[3]["err"]}' for x in a_only).most_common(6)))
    if args.flips and name == args.flips.split('=')[0]:
      shown = 0
      for i, xa, xb, oa_, ob_ in flip:
        if oa_['cat'] == 'correct' and ob_['cat'] != 'correct' and shown < args.n_show:
          shown += 1
          print(f'\n--- flip {i}: base correct, B {ob_["cat"]}/{ob_["err"]} '
                f'(fail at {ob_.get("at", 0):.2f} of code)')
          print('Q:', xa['prompt'][:300].replace('\n', ' '))
          print('B response:', xb['response'][:700].replace('\n', ' | '))


if __name__ == '__main__':
  main()
