"""SFT data from DeepSeek (D09 option A): GSM8K train question -> TinyGSM-style
Python, then filtered. Execution agreement with gold is necessary but not
sufficient (docs/08 Q11), so every sample is also tagged with checks that
catch "right number, wrong reasoning".

  # 1. build task file (train split, 6473 questions)
  python scripts/sft_data.py tasks --out /root/autodl-tmp/sft/tasks.jsonl
  # 2. call DeepSeek (resumable); see /root/autodl-tmp/ds-api-tools/README.md
  python /root/autodl-tmp/ds-api-tools/dsgen.py batch \
      --in /root/autodl-tmp/sft/tasks.jsonl --out /root/autodl-tmp/sft/raw.jsonl \
      --system "$(python scripts/sft_data.py system)" --temperature 0.7 \
      --max-tokens 1024 --workers 16
  # 3. score / filter / write dataset + stats
  python scripts/sft_data.py build --raw /root/autodl-tmp/sft/raw.jsonl \
      --out_dir /root/autodl-tmp/sft

Outputs of `build`:
  all.jsonl       every sample with its check results
  filtered.json   passes the hard checks (format, execution == gold, result not
                  a typed-in literal, <= 512 tokens); `suspicious` marks samples
                  that pass but trip a soft check
  unfiltered.json every question, one generation, only the length check (D10)
  stats.json      counts per check
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import sandbox_gsm8k

SYSTEM = '''You convert grade-school math word problems into short Python programs.

Output format, exactly (no markdown fences, no text before or after):

def simple_math_problem() -> int:
    """
    <the problem text, copied verbatim, one sentence per line>
    """
    <one variable per quantity from the problem, with a descriptive snake_case name>
    <intermediate steps, one computation per line, each stored in a named variable>
    result = <final variable>
    return result

Rules:
- Every number you use must come from the problem text, or be a standard unit
  conversion (e.g. 60 minutes per hour, 12 per dozen, 7 days per week, 100 for percent).
- Never write the final answer as a literal. Compute it from the inputs.
- One arithmetic operation or a short expression per line; no loops unless the
  problem really needs one; no imports except math; no print.
- Keep the return annotation "-> int" even if the result is a float.
- Keep it short: typically 4-12 lines of code.'''

EXAMPLE_Q = ("Julie bought a dress that cost $50, but there was a 25% discount. "
             "How much did Julie pay?")
EXAMPLE_A = '''def simple_math_problem() -> int:
    """
    Julie bought a dress that cost $50, but there was a 25% discount.
    How much did Julie pay?
    """
    original_price = 50
    discount_percent = 25
    discount_amount = original_price * discount_percent / 100
    final_price = original_price - discount_amount
    result = final_price
    return result'''

USER_TEMPLATE = ('Example problem:\n' + EXAMPLE_Q + '\n\nExample output:\n'
                 + EXAMPLE_A + '\n\nNow convert this problem:\n{question}')

# Numbers allowed in code without appearing in the question.
ALLOWED_CONSTANTS = {0, 1, 2, 3, 4, 7, 10, 12, 24, 30, 52, 60, 100, 365, 1000,
                     0.5, 0.01, 0.1, 3600, 1440, 360, 31, 28, 16, 8}
WORD_NUMBERS = {'half': 0.5, 'twice': 2, 'double': 2, 'triple': 3, 'thrice': 3,
                'quarter': 0.25, 'third': 3, 'dozen': 12, 'one': 1, 'two': 2,
                'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7,
                'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12,
                'fifteen': 15, 'twenty': 20, 'thirty': 30, 'forty': 40,
                'fifty': 50, 'hundred': 100, 'thousand': 1000,
                # ordinals used as fractions ("cuts it into fifths")
                'fourth': 4, 'fourths': 4, 'fifth': 5, 'fifths': 5,
                'sixth': 6, 'sixths': 6, 'eighth': 8, 'eighths': 8,
                'tenth': 10, 'tenths': 10, 'thirds': 3, 'quarters': 4,
                # coins (cents)
                'nickel': 5, 'nickels': 5, 'dime': 10, 'dimes': 10,
                'penny': 1, 'pennies': 1}
COIN_VALUES = {'quarter': 25, 'quarters': 25}
ONE_AND_A_HALF = re.compile(r'\b(one|a) and a half\b|\b1 1/2\b')
NUM_RE = re.compile(r'(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?')


def numbers_in(text):
  out = set()
  for m in NUM_RE.findall(text):
    v = float(m.replace(',', ''))
    out.add(v)
    out.add(v / 100)  # "25%" used as 0.25
  low = text.lower()
  for w, v in list(WORD_NUMBERS.items()) + list(COIN_VALUES.items()):
    if re.search(r'\b' + w + r'\b', low):
      out.add(float(v))
  if ONE_AND_A_HALF.search(low):
    out.add(1.5)
  # percentages also appear as growth / discount factors: 20% -> 1.2, 0.8
  for m in re.findall(r'(\d+(?:\.\d+)?)\s*(?:%|percent)', low):
    v = float(m) / 100
    out.update({1 + v, 1 - v})
  return out


def code_body(code):
  """Code without the docstring (where the question numbers live)."""
  return re.sub(r'("""|\'\'\')[\s\S]*?\1', '', code, count=1)


def gold_intermediates(answer):
  """Values in GSM8K's <<expr=value>> annotations."""
  vals = []
  for m in re.findall(r'<<[^=<>]*=([^<>]+)>>', answer):
    try:
      vals.append(float(m.replace(',', '').strip()))
    except ValueError:
      pass
  return vals


def returns_literal(code):
  """True if `result` is (an alias of) a bare numeric literal."""
  assigns = dict(re.findall(r'^\s+(\w+)\s*=\s*(.+?)\s*$', code_body(code), re.M))
  name, seen = 'result', set()
  while name in assigns and name not in seen:
    seen.add(name)
    rhs = assigns[name]
    if re.fullmatch(r'-?\d+(?:\.\d+)?', rhs):
      return True
    if not re.fullmatch(r'\w+', rhs):
      return False
    name = rhs
  return False


def run_code(code):
  """Execute like sandbox_gsm8k and also return all numeric locals."""
  ns = {}
  try:
    with sandbox_gsm8k._time_limit(5.0):
      ns = sandbox_gsm8k._safe_exec_no_timer(code)
      fn = ns.get('simple_math_problem')
      if fn is None:
        return None, None
      # Re-run the body with a tracer to capture intermediate locals.
      captured = {}

      def tracer(frame, event, arg):
        if frame.f_code is fn.__code__ and event == 'return':
          captured.update(frame.f_locals)
        return tracer
      sys.settrace(tracer)
      try:
        out = fn()
      finally:
        sys.settrace(None)
  except Exception:
    return None, None
  locals_num = [float(v) for v in captured.values()
                if isinstance(v, (int, float)) and not isinstance(v, bool)]
  return out, locals_num


def close(a, b):
  return abs(a - b) <= 1e-3 * max(1.0, abs(b))


def check(sample_code, question, answer, tokenizer=None):
  """Return dict of check results for one generated program."""
  code = sample_code.strip()
  if code.startswith('```'):
    code = re.sub(r'^```\w*\n|```\s*$', '', code).strip()
  r = {'code': code}
  r['format_ok'] = bool(
    code.startswith('def simple_math_problem() -> int:')
    and re.search(r'^\s+result = .+\n\s+return result\s*$', code, re.M)
    and ('"""' in code or "'''" in code))
  r['exec_correct'] = bool(sandbox_gsm8k.evaluate_samples(code, answer, 5.0))
  gold = sandbox_gsm8k._extract_gold_answer(answer)
  body = code_body(code)
  # Literal answer: gold value appears as a number in the code body and is
  # not in the question (so it was not derived from inputs).
  q_nums = numbers_in(question)
  body_nums = {float(m.replace(',', '')) for m in NUM_RE.findall(body)}
  # Hard: the returned value is a literal (answer typed in, not computed).
  r['gold_literal'] = returns_literal(code)
  # Soft: the gold value is typed somewhere in the body although it is not
  # in the question and not a common constant.
  r['gold_in_body'] = bool(gold is not None and any(close(v, gold) for v in body_nums)
                           and not any(close(v, gold) for v in q_nums)
                           and gold not in ALLOWED_CONSTANTS)
  foreign = sorted(v for v in body_nums
                   if not any(close(v, q) for q in q_nums)
                   and v not in ALLOWED_CONSTANTS)
  r['foreign_numbers'] = foreign
  r['numbers_ok'] = len(foreign) == 0
  # Cross-check with GSM8K's human intermediate results.
  _, local_vals = run_code(code)
  inter = gold_intermediates(answer)
  if local_vals is None or not inter:
    r['intermediate_match'] = None
  else:
    hit = sum(any(close(v, g) for v in local_vals) for g in inter)
    r['intermediate_match'] = hit / len(inter)
  if tokenizer is not None:
    n = (1 + len(tokenizer(question.strip(), add_special_tokens=False).input_ids)
         + 1 + len(tokenizer(code, add_special_tokens=False).input_ids) + 1)
    r['num_tokens'] = n
    r['length_ok'] = n <= 512
  else:
    r['length_ok'] = True
  # Hard checks are the ones whose failure means the sample is unusable.
  # numbers_ok is a heuristic: most "foreign" numbers are world knowledge
  # (5 weekdays, 2**5 for the 6th month), so it only marks samples as
  # suspicious. See data/sft/README.md for the measured breakdown.
  r['hard_pass'] = (r['format_ok'] and r['exec_correct'] and not r['gold_literal']
                    and r['length_ok'])
  # Soft flag: passes hard checks but reasoning trace diverges from GSM8K's.
  r['suspicious'] = bool(r['hard_pass'] and (
    r['gold_in_body'] or not r['numbers_ok']
    or (r['intermediate_match'] is not None and r['intermediate_match'] < 0.5)))
  return r


def cmd_tasks(args):
  with open(os.path.join(args.data_dir, 'gsm8k_train.json')) as f:
    train = json.load(f)
  os.makedirs(os.path.dirname(args.out), exist_ok=True)
  with open(args.out, 'w') as f:
    for rec in train:
      f.write(json.dumps({
        'id': str(rec['gsm8k_index']),
        'prompt': USER_TEMPLATE.format(question=rec['prompt'].strip()),
      }, ensure_ascii=False) + '\n')
  print(f'{len(train)} tasks -> {args.out}')


def cmd_build(args):
  import transformers
  tok = transformers.AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM-135M')
  with open(os.path.join(args.data_dir, 'gsm8k_train.json')) as f:
    train = {str(r['gsm8k_index']): r for r in json.load(f)}
  raws = {}
  with open(args.raw) as f:
    for line in f:
      o = json.loads(line)
      if o.get('ok'):
        raws[o['id']] = o  # last ok line wins
  rows, stats = [], Counter()
  for gid, o in raws.items():
    rec = train[gid]
    r = check(o['text'], rec['prompt'], rec['response_ground_truth'], tok)
    r.update({'gsm8k_index': int(gid), 'prompt': rec['prompt'],
              'response_ground_truth': rec['response_ground_truth']})
    rows.append(r)
    stats['total'] += 1
    for k in ['format_ok', 'exec_correct', 'gold_literal', 'gold_in_body', 'numbers_ok',
              'length_ok', 'hard_pass', 'suspicious']:
      stats[k] += int(bool(r[k]))
    if r['intermediate_match'] is not None:
      stats['intermediate_checked'] += 1
      stats['intermediate_full_match'] += int(r['intermediate_match'] == 1.0)
  stats['missing_from_raw'] = len(train) - len(raws)
  rows.sort(key=lambda r: r['gsm8k_index'])
  os.makedirs(args.out_dir, exist_ok=True)
  with open(os.path.join(args.out_dir, 'all.jsonl'), 'w') as f:
    for r in rows:
      f.write(json.dumps(r, ensure_ascii=False) + '\n')

  def export(sel):
    return [{'gsm8k_index': r['gsm8k_index'], 'prompt': r['prompt'],
             'response_ground_truth': r['response_ground_truth'],
             'code': r['code'], 'suspicious': r['suspicious']} for r in sel]
  with open(os.path.join(args.out_dir, 'filtered.json'), 'w') as f:
    json.dump(export([r for r in rows if r['hard_pass']]), f, indent=1,
              ensure_ascii=False)
  with open(os.path.join(args.out_dir, 'unfiltered.json'), 'w') as f:
    json.dump(export([r for r in rows if r['length_ok']]), f, indent=1,
              ensure_ascii=False)
  with open(os.path.join(args.out_dir, 'stats.json'), 'w') as f:
    json.dump(dict(stats), f, indent=1)
  print(json.dumps(dict(stats), indent=1))


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  sub = p.add_subparsers(dest='cmd', required=True)
  t = sub.add_parser('tasks')
  t.add_argument('--out', required=True)
  t.add_argument('--data_dir', default='/root/autodl-tmp/data')
  sub.add_parser('system')
  b = sub.add_parser('build')
  b.add_argument('--raw', required=True)
  b.add_argument('--out_dir', required=True)
  b.add_argument('--data_dir', default='/root/autodl-tmp/data')
  a = p.parse_args()
  if a.cmd == 'tasks':
    cmd_tasks(a)
  elif a.cmd == 'system':
    print(SYSTEM)
  else:
    cmd_build(a)
