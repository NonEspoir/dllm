"""Semantic checks of the DeepSeek SFT data with DeepSeek in thinking mode
(docs/08 Q13-Q14). Two prompts:

  adjudicate  for samples whose execution != gold: show the question, the
              program and GSM8K's reference solution, ask which is right.
  audit       blind check for samples that passed: show only question and
              program (no gold, no execution result), ask whether every step
              is faithful to the problem.

  python scripts/sft_verify.py tasks --mode adjudicate --out .../adj_tasks.jsonl
  python /root/autodl-tmp/ds-api-tools/dsgen.py batch --in .../adj_tasks.jsonl \
      --out .../adj_raw.jsonl --thinking --json --max-tokens 16384 --workers 32
  python scripts/sft_verify.py collect --mode adjudicate --raw .../adj_raw.jsonl \
      --out .../adj.jsonl
"""
import argparse
import json
import os

DATA = os.path.join(os.path.dirname(__file__), '../../data/sft/deepseek_v1.jsonl')

ADJUDICATE = '''Below is a grade-school math problem, a Python program that tries to solve it, and a reference solution written by a human annotator. They reach DIFFERENT final answers. Decide carefully which one is correct by solving the problem yourself step by step. Read the problem literally; do not assume facts that are not stated, except standard common knowledge (e.g. 7 days a week, 12 in a dozen).

Problem:
{question}

Program:
{code}

Reference solution:
{reference}

Return a JSON object with these fields:
- "correct_answer": your own final numeric answer (a number), or null if the problem is ambiguous
- "verdict": one of "program_correct", "reference_correct", "both_wrong", "ambiguous"
- "error_step": one sentence naming the first step where the wrong solution goes wrong (or why the problem is ambiguous)'''

AUDIT = '''Below is a grade-school math problem and a Python program written to solve it. Check whether the program's reasoning is correct: every quantity must be read correctly from the problem, every step must follow from the problem statement, and the final value must answer exactly what is asked. Standard common knowledge (e.g. 7 days a week, 12 in a dozen) is fine. Solve the problem yourself first, then compare.

Problem:
{question}

Program:
{code}

Return a JSON object with these fields:
- "your_answer": your own final numeric answer (a number), or null if the problem is ambiguous
- "verdict": one of "correct", "wrong", "ambiguous"
- "error_line": if wrong, the first incorrect line of the program copied verbatim, else null
- "reason": one sentence'''


def load(path=DATA):
  with open(path) as f:
    return [json.loads(l) for l in f]


def cmd_tasks(a):
  rows = load()
  extra = set()
  if a.include_ids:
    with open(a.include_ids) as f:
      extra = set(json.load(f))
  out = []
  for r in rows:
    if a.mode == 'adjudicate':
      if r['checks']['exec_correct']:
        continue
      prompt = ADJUDICATE.format(question=r['question'], code=r['code'],
                                 reference=r['reference_solution'])
    else:
      if not (r['hard_pass'] or r['gsm8k_index'] in extra):
        continue
      prompt = AUDIT.format(question=r['question'], code=r['code'])
    out.append({'id': str(r['gsm8k_index']), 'prompt': prompt})
  os.makedirs(os.path.dirname(a.out), exist_ok=True)
  with open(a.out, 'w') as f:
    for o in out:
      f.write(json.dumps(o, ensure_ascii=False) + '\n')
  print(f'{len(out)} tasks -> {a.out}')


def cmd_collect(a):
  res = {}
  with open(a.raw) as f:
    for line in f:
      o = json.loads(line)
      if o.get('ok') and o.get('json_ok'):
        res[o['id']] = o
  rows = {str(r['gsm8k_index']): r for r in load()}
  out = []
  for gid, o in sorted(res.items(), key=lambda kv: int(kv[0])):
    d = o['data']
    out.append({'gsm8k_index': int(gid), 'gold_answer': rows[gid]['gold_answer'],
                'exec_correct': rows[gid]['checks']['exec_correct'], **d,
                'completion_tokens': o['usage']['completion_tokens']})
  with open(a.out, 'w') as f:
    for o in out:
      f.write(json.dumps(o, ensure_ascii=False) + '\n')
  from collections import Counter
  print(len(out), 'collected;', Counter(o['verdict'] for o in out))
  print('completion tokens:', sum(o['completion_tokens'] for o in out))


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  sub = p.add_subparsers(dest='cmd', required=True)
  t = sub.add_parser('tasks')
  t.add_argument('--mode', choices=['adjudicate', 'audit'], required=True)
  t.add_argument('--out', required=True)
  t.add_argument('--include_ids', default=None,
                 help='JSON list of extra gsm8k_index to audit (calibration)')
  c = sub.add_parser('collect')
  c.add_argument('--mode', choices=['adjudicate', 'audit'], required=True)
  c.add_argument('--raw', required=True)
  c.add_argument('--out', required=True)
  a = p.parse_args()
  {'tasks': cmd_tasks, 'collect': cmd_collect}[a.cmd](a)
