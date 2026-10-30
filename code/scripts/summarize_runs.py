"""Summaries for GSM8K eval runs under runs/.

  python scripts/summarize_runs.py metrics runs/<name>
      Re-scores samples/results.json and writes runs/<name>/metrics.json
      (accuracy, CI, timing, peak memory, per-question correctness string
      so paired tests can be done later without the generations).

  python scripts/summarize_runs.py compare runs/<a> runs/<b>
      Paired comparison on the same questions (exact McNemar test).

  python scripts/summarize_runs.py table runs/<a> runs/<b> ...
      Markdown table of the given runs.
"""
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import sandbox_gsm8k


def read_info(run_dir):
  info = {}
  with open(os.path.join(run_dir, 'run_info.txt')) as f:
    for line in f:
      if ': ' in line:
        k, v = line.rstrip('\n').split(': ', 1)
        info[k] = v
  return info


def sample_seconds(run_dir):
  """Sampling time from the last tqdm bar in log.txt (excludes model load)."""
  try:
    with open(os.path.join(run_dir, 'log.txt')) as f:
      bars = re.findall(r'sampling: 100%[^\[]*\[([\d:]+)<', f.read())
  except FileNotFoundError:
    return None
  if not bars:
    return None
  sec = 0
  for part in bars[-1].split(':'):
    sec = 60 * sec + int(part)
  return sec


def metrics(run_dir):
  with open(os.path.join(run_dir, 'samples', 'results.json')) as f:
    res = json.load(f)
  correct = ''.join(
    '1' if sandbox_gsm8k.evaluate_samples(
      r['response'], r['response_ground_truth'], timeout_s=5.0) else '0'
    for r in res['records'])
  info = read_info(run_dir)
  n = len(correct)
  out = {
    'split': info.get('split'), 'sampler': info.get('sampler'),
    'steps': int(info['steps']), 'temperature': float(info['temperature']),
    'batch': int(info['batch']), 'seed': int(info['seed']),
    'git_commit': info.get('git_commit'), 'gpu': info.get('gpu'),
    'num_total': n, 'num_correct': correct.count('1'),
    'accuracy': correct.count('1') / n,
    'bootstrap_mean': res['accuracy'],
    'ci95': [res['accuracy_ci_lo'], res['accuracy_ci_hi']],
    'wall_seconds': int(info.get('wall_seconds', -1)),
    'sample_seconds': sample_seconds(run_dir),
    'peak_gpu_mem_mib': int(info['peak_gpu_mem_mib'])
      if info.get('peak_gpu_mem_mib', '').isdigit() else None,
    'per_question_correct': correct,
  }
  # The upstream scorer and our re-score must agree.
  assert out['num_correct'] == res['num_correct'], (
    out['num_correct'], res['num_correct'])
  with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
    json.dump(out, f, indent=1)
  print(f"{run_dir}: {out['num_correct']}/{n} = {out['accuracy']:.4f}")
  return out


def load_metrics(run_dir):
  with open(os.path.join(run_dir, 'metrics.json')) as f:
    return json.load(f)


def mcnemar(a, b):
  """Exact two-sided McNemar test on paired 0/1 strings."""
  assert len(a) == len(b)
  n01 = sum(x == '0' and y == '1' for x, y in zip(a, b))
  n10 = sum(x == '1' and y == '0' for x, y in zip(a, b))
  n = n01 + n10
  if n == 0:
    return n01, n10, 1.0
  k = min(n01, n10)
  p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
  return n01, n10, min(1.0, 2 * p)


def compare(dir_a, dir_b):
  a, b = load_metrics(dir_a), load_metrics(dir_b)
  n01, n10, p = mcnemar(a['per_question_correct'], b['per_question_correct'])
  line = (f"{os.path.basename(dir_a)} {a['accuracy']:.4f} vs "
          f"{os.path.basename(dir_b)} {b['accuracy']:.4f}: "
          f"only-b-correct={n01} only-a-correct={n10} McNemar p={p:.3f}")
  print(line)
  return line


def table(dirs):
  rows = ['| run | split | sampler | steps | T | acc | 95% CI | wall (s) | peak mem (MiB) |',
          '|---|---|---|---|---|---|---|---|---|']
  for d in dirs:
    if not os.path.exists(os.path.join(d, 'metrics.json')):
      rows.append(f'| {os.path.basename(d)} | FAILED / missing | | | | | | | |')
      continue
    m = load_metrics(d)
    rows.append(
      f"| {os.path.basename(d)} | {m['split']} | {m['sampler']} | {m['steps']} "
      f"| {m['temperature']} | {100 * m['accuracy']:.2f}% "
      f"({m['num_correct']}/{m['num_total']}) "
      f"| [{100 * m['ci95'][0]:.1f}, {100 * m['ci95'][1]:.1f}] "
      f"| {m['wall_seconds']} | {m['peak_gpu_mem_mib']} |")
  text = '\n'.join(rows)
  print(text)
  return text


if __name__ == '__main__':
  cmd, args = sys.argv[1], sys.argv[2:]
  if cmd == 'metrics':
    metrics(args[0])
  elif cmd == 'compare':
    compare(args[0], args[1])
  elif cmd == 'table':
    table(args)
  else:
    raise ValueError(cmd)
