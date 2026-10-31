"""Write the GRPO runs' logs (train_log.jsonl + dev evals) as TensorBoard event
files under /root/tf-logs (the directory AutoDL's TensorBoard serves). Our
training scripts log to JSON only; this converts after the fact and can be
re-run any time (each run's event directory is rewritten).

  python scripts/runs_to_tb.py                 # all GRPO runs except smoke tests
  python scripts/runs_to_tb.py 1031-grpo-base-fmt0.1-kl0.04

Tags: 0_key/* the curves to read first: 25-step trailing means of the
training-batch accuracy, reward, action log-prob and runtime-error share (a
single step is dominated by which 32 questions it drew), plus dev200 at both
temperatures. train/* every step, raw (train/loss is the GRPO surrogate: group
advantages have mean 0, so its value stays near 0 and says nothing about
progress; only its gradient matters). eval/dev200_T1.0 and eval/dev200_T0.1 at
the eval steps; the
pseudo-run 'base-reference' draws base's dev200 values (same protocol) as flat
lines. Runs logged before R43 have 'kl' too small by about the number of chunks
(docs/09 §2.2): written as train/kl_logged_old plus train/kl_est = logged x
ceil(kept_rows / 8) (approximate).
"""
import glob
import json
import math
import os
import shutil
import sys

from torch.utils.tensorboard import SummaryWriter

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
TB = '/root/tf-logs'
BASE_DEV200 = {'1.0': 0.215, '0.1': 0.39}  # base, 512 steps + skip, seed 1 (06 D4)
SKIP = ('smoke', 'crashed')


def evals(run_dir):
  out = []
  path = os.path.join(run_dir, 'stdout.txt')
  if os.path.exists(path):
    with open(path) as f:
      out = [json.loads(l[5:]) for l in f if l.startswith('EVAL ')]
  return out


def convert(run_dir):
  name = os.path.basename(run_dir)
  logdir = os.path.join(TB, name)
  shutil.rmtree(logdir, ignore_errors=True)
  w = SummaryWriter(logdir)
  with open(os.path.join(run_dir, 'train_log.jsonl')) as f:
    rows = [json.loads(l) for l in f]
  new_kl = any('grad_norm_max' in r for r in rows)
  for r in rows:
    s = r['step']
    for k, v in r.items():
      if k in ('step', 'skipped') or not isinstance(v, (int, float)) or v is None:
        continue
      if k == 'kl' and not new_kl:
        w.add_scalar('train/kl_logged_old', v, s)
        w.add_scalar('train/kl_est', v * math.ceil(r.get('kept_rows', 8) / 8), s)
        continue
      w.add_scalar(f'train/{k}', v, s)
  key = {'correct': 'train_correct_ma25', 'reward': 'train_reward_ma25',
         'action_logp': 'action_logp_ma25', 'cat_runtime': 'runtime_error_ma25'}
  for i, r in enumerate(rows):
    win = rows[max(0, i - 24):i + 1]
    for k, tag in key.items():
      v = [x[k] for x in win if isinstance(x.get(k), (int, float))]
      if len(v) >= 5:
        w.add_scalar(f'0_key/{tag}', sum(v) / len(v), r['step'])
  for e in evals(run_dir):
    for T in ('1.0', '0.1'):
      v = e.get(f'acc_T{T}')
      if v is not None:
        w.add_scalar(f'eval/dev200_T{T}', v, e['step'])
        w.add_scalar(f'0_key/dev200_T{T}', v, e['step'])
  w.close()
  return name, len(rows)


def main():
  os.makedirs(TB, exist_ok=True)
  if len(sys.argv) > 1:
    dirs = [os.path.join(REPO, 'runs', n) for n in sys.argv[1:]]
  else:
    dirs = sorted(os.path.dirname(p) for p in
                  glob.glob(os.path.join(REPO, 'runs', '*grpo*', 'train_log.jsonl')))
    dirs = [d for d in dirs if not any(x in os.path.basename(d) for x in SKIP)]
  for d in dirs:
    print(*convert(d))
  logdir = os.path.join(TB, 'base-reference')
  shutil.rmtree(logdir, ignore_errors=True)
  w = SummaryWriter(logdir)
  for s in (0, 300):
    for T, v in BASE_DEV200.items():
      w.add_scalar(f'eval/dev200_T{T}', v, s)
      w.add_scalar(f'0_key/dev200_T{T}', v, s)
  w.close()


if __name__ == '__main__':
  main()
