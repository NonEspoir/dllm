"""13-gram overlap between TinyGSM (pre-training data) and GSM8K splits.

For every GSM8K question (test / dev / train minus dev) we take its word
13-grams (lowercased, letters and digits only) and look them up in every
TinyGSM question and code string. Reported per split:
  - share of questions with at least one 13-gram found anywhere in TinyGSM;
  - for each question, the single TinyGSM item covering most of its 13-grams
    (coverage = matched / total); >= 0.5 is called a near-duplicate;
  - questions shorter than 13 words (no 13-gram, cannot be checked).
TinyGSM was generated from GSM8K *train* questions as seeds, so dev / train
are expected to overlap more than test (docs/04 §3.5).

  python scripts/contamination.py --tinygsm_dir /root/autodl-tmp/hf/tinygsm_full \
      --out ../runs/1030-contamination-13gram
"""
import argparse
import collections
import glob
import json
import multiprocessing as mp
import os
import re
import time

N = 13
DATA = '/root/autodl-tmp/data'
WORD = re.compile(r'[a-z0-9]+')

_TARGET = None  # ngram -> list of (split, qid); set in the parent before fork


def words(text):
  return WORD.findall(text.lower().replace('’', "'"))


def ngrams(ws):
  return [' '.join(ws[i:i + N]) for i in range(len(ws) - N + 1)]


def scan(path):
  """Returns, for one parquet shard: best TinyGSM item per question and the
  set of matched ngrams."""
  import pandas as pd
  df = pd.read_parquet(path)
  best = {}  # (split, qid) -> (count, shard, row, field)
  matched = set()
  shard = os.path.basename(path)
  for field in ('question', 'code'):
    for row, text in enumerate(df[field].values):
      hits = [g for g in ngrams(words(text)) if g in _TARGET]
      if not hits:
        continue
      matched.update(hits)
      cnt = collections.Counter(q for g in set(hits) for q in _TARGET[g])
      for q, c in cnt.items():
        if q not in best or c > best[q][0]:
          best[q] = (c, shard, row, field)
  return shard, len(df), best, matched


def main():
  global _TARGET
  p = argparse.ArgumentParser()
  p.add_argument('--tinygsm_dir', required=True)
  p.add_argument('--out', required=True)
  p.add_argument('--procs', type=int, default=17)
  args = p.parse_args()

  questions, grams = {}, {}
  target = collections.defaultdict(list)
  for split in ('test', 'dev', 'train'):
    with open(os.path.join(DATA, f'gsm8k_{split}.json')) as f:
      for r in json.load(f):
        q = (split, r['gsm8k_index'])
        questions[q] = r['prompt']
        grams[q] = set(ngrams(words(r['prompt'])))
        for g in grams[q]:
          target[g].append(q)
  _TARGET = dict(target)
  shards = sorted(glob.glob(os.path.join(args.tinygsm_dir, '**', '*.parquet'),
                            recursive=True))
  print(f'{len(questions)} GSM8K questions, {len(_TARGET)} distinct 13-grams, '
        f'{len(shards)} TinyGSM shards', flush=True)

  t0 = time.time()
  best, matched, n_rows = {}, set(), 0
  with mp.get_context('fork').Pool(args.procs) as pool:
    for shard, n, b, m in pool.imap_unordered(scan, shards):
      n_rows += n
      matched |= m
      for q, v in b.items():
        if q not in best or v[0] > best[q][0]:
          best[q] = v
      print(f'  {shard}: {n} rows, {time.time() - t0:.0f} s', flush=True)

  os.makedirs(args.out, exist_ok=True)
  per_split = {}
  examples = collections.defaultdict(list)
  for split in ('test', 'dev', 'train'):
    qs = [q for q in questions if q[0] == split]
    checkable = [q for q in qs if grams[q]]
    any_hit = [q for q in checkable if grams[q] & matched]
    cov = {q: best[q][0] / len(grams[q]) for q in checkable if q in best}
    near = sorted((q for q in cov if cov[q] >= 0.5), key=lambda q: -cov[q])
    per_split[split] = {
      'questions': len(qs), 'checkable': len(checkable),
      'too_short': len(qs) - len(checkable),
      'any_13gram_in_tinygsm': len(any_hit),
      'any_13gram_rate': len(any_hit) / len(checkable),
      'best_item_coverage_ge_0.5': len(near),
      'best_item_coverage_ge_0.5_rate': len(near) / len(checkable),
      'best_item_coverage_ge_0.8': sum(cov[q] >= 0.8 for q in cov),
      'mean_best_item_coverage': sum(cov.values()) / len(checkable),
      'near_duplicate_ids': [q[1] for q in near],
    }
    for q in sorted(cov, key=lambda q: -cov[q])[:5]:
      examples[split].append({'gsm8k_index': q[1], 'coverage': round(cov[q], 3),
                              'tinygsm': list(best[q][1:]), 'question': questions[q]})

  # Fetch the TinyGSM text of the shown examples.
  import pandas as pd
  by_shard = collections.defaultdict(list)
  for split in examples:
    for e in examples[split]:
      by_shard[e['tinygsm'][0]].append(e)
  for shard, es in by_shard.items():
    path = next(s for s in shards if os.path.basename(s) == shard)
    df = pd.read_parquet(path)
    for e in es:
      e['tinygsm_text'] = df.iloc[e['tinygsm'][1]][e['tinygsm'][2]]

  out = {'n': N, 'tinygsm_rows': n_rows, 'tinygsm_shards': len(shards),
         'normalization': 'lowercase, [a-z0-9]+ words, curly apostrophe -> straight',
         'fields_checked': ['question', 'code'], 'seconds': round(time.time() - t0),
         'per_split': per_split, 'examples': examples}
  with open(os.path.join(args.out, 'metrics.json'), 'w') as f:
    json.dump(out, f, indent=1)
  for split, v in per_split.items():
    print(split, {k: (round(x, 4) if isinstance(x, float) else x)
                  for k, x in v.items() if k != 'near_duplicate_ids'})


if __name__ == '__main__':
  main()
