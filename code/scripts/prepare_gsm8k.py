"""Build the fixed GSM8K splits used throughout the project (D08).

Outputs:
  data/splits/gsm8k_dev_ids.json   committed; indices into GSM8K train (openai/gsm8k, main)
  <out_dir>/gsm8k_test.json        1319 records, upstream format {prompt, response_ground_truth}
  <out_dir>/gsm8k_dev.json         1000 records, same format, in the order of dev_ids
  <out_dir>/gsm8k_dev200.json      first 200 of dev (speed tests, sampler check)
  <out_dir>/gsm8k_train.json       6473 records, train minus dev

The dev ids are generated once and then only read. Re-running with an existing
ids file verifies it instead of overwriting it.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

SEED = 42
DEV_SIZE = 1000


def to_records(df, ids):
  return [{'gsm8k_index': int(i),
           'prompt': df.iloc[i]['question'],
           'response_ground_truth': df.iloc[i]['answer']} for i in ids]


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--gsm8k_dir', default='/root/autodl-tmp/hf/gsm8k/main')
  parser.add_argument('--out_dir', default='/root/autodl-tmp/data')
  parser.add_argument('--ids_path', default=os.path.join(
    os.path.dirname(__file__), '../../data/splits/gsm8k_dev_ids.json'))
  args = parser.parse_args()

  train = pd.read_parquet(
    os.path.join(args.gsm8k_dir, 'train-00000-of-00001.parquet'))
  test = pd.read_parquet(
    os.path.join(args.gsm8k_dir, 'test-00000-of-00001.parquet'))
  assert len(train) == 7473 and len(test) == 1319

  rng = np.random.default_rng(SEED)
  # Kept in permutation order, so any prefix (e.g. dev 200) is a random subset.
  dev_ids = [int(i) for i in rng.permutation(len(train))[:DEV_SIZE]]
  ids_path = os.path.abspath(args.ids_path)
  if os.path.exists(ids_path):
    with open(ids_path) as f:
      saved = json.load(f)
    assert saved['dev_ids'] == dev_ids, 'existing dev split differs'
    print(f'Verified existing split: {ids_path}')
  else:
    os.makedirs(os.path.dirname(ids_path), exist_ok=True)
    with open(ids_path, 'w') as f:
      json.dump({'source': 'openai/gsm8k main train (7473)',
                 'seed': SEED,
                 'method': 'numpy default_rng(seed).permutation(n)[:1000], permutation order',
                 'dev_ids': dev_ids}, f, indent=1)
    print(f'Wrote split: {ids_path}')

  dev_set = set(dev_ids)
  train_ids = [i for i in range(len(train)) if i not in dev_set]
  os.makedirs(args.out_dir, exist_ok=True)
  outputs = {
    'gsm8k_test.json': to_records(test, range(len(test))),
    'gsm8k_dev.json': to_records(train, dev_ids),
    'gsm8k_dev200.json': to_records(train, dev_ids[:200]),
    'gsm8k_train.json': to_records(train, train_ids),
  }
  for name, records in outputs.items():
    with open(os.path.join(args.out_dir, name), 'w') as f:
      json.dump(records, f, indent=1)
    print(f'{name}: {len(records)}')


if __name__ == '__main__':
  main()
