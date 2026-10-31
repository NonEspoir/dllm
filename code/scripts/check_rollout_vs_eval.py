"""D3 check (04 §4.3, item 2): accuracy of the RL rollout path vs the eval script.

Generates one answer per dev1000 question with rl.rollout.generate (the RL path:
encode_prompts, FastAncestralSampler with skip_unchanged, rollout scorer) and
compares the accuracy with the eval script's run in the same setting
(runs/1030-sweep-fast-dev-s<steps>-T<T>/metrics.json, D1 sweep, seed 1).
The two runs use different random streams, so the comparison is unpaired
(two-proportion z-test); the rollout scorer itself was already checked against
the upstream one (tests/test_rl_reward.py, 0 disagreements).

  python -m scripts.check_rollout_vs_eval --steps 128 --temps 1.0 0.1
"""
import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts import sft  # noqa: E402


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--ckpt', default=sft.BASE_CKPT)
  p.add_argument('--steps', type=int, default=128)
  p.add_argument('--temps', type=float, nargs='+', default=[1.0, 0.1])
  p.add_argument('--seed', type=int, default=1)
  p.add_argument('--out', default=None)
  args = p.parse_args()

  from rl import rollout
  import dataloader
  import omegaconf
  torch.manual_seed(args.seed)
  tok = dataloader.get_tokenizer(omegaconf.OmegaConf.create(
    {'data': {'tokenizer_name_or_path': 'HuggingFaceTB/SmolLM-135M'}}))
  config = sft.build_config(args.ckpt)
  model = sft.load_model(config, tok, args.ckpt).to('cuda')
  model.backbone.eval()
  with open('/root/autodl-tmp/data/gsm8k_dev.json') as f:
    dev = json.load(f)
  with open(sft.DEV_IDS) as f:
    assert [r['gsm8k_index'] for r in dev] == json.load(f)['dev_ids']

  res = []
  for T in args.temps:
    ro = rollout.generate(model, tok, [r['prompt'] for r in dev],
                          [r['response_ground_truth'] for r in dev],
                          group_size=1, num_steps=args.steps, temperature=T,
                          separator=config.data.separator, batch_size=200)
    k1, n1 = int(ro.rewards.sum()), len(dev)
    ref = os.path.join(sft.REPO, 'runs', f'1030-sweep-fast-dev-s{args.steps}-T{T}', 'metrics.json')
    with open(ref) as f:
      m = json.load(f)
    k2, n2 = m['num_correct'], m['num_total']
    p1, p2 = k1 / n1, k2 / n2
    pp = (k1 + k2) / (n1 + n2)
    z = (p1 - p2) / math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    pval = math.erfc(abs(z) / math.sqrt(2))
    r = {'steps': args.steps, 'T': T, 'rollout_acc': p1, 'eval_acc': p2,
         'diff_pp': round(100 * (p1 - p2), 2), 'z': round(z, 3), 'p': round(pval, 4)}
    print(json.dumps(r), flush=True)
    res.append(r)
  if args.out:
    with open(args.out, 'w') as f:
      json.dump(res, f, indent=1)


if __name__ == '__main__':
  main()
