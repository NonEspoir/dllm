"""Demo: the same question through several checkpoints, side by side (D7 展示).

Generates one answer per (question, checkpoint) with the same seed and decoding
as the evaluation (FastAncestralSampler, 512 steps, skip_unchanged), executes it
with the eval's own scorer, and prints a card per question:

  question, gold answer,
  for each checkpoint: the code, the value it returns / prints, the verdict
  (correct / runs but wrong / error) and why.

Takes test or dev questions by their index in the split file; the ids are
printed so the report can say how the samples were chosen (do not hand-pick the
prettiest ones -- say how they were picked).

  python -m scripts.demo --split dev --ids 3 17 42 \
      --ckpts base:/root/autodl-tmp/checkpoints/tinygsm/mdlm.ckpt \
              rl:/root/autodl-tmp/rl_ckpt/1101-grpo-conv-fmt0.1-kl0.04/step400.ckpt
  # add --out runs/demo.json to keep the cards for the report
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts import sft  # noqa: E402


def load_split(split):
  path = f'/root/autodl-tmp/data/gsm8k_{split}.json'
  with open(path) as f:
    return json.load(f)


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--split', default='dev', choices=['dev', 'test', 'train'])
  p.add_argument('--ids', type=int, nargs='+', required=True,
                 help='index within the split file')
  p.add_argument('--ckpts', nargs='+', required=True, help='name:path ...')
  p.add_argument('--steps', type=int, default=512)
  p.add_argument('--temperature', type=float, default=0.1,
                 help='0.1 = the best decoding setting; 1.0 shows more variety')
  p.add_argument('--seed', type=int, default=1)
  p.add_argument('--out', default=None)
  args = p.parse_args()

  from rl import reward, rollout
  import dataloader
  import omegaconf
  torch.manual_seed(args.seed)
  tok = dataloader.get_tokenizer(omegaconf.OmegaConf.create(
    {'data': {'tokenizer_name_or_path': 'HuggingFaceTB/SmolLM-135M'}}))
  recs = load_split(args.split)
  picked = [recs[i] for i in args.ids]
  cards = []
  for spec in args.ckpts:
    name, path = spec.split(':', 1)
    config = sft.build_config(path)
    model = sft.load_model(config, tok, path).to('cuda')
    model.backbone.eval()
    ro = rollout.generate(model, tok, [r['prompt'] for r in picked],
                          [r['response_ground_truth'] for r in picked],
                          group_size=1, num_steps=args.steps,
                          temperature=args.temperature,
                          separator=config.data.separator, batch_size=len(picked))
    cats = rollout.grade(ro.responses, [r['response_ground_truth'] for r in picked])
    for i, rec in enumerate(picked):
      cards.append({'ckpt': name, 'split': args.split, 'index': args.ids[i],
                    'question': rec['prompt'].strip(), 'gold': rec['response_ground_truth'],
                    'code': ro.responses[i], 'category': cats[i]})
    del model
    torch.cuda.empty_cache()

  by_q = {}
  for c in cards:
    by_q.setdefault((c['split'], c['index']), []).append(c)
  for (split, idx), group in by_q.items():
    print('=' * 78)
    print(f'[{split} #{idx}] {group[0]["question"][:400]}')
    print(f'gold: {group[0]["gold"].strip()[-80:]}')
    for c in group:
      print('-' * 78)
      print(f'{c["ckpt"]}: {c["category"]}')
      print(c['code'].strip()[:1200])
  if args.out:
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
      json.dump(cards, f, ensure_ascii=False, indent=1)
    print(f'wrote {args.out}')


if __name__ == '__main__':
  main()
