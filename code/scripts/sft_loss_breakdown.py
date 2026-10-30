"""Where does the SFT loss come from? Splits the masked-token CE into answer /
EOS / PAD positions at fixed masking rates t, for several checkpoints, on the
SFT val split and on TinyGSM pre-training examples in the same layout.

  python -m scripts.sft_loss_breakdown --ckpts base raw /root/autodl-tmp/sft_ckpt/<run>/epoch1.ckpt
"""
import argparse
import json
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts import sft  # noqa: E402

TINYGSM = '/root/autodl-tmp/hf/tinygsm/train-00000-of-00017.parquet'


@torch.no_grad()
def breakdown(model, ids, masks, tokenizer, ts, batch, device):
  pad, eos = tokenizer.pad_token_id, tokenizer.eos_token_id
  out = {}
  for t in ts:
    tot = {'answer': [0.0, 0], 'eos': [0.0, 0], 'pad': [0.0, 0]}
    with torch.random.fork_rng(devices=[device]):
      torch.manual_seed(1234)
      for i in range(0, len(ids), batch):
        x, m = ids[i:i + batch].to(device), masks[i:i + batch].to(device)
        alpha = torch.full((x.shape[0], 1), 1 - t, device=device)
        xt = model.q_xt(x, alpha, use_pure_noise=False, valid_tokens=m)
        logp = model.forward(x0=x, xt=xt, sigma=model._sigma_from_alphat(alpha))
        ce = -torch.gather(logp, -1, x[..., None]).squeeze(-1)
        hit = (xt == model.mask_index) & m.bool()
        for k, sel in [('answer', (x != pad) & (x != eos)), ('eos', x == eos),
                       ('pad', x == pad)]:
          s = hit & sel
          tot[k][0] += float(ce[s].sum())
          tot[k][1] += int(s.sum())
    out[t] = {k: round(v[0] / max(v[1], 1), 4) for k, v in tot.items()}
  return out


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--ckpts', nargs='+', required=True)
  p.add_argument('--select', default='sft_v1b')
  p.add_argument('--n_tinygsm', type=int, default=256)
  p.add_argument('--ts', type=float, nargs='+', default=[0.1, 0.5, 0.9])
  args = p.parse_args()

  import dataloader
  import omegaconf
  tokenizer = dataloader.get_tokenizer(omegaconf.OmegaConf.create(
    {'data': {'tokenizer_name_or_path': 'HuggingFaceTB/SmolLM-135M'}}))
  rows = sft.load_records(args.select)
  ids, masks, _, _ = sft.tokenize(rows, tokenizer)
  g = torch.Generator().manual_seed(1)
  perm = torch.randperm(len(ids), generator=g)
  n_val = int(round(0.02 * len(ids)))
  sets = {'sft_val': (ids[perm[:n_val]], masks[perm[:n_val]])}
  df = pd.read_parquet(TINYGSM).head(args.n_tinygsm * 2)
  tg = [{'question': q, 'code': c, 'gsm8k_index': -1}
        for q, c in zip(df['question'], df['code'])]
  t_ids, t_masks, _, _ = sft.tokenize(tg, tokenizer)
  sets['tinygsm'] = (t_ids[:args.n_tinygsm], t_masks[:args.n_tinygsm])
  for name, (i, m) in sets.items():
    print(name, 'n', len(i), 'answer tokens/ex',
          float(((i != tokenizer.pad_token_id) & (m == 1)).sum(1).float().mean()))

  device = torch.device('cuda')
  res = {}
  for ck in args.ckpts:
    path = sft.BASE_CKPT if ck in ('base', 'raw') else ck
    config = sft.build_config(path)
    if ck == 'raw':  # raw training weights of the released checkpoint, no EMA
      import algo
      model = algo.MDLM.load_from_checkpoint(
        path, tokenizer=tokenizer, config=config, strict=True)
      model.ema = None
      model = model.to(device).eval()
    else:
      model = sft.load_model(config, tokenizer, path).to(device).eval()
    res[ck] = {}
    for name, (i, m) in sets.items():
      res[ck][name] = breakdown(model, i, m, tokenizer, args.ts, 64, device)
      res[ck][name]['nelbo'] = round(sft.val_loss(model, i, m, 64, device), 4)
      model.backbone.eval()
    print(ck, json.dumps(res[ck], indent=1))
    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
  main()
