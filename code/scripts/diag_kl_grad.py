"""Diagnostic (R43): size of the KL term's gradient vs the policy-gradient term,
and how much of the policy gradient comes from PAD actions (D20).

One rollout batch from --ckpt (32 questions x G=8, 128 steps, T=1, format reward
0.1, no std: the D5 / KL-round settings), 8 stratified rounds per kept
trajectory, then separately backpropagates
  pg_all    the GRPO loss at update 1 (ratio 1), all actions;
  pg_nonpad the same, PAD actions masked out (numerator only, same denominator);
  pg_pad    the same, PAD actions only;
  kl        beta * k3 KL to --ref (default: base), as in scripts/grpo.py;
and prints each gradient's L2 norm, cosines between them, and the distribution
of d = log p_ref - log p over the actions. Read-only: no optimizer step.

  python -m scripts.diag_kl_grad --ckpt /root/autodl-tmp/rl_ckpt/<run>/step225.ckpt
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts import grpo as grpo_script  # noqa: E402
from scripts import sft  # noqa: E402


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--ckpt', required=True)
  p.add_argument('--ref', default=sft.BASE_CKPT)
  p.add_argument('--beta', type=float, default=0.04)
  p.add_argument('--batch_index', type=int, default=28,
                 help='which 32-question slice of the RL pool to roll out')
  p.add_argument('--seed', type=int, default=1)
  p.add_argument('--out', default=None)
  args = p.parse_args()

  from rl import grpo, logprob, rollout
  import dataloader
  import omegaconf
  torch.manual_seed(args.seed)
  tok = dataloader.get_tokenizer(omegaconf.OmegaConf.create(
    {'data': {'tokenizer_name_or_path': 'HuggingFaceTB/SmolLM-135M'}}))
  config = sft.build_config(args.ckpt)
  sep = config.data.separator
  model = sft.load_model(config, tok, args.ckpt).to('cuda')
  model.backbone.eval()
  ref = sft.load_model(sft.build_config(args.ref), tok, args.ref).to('cuda')
  ref.backbone.eval()
  for q in ref.parameters():
    q.requires_grad_(False)
  params = [q for q in model._get_parameters() if q.requires_grad]

  pool, _ = grpo_script.question_pool(args.seed)
  batch = pool[args.batch_index * 32:(args.batch_index + 1) * 32]
  ro = rollout.generate(model, tok, [r['prompt'] for r in batch],
                        [r['response_ground_truth'] for r in batch],
                        group_size=8, num_steps=128, temperature=1.0,
                        separator=sep, batch_size=256, format_reward=0.1)
  adv = grpo.group_advantages(ro.rewards, ro.group, scale_std=False)
  idx = grpo.mixed_groups(ro.rewards, ro.group).nonzero()[:, 0]
  tokens, rs, a = ro.tokens[idx], ro.reveal_step[idx], adv[idx]
  pad = tok.pad_token_id
  n_act = grpo.action_counts(rs, tokens, pad).sum().reshape(1)
  rounds, w = logprob.select_rounds(rs, 8, generator=torch.Generator().manual_seed(args.seed))
  chunks = [slice(s, s + 8) for s in range(0, len(idx), 8)]

  def flat_grad():
    return torch.cat([(q.grad if q.grad is not None else torch.zeros_like(q)).flatten()
                      for q in params])

  grads, dstats = {}, {'d': [], 'is_pad': [], 'lp': []}
  for comp in ('pg_all', 'pg_nonpad', 'pg_pad', 'kl'):
    model.zero_grad(set_to_none=True)
    for c in chunks:
      lp, act = logprob.logp_at_rounds(model, tokens[c], rs[c], rounds[c], 1.0, chunk=64)
      is_pad = (tokens[c] == pad)[:, None, :].expand_as(act)
      if comp == 'kl':
        with torch.no_grad():
          lp_ref, _ = logprob.logp_at_rounds(ref, tokens[c], rs[c], rounds[c], 1.0, chunk=64)
        d = lp_ref - lp
        kl = (w[c][:, :, None] * act * (torch.exp(d) - 1 - d)).sum() / n_act.sum()
        loss = args.beta * kl
        dstats['d'].append(d.detach()[act].cpu())
        dstats['is_pad'].append(is_pad[act].cpu())
        dstats['lp'].append(lp.detach()[act].cpu())
      else:
        m = act if comp == 'pg_all' else act & (~is_pad if comp == 'pg_nonpad' else is_pad)
        loss, _ = grpo.grpo_loss(lp, lp.detach(), m, w[c], a[c], n_act)
      loss.backward()
    grads[comp] = flat_grad().clone()
  model.zero_grad(set_to_none=True)

  def cos(x, y):
    return float(torch.dot(x, y) / (x.norm() * y.norm() + 1e-30))

  d = torch.cat(dstats['d'])
  isp = torch.cat(dstats['is_pad'])
  lp = torch.cat(dstats['lp'])
  q = torch.tensor([0.5, 0.9, 0.99, 0.999])
  res = {
    'ckpt': args.ckpt, 'ref': args.ref, 'beta': args.beta,
    'kept_rows': int(len(idx)), 'correct': float((ro.rewards == 1).float().mean()),
    'actions_sampled': int(len(d)), 'pad_frac_actions': float(isp.float().mean()),
    'lp_mean_pad': float(lp[isp].mean()), 'lp_mean_nonpad': float(lp[~isp].mean()),
    'grad_norm': {k: float(v.norm()) for k, v in grads.items()},
    'cos': {'pg_all~kl': cos(grads['pg_all'], grads['kl']),
            'pg_nonpad~pg_pad': cos(grads['pg_nonpad'], grads['pg_pad']),
            'pg_all~pg_nonpad': cos(grads['pg_all'], grads['pg_nonpad'])},
    'd_quantiles_abs': dict(zip(map(str, q.tolist()), torch.quantile(d.abs().float()[:2_000_000], q).tolist())),
    'd_max': float(d.max()), 'd_min': float(d.min()),
    'kl_factor_max_abs': float((1 - torch.exp(d)).abs().max()),
    'mean_k3': float((torch.exp(d) - 1 - d).mean()),
  }
  print(json.dumps(res, indent=1))
  if args.out:
    with open(args.out, 'w') as f:
      json.dump(res, f, indent=1)


if __name__ == '__main__':
  main()
