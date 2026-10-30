"""Does one GRPO update move log pi(tau) the right way? (R37 diagnosis)

  cd code && python -m tests.test_grpo_direction --lr 1e-4 --updates 5

Rollouts for a few questions from base, exact log pi(tau) (all rounds) of every
kept trajectory before and after `--updates` GRPO steps on the same batch.
If the loss has the right sign, trajectories with positive advantage gain
log-prob and those with negative advantage lose it. Also reports how much of
the change sits on answer tokens vs PAD.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--lr', type=float, default=1e-4)
  p.add_argument('--updates', type=int, default=5)
  p.add_argument('--questions', type=int, default=16)
  p.add_argument('--rollout_steps', type=int, default=128)
  args = p.parse_args()
  from rl import grpo, logprob, rollout
  from scripts import sft
  import dataloader
  import omegaconf

  torch.manual_seed(0)
  tok = dataloader.get_tokenizer(omegaconf.OmegaConf.create(
    {'data': {'tokenizer_name_or_path': 'HuggingFaceTB/SmolLM-135M'}}))
  config = sft.build_config(sft.BASE_CKPT)
  model = sft.load_model(config, tok, sft.BASE_CKPT).to('cuda')
  model.backbone.eval()
  with open('/root/autodl-tmp/data/gsm8k_train.json') as f:
    recs = json.load(f)[:args.questions]
  ro = rollout.generate(model, tok, [r['prompt'] for r in recs],
                        [r['response_ground_truth'] for r in recs], group_size=8,
                        num_steps=args.rollout_steps, temperature=1.0,
                        separator=config.data.separator, batch_size=256)
  adv = grpo.group_advantages(ro.rewards, ro.group)
  idx = grpo.mixed_groups(ro.rewards, ro.group).nonzero()[:, 0]
  tokens, rs, a = ro.tokens[idx], ro.reveal_step[idx], adv[idx]
  print(f'{len(ro.rewards)} rollouts, reward {float(ro.rewards.mean()):.3f}, kept {len(idx)}')
  pad = tok.pad_token_id
  rounds_all, _ = logprob.all_rounds(rs)

  def exact():
    out_all, out_ans = [], []
    with torch.no_grad():
      for s in range(0, len(idx), 4):
        lp, act = logprob.logp_at_rounds(model, tokens[s:s + 4], rs[s:s + 4],
                                         rounds_all[s:s + 4], 1.0, chunk=64)
        per = (lp * act).sum(1)                       # (n, L)
        out_all.append(per.sum(1))
        out_ans.append((per * (tokens[s:s + 4] != pad)).sum(1))
    return torch.cat(out_all), torch.cat(out_ans)

  before, before_ans = exact()
  params = [q for q in model._get_parameters() if q.requires_grad]
  opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
  n_act = grpo.action_counts(rs).sum().reshape(1)
  for u in range(args.updates):
    rounds, w = logprob.select_rounds(rs, 8)
    opt.zero_grad(set_to_none=True)
    for s in range(0, len(idx), 8):
      c = slice(s, s + 8)
      lp, act = logprob.logp_at_rounds(model, tokens[c], rs[c], rounds[c], 1.0, chunk=64)
      loss, _ = grpo.grpo_loss(lp, lp.detach(), act, w[c], a[c], n_act)
      loss.backward()
    opt.step()
  after, after_ans = exact()
  d, d_ans = after - before, after_ans - before_ans
  pos, neg = a > 0, a < 0
  print(f'lr {args.lr}, {args.updates} updates on the same batch:')
  print(f'  positive-advantage trajectories (n={int(pos.sum())}): mean d log pi {float(d[pos].mean()):+.3f} '
        f'(answer tokens {float(d_ans[pos].mean()):+.3f}); share increased {float((d[pos] > 0).float().mean()):.2f}')
  print(f'  negative-advantage trajectories (n={int(neg.sum())}): mean d log pi {float(d[neg].mean()):+.3f} '
        f'(answer tokens {float(d_ans[neg].mean()):+.3f}); share decreased {float((d[neg] < 0).float().mean()):.2f}')
  print(f'  correlation(advantage, d log pi) {float(torch.corrcoef(torch.stack([a, d]))[0, 1]):+.3f}; '
        f'mean |log pi| before {float(before.abs().mean()):.1f}')


if __name__ == '__main__':
  main()
