"""Checks for rl/ (D12 trajectory log-probs), before any RL training.

  cd code && python -m tests.test_rl_logprob          # CPU part + GPU part
  cd code && python -m tests.test_rl_logprob --cpu_only

CPU: stratified round sampling is unbiased.
GPU (small: a few dev200 questions, 32 steps):
  1. canvas_at() rebuilds the sampler's canvas before every round, exactly;
  2. logic: recomputed per-token log-probs (all rounds) with the sampler's
     own precision (fp32_softmax=False) equal the sampler's values; the
     training precision (fp32 log-softmax) is reported as the
     training-inference gap (bf16 rounding of the sampler, not a bug);
  3. recomputing at the wrong temperature is clearly off (the check has teeth);
  (first version compared fp32 recompute with the bf16 sampler under a 5e-2
  bound and failed at 0.061 = one bf16 ulp at |lse| 8-16; the bound had been
  set without accounting for the sampler's bf16 log-softmax.)
  4. old == new before an update (ratio 1), and the GRPO loss backpropagates;
  5. rl.rollout.generate() runs end to end and rewards are 0/1.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from rl import grpo, logprob  # noqa: E402


def check_unbiased():
  g = torch.Generator().manual_seed(0)
  N, K, m = 3, 40, 8
  rs = torch.randint(-1, K, (N, 100), generator=g)   # some rounds stay empty
  vals = torch.randn(N, K, generator=g, dtype=torch.float64)
  exact = torch.zeros(N, dtype=torch.float64)
  for n in range(N):
    for k in torch.unique(rs[n][rs[n] >= 0]):
      exact[n] += vals[n, k]
  draws = 20000
  est = torch.zeros(N, dtype=torch.float64)
  for _ in range(draws):
    rounds, w = logprob.select_rounds(rs, m, generator=g)
    v = torch.where(rounds >= 0, vals.gather(1, rounds.clamp(min=0)), 0.0)
    est += (w.double() * v).sum(1)
  est /= draws
  n_nonempty = torch.tensor([len(torch.unique(r[r >= 0])) for r in rs])
  ok_w = bool((w.sum(1) == n_nonempty).all())
  err = float((est - exact).abs().max())
  print(f'[cpu] stratified estimator: exact {exact.tolist()} vs mean of {draws} '
        f'draws {[round(x, 3) for x in est.tolist()]}, max err {err:.3f}; '
        f'weights sum to #non-empty rounds: {ok_w}')
  return err < 0.05 * float(exact.abs().max()) + 0.05 and ok_w


def gpu_checks(n_questions, steps):
  import samplers
  import fast_sampler
  from rl import rollout
  from scripts import sft
  import dataloader
  import omegaconf

  tok = dataloader.get_tokenizer(omegaconf.OmegaConf.create(
    {'data': {'tokenizer_name_or_path': 'HuggingFaceTB/SmolLM-135M'}}))
  config = sft.build_config(sft.BASE_CKPT)
  model = sft.load_model(config, tok, sft.BASE_CKPT).to('cuda')
  model.backbone.eval()
  with open('/root/autodl-tmp/data/gsm8k_dev200.json') as f:
    recs = json.load(f)[:n_questions]
  questions = [r['prompt'] for r in recs]
  golds = [r['response_ground_truth'] for r in recs]
  sep = config.data.separator
  assert sep == '\\n', repr(sep)
  ok = True

  for T in (1.0, 0.7):
    torch.manual_seed(0)
    prompts = [p for p in rollout.encode_prompts(tok, questions, sep) for _ in range(2)]
    prefix, lengths = rollout.pad_prefix(prompts, 'cuda')
    sampler = fast_sampler.FastAncestralSampler(
      use_float64=True, noise_removal='ancestral', temperature=T,
      logits_mode='selected', noise_mode='fast', skip_unchanged=False)
    with torch.no_grad():
      state = sampler.init_state(model, len(prompts), num_steps=steps,
                                 prefix_tokens=prefix, prefix_lengths=lengths)
      before = {}
      while not state.done:
        before[state.step_idx] = state.xt.clone()
        state = sampler.step(model, state)
    tokens, rs, slp = state.xt, state.reveal_step, state.token_logp

    # 1. canvas reconstruction
    n_bad = 0
    for k, xt in before.items():
      rk = torch.full((tokens.shape[0],), k, device='cuda')
      n_bad += int((logprob.canvas_at(tokens, rs, rk, model.mask_index) != xt).sum())
    print(f'[gpu] T={T}: canvas rebuilt for {len(before)} rounds, mismatching tokens {n_bad}')
    ok &= n_bad == 0

    # 2. full recompute vs sampler values
    rounds, w = logprob.all_rounds(rs)
    gen = rs >= 0

    def gap(temp, fp32):
      with torch.no_grad():
        lp, act = logprob.logp_at_rounds(model, tokens, rs, rounds, temp,
                                         fp32_softmax=fp32)
      d = (lp * act).sum(1).double() - slp
      return d[gen].abs(), float((d * gen).sum(1).abs().max()), act

    d_bf16, t_bf16, act = gap(T, False)
    covered = bool((act.sum(1) == gen).all())
    print(f'[gpu] T={T}: every generated position is an action of exactly one round: {covered}')
    print(f'[gpu] T={T}: sampler precision (bf16 log-softmax): per-token max |diff| '
          f'{float(d_bf16.max()):.2e}, per-trajectory max {t_bf16:.2e}')
    ok &= covered and float(d_bf16.max()) < 1e-6
    d32, t32, _ = gap(T, True)
    print(f'[gpu] T={T}: training precision (fp32 log-softmax) vs sampler: per-token '
          f'max {float(d32.max()):.2e} mean {float(d32.mean()):.2e}, share > 1e-2 '
          f'{float((d32 > 1e-2).float().mean()):.3f}; per-trajectory max {t32:.3f} '
          f'(trajectory log-prob ~ {float((slp * gen).sum(1).mean()):.1f}) [reported, not a pass/fail]')

    # 3. wrong temperature (sampler precision, so the right T gives ~0)
    if T != 1.0:
      d_wrong, _, _ = gap(1.0, False)
      print(f'[gpu] T={T}: recomputed at T=1 instead: per-token mean |diff| '
            f'{float(d_wrong.mean()):.2e} (right T: {float(d_bf16.mean()):.2e})')
      ok &= float(d_wrong.mean()) > 1e-3

    # 4. ratio == 1 before an update; loss backpropagates
    rounds, w = logprob.select_rounds(rs, 8)
    with torch.no_grad():
      old, act = logprob.logp_at_rounds(model, tokens, rs, rounds, T)
    new, act2 = logprob.logp_at_rounds(model, tokens, rs, rounds, T)
    r_err = float((torch.exp(new.detach() - old) - 1)[act].abs().max())
    rewards = torch.tensor([1., 0.] * n_questions, device='cuda')
    group = torch.arange(n_questions, device='cuda').repeat_interleave(2)
    adv = grpo.group_advantages(rewards, group)
    loss, stats = grpo.grpo_loss(new, old, act, w, adv, grpo.action_counts(rs))
    model.backbone.zero_grad(set_to_none=True)
    loss.backward()
    gn = float(torch.sqrt(sum((p.grad.float() ** 2).sum()
                              for p in model.backbone.parameters() if p.grad is not None)))
    print(f'[gpu] T={T}: before update max |ratio-1| {r_err:.1e}; loss {float(loss):.4f}; '
          f'grad norm {gn:.3e}; {stats}')
    ok &= r_err < 1e-5 and torch.isfinite(loss) and gn > 0 and bool((act == act2).all())
    model.backbone.zero_grad(set_to_none=True)

  # 5. end to end
  ro = rollout.generate(model, tok, questions, golds, group_size=2, num_steps=steps,
                        temperature=1.0, separator=sep, batch_size=8, score_procs=2)
  print(f'[gpu] generate(): {ro.tokens.shape[0]} rollouts, rewards {ro.rewards.tolist()}, '
        f'mixed groups {int(grpo.mixed_groups(ro.rewards, ro.group).sum())} rows')
  print('   first response:', ro.responses[0][:200].replace('\n', ' | '))
  ok &= set(ro.rewards.tolist()) <= {0.0, 1.0}
  return ok


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--cpu_only', action='store_true')
  p.add_argument('--n_questions', type=int, default=4)
  p.add_argument('--steps', type=int, default=32)
  args = p.parse_args()
  ok = check_unbiased()
  if not args.cpu_only:
    ok &= gpu_checks(args.n_questions, args.steps)
  print('ALL PASSED' if ok else 'FAILED')
  sys.exit(0 if ok else 1)


if __name__ == '__main__':
  main()
