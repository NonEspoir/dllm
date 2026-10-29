"""Layer 1 of the sampler validation (docs/04 §2.4): token-exact equivalence.

Runs upstream AncestralSampler and FastAncestralSampler(logits_mode=full,
noise_mode=reference) from the same RNG state and requires identical tokens.
Then checks the fast path (logits_mode=selected) against the full path on
log-probs, and skip_unchanged against selected on tokens.

Needs a GPU and the MDLM checkpoint:
  cd code && python -m tests.test_fast_sampler \
      --ckpt /root/autodl-tmp/checkpoints/tinygsm/mdlm.ckpt
"""
import argparse
import json
import os
import sys

import hydra
import torch
import torch._inductor.config

# Upstream's sample_absorbing_posterior is torch.compile'd; by default
# Inductor draws random numbers with its own generator, so eager and
# compiled code would consume different streams. Make them identical.
torch._inductor.config.fallback_random = True

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import main as upstream_main  # noqa: F401  (registers OmegaConf resolvers)
import algo
import dataloader
import samplers
import fast_sampler


def load(ckpt, overrides):
  with hydra.initialize(version_base=None, config_path='../configs'):
    config = hydra.compose(config_name='config', overrides=[
      'mode=gsm8k_eval', 'data=gsm8k-test',
      'data.tokenizer_name_or_path=HuggingFaceTB/SmolLM-135M',
      'model=small', 'model.length=512', 'algo=mdlm',
      'sampler=ancestral', f'eval.checkpoint_path={ckpt}',
      'trainer.devices=1', 'trainer.num_nodes=1'] + overrides)
  tokenizer = dataloader.get_tokenizer(config)
  model = algo.MDLM.load_from_checkpoint(
    ckpt, tokenizer=tokenizer, config=config, strict=True)
  model.to('cuda')
  model._eval_mode()
  return model, tokenizer, config


def prompts(tokenizer, path, n):
  with open(path) as f:
    records = json.load(f)[:n]
  ids = [[tokenizer.bos_token_id]
         + tokenizer(r['prompt'].strip(), add_special_tokens=False).input_ids
         + tokenizer('\n', add_special_tokens=False).input_ids
         for r in records]
  lengths = torch.tensor([len(x) for x in ids], device='cuda')
  padded = torch.zeros(len(ids), int(lengths.max()), dtype=torch.long,
                       device='cuda')
  for i, x in enumerate(ids):
    padded[i, :len(x)] = torch.tensor(x)
  return padded, lengths


def run(sampler, model, prefix, lengths, steps, seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  return samplers.run_sampler(sampler, model, prefix.shape[0],
                              num_steps=steps, prefix_tokens=prefix,
                              prefix_lengths=lengths)


@torch.no_grad()
def main():
  p = argparse.ArgumentParser()
  p.add_argument('--ckpt', required=True)
  p.add_argument('--data', default='/root/autodl-tmp/data/gsm8k_dev200.json')
  p.add_argument('--batch', type=int, default=2)
  p.add_argument('--steps', nargs='+', type=int, default=[8, 64])
  p.add_argument('--temps', nargs='+', type=float, default=[1.0, 0.5])
  p.add_argument('--seeds', nargs='+', type=int, default=[0, 1])
  args = p.parse_args()

  model, tokenizer, _ = load(args.ckpt, [])
  prefix, lengths = prompts(tokenizer, args.data, args.batch)
  failures = 0

  for T in args.temps:
    for steps in args.steps:
      for noise_removal in ['ancestral', 'greedy']:
        for seed in args.seeds:
          tag = f'T={T} steps={steps} removal={noise_removal} seed={seed}'
          ref = samplers.AncestralSampler(
            use_float64=True, noise_removal=noise_removal,
            temperature=T)
          new = fast_sampler.FastAncestralSampler(
            use_float64=True, noise_removal=noise_removal,
            temperature=T, logits_mode='full', noise_mode='reference')
          x_ref, _ = run(ref, model, prefix, lengths, steps, seed)
          x_new, meta = run(new, model, prefix, lengths, steps, seed)
          n_diff = int((x_ref != x_new).sum())
          ok = n_diff == 0
          # Exactly the generated positions (after each row's prompt)
          # are revealed.
          pos = torch.arange(x_new.shape[1], device=x_new.device)
          gen = pos[None, :] >= lengths[:, None]
          ok_reveal = bool(((meta['reveal_step'] >= 0) == gen).all())
          ok_mask = bool((x_new != model.mask_index).all())
          status = 'PASS' if ok and ok_reveal and ok_mask else 'FAIL'
          failures += status == 'FAIL'
          print(f'[exact] {tag}: diff_tokens={n_diff} '
                f'reveal_ok={ok_reveal} no_mask_left={ok_mask} {status}')

  # Fast path: selected-rows output layer vs full forward. Same RNG
  # (fast noise mode), so tokens agree unless bf16 rounding flips an argmax.
  for T in args.temps:
    steps = max(args.steps)
    full = fast_sampler.FastAncestralSampler(
      temperature=T, logits_mode='full', noise_mode='fast')
    sel = fast_sampler.FastAncestralSampler(
      temperature=T, logits_mode='selected', noise_mode='fast')
    skip = fast_sampler.FastAncestralSampler(
      temperature=T, logits_mode='selected', noise_mode='fast',
      skip_unchanged=True)
    x_full, m_full = run(full, model, prefix, lengths, steps, 0)
    x_sel, m_sel = run(sel, model, prefix, lengths, steps, 0)
    x_skip, m_skip = run(skip, model, prefix, lengths, steps, 0)
    same = (x_full == x_sel)
    lp_err = (m_full['token_logp'] - m_sel['token_logp'])[same].abs().max()
    print(f'[fast] T={T} steps={steps}: selected vs full '
          f'token_agree={same.float().mean().item():.4f} '
          f'max|dlogp| on agreeing={lp_err.item():.2e}')
    n_skip = int((x_skip != x_sel).sum())
    print(f'[skip] T={T} steps={steps}: skip_unchanged vs selected '
          f'diff_tokens={n_skip} row_forwards {m_skip["row_forwards"]} '
          f'vs {m_sel["row_forwards"]}')

  print('ALL EXACT TESTS PASSED' if failures == 0
        else f'{failures} EXACT TESTS FAILED')
  sys.exit(1 if failures else 0)


if __name__ == '__main__':
  main()
