"""Rollouts for RL on MDLM (D11, D12).

A rollout runs FastAncestralSampler (random reveal order, ancestral sampling
also on the last step) and keeps, for every position, the round that revealed
it. Because MDLM never changes a revealed token, (final tokens, reveal round)
is enough to rebuild the canvas of every round (rl/logprob.py), so the
trajectory costs L integers to store.

Prompts, decoding and scoring are the same as the GSM8K eval
(dataloader.get_gsm8k_test_dataset, main._sample_gsm8k, sandbox_gsm8k), so a
rollout's reward is exactly what the eval would say about the same text.
Separator: pass config.data.separator (the literal backslash-n, see
scripts/sft.py); never write '\\n' by hand.
"""
import dataclasses
import multiprocessing as mp
import os

import torch

import fast_sampler
import samplers
import sandbox_gsm8k
from rl import reward as rl_reward


@dataclasses.dataclass
class Rollout:
  tokens: torch.Tensor         # (N, L) final canvas, prompt included
  prompt_len: torch.Tensor     # (N,)
  reveal_step: torch.Tensor    # (N, L) round that revealed each position; -1 = prompt
  sampler_logp: torch.Tensor   # (N, L) float64, sampler's own value (diagnostics only)
  group: torch.Tensor          # (N,) index of the question within this batch
  responses: list
  rewards: torch.Tensor        # (N,) float32, 1 = executes to the gold answer
  num_steps: int
  temperature: float
  categories: list = None      # rl.reward.grade category per rollout (format_reward > 0)


def encode_prompts(tokenizer, questions, separator):
  """[BOS] question.strip() SEP, as in dataloader.get_gsm8k_test_dataset."""
  sep = tokenizer(separator, add_special_tokens=False).input_ids
  return [torch.tensor([tokenizer.bos_token_id]
                       + tokenizer(q.strip(), add_special_tokens=False).input_ids
                       + sep) for q in questions]


def pad_prefix(id_lists, device):
  """Same as main._pad_prefix_batch."""
  lengths = torch.tensor([len(x) for x in id_lists], device=device)
  padded = torch.zeros(len(id_lists), int(lengths.max()), dtype=torch.long,
                       device=device)
  for i, x in enumerate(id_lists):
    padded[i, :len(x)] = x.to(device)
  return padded, lengths


def decode(tokenizer, tokens, prompt_len):
  """Same as main._sample_gsm8k: everything after the prompt, specials skipped."""
  return [tokenizer.decode(tokens[i, int(prompt_len[i]):].tolist(),
                           skip_special_tokens=True)
          for i in range(tokens.shape[0])]


def _score_one(args):
  response, gold, timeout_s = args
  return int(sandbox_gsm8k.evaluate_samples(response, gold, timeout_s=timeout_s))


def _grade_one(args):
  response, gold, timeout_s = args
  return rl_reward.grade(response, gold, timeout_s=timeout_s)


def score(responses, golds, timeout_s=5.0, procs=8):
  """Upstream scorer. It uses SIGALRM, so parallel scoring needs processes."""
  return _map(_score_one, [(r, g, timeout_s) for r, g in zip(responses, golds)], procs)


def grade(responses, golds, timeout_s=5.0, procs=8):
  """rl.reward.grade categories (same verdict as score() on 'correct')."""
  return _map(_grade_one, [(r, g, timeout_s) for r, g in zip(responses, golds)], procs)


def _map(fn, jobs, procs):
  os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')  # forked workers
  if procs <= 1:
    return [fn(j) for j in jobs]
  with mp.get_context('fork').Pool(procs) as pool:
    return pool.map(fn, jobs, chunksize=8)


@torch.no_grad()
def generate(model, tokenizer, questions, golds, *, group_size, num_steps,
             temperature, separator, batch_size=256, skip_unchanged=True,
             score_procs=8, format_reward=0.0):
  """G rollouts per question; rows of the same question are contiguous.

  The backbone runs in eval mode (no dropout): the policy is the eval-mode
  network, and rl/logprob.py recomputes log-probs in eval mode too.
  """
  assert model.ema is None, 'load with scripts.sft.load_model (EMA folded in)'
  model.backbone.eval()
  sampler = fast_sampler.FastAncestralSampler(
    use_float64=True, noise_removal='ancestral', temperature=temperature,
    logits_mode='selected', noise_mode='fast', skip_unchanged=skip_unchanged)
  prompts = encode_prompts(tokenizer, questions, separator)
  rows = [(qi, p) for qi, p in enumerate(prompts) for _ in range(group_size)]
  out = {'tokens': [], 'prompt_len': [], 'reveal_step': [], 'logp': [], 'group': []}
  for s in range(0, len(rows), batch_size):
    chunk = rows[s:s + batch_size]
    prefix, lengths = pad_prefix([p for _, p in chunk], model.device)
    xt, meta = samplers.run_sampler(
      sampler, model, len(chunk), num_steps=num_steps,
      prefix_tokens=prefix, prefix_lengths=lengths)
    out['tokens'].append(xt)
    out['prompt_len'].append(lengths)
    out['reveal_step'].append(meta['reveal_step'])
    out['logp'].append(meta['token_logp'])
    out['group'].append(torch.tensor([qi for qi, _ in chunk], device=xt.device))
  tokens = torch.cat(out['tokens'])
  prompt_len = torch.cat(out['prompt_len'])
  group = torch.cat(out['group'])
  assert (tokens != model.mask_index).all(), 'unrevealed positions left'
  responses = decode(tokenizer, tokens, prompt_len)
  gold_rows = [golds[int(g)] for g in group]
  categories = None
  if format_reward:  # ablation switch (R36); main line is the binary reward
    categories = grade(responses, gold_rows, procs=score_procs)
    rewards = [rl_reward.reward(c, format_reward) for c in categories]
  else:
    rewards = score(responses, gold_rows, procs=score_procs)
  return Rollout(
    tokens=tokens, prompt_len=prompt_len,
    reveal_step=torch.cat(out['reveal_step']),
    sampler_logp=torch.cat(out['logp']), group=group, responses=responses,
    rewards=torch.tensor(rewards, dtype=torch.float32, device=tokens.device),
    num_steps=num_steps, temperature=temperature, categories=categories)
