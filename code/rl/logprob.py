"""Trajectory log-probs for MDLM (D12, docs/03 §2–§5).

  log pi(tau) = sum_k sum_{i revealed in round k} log p_theta(x_i | canvas_k)

The "which positions are revealed" part does not depend on theta (coins drawn
before the model is called), so only the token terms carry gradient.

Canvas of round k: positions revealed before round k hold their final token,
the rest are [MASK] (MDLM never rewrites a revealed token). Prompt positions
have reveal_step -1 and are always visible.

Cost: one forward per round. select_rounds() picks m rounds per trajectory,
stratified over that trajectory's non-empty rounds (rounds that revealed at
least one position), with weight = stratum size, so
  E[ sum_j w_j * term(round_j) ] = sum over all rounds  (unbiased).
Empty rounds contribute nothing and are never drawn.

Pitfalls handled here (docs/03 §5):
  - temperature: log-softmax of logits / T, the distribution actually sampled;
  - precision: old and new log-probs both come from logp_at_rounds (same code
    path); the sampler's own float64 values are diagnostics only. The output
    layer returns bf16 logits and the sampler (like upstream) takes the
    log-softmax in bf16, so its log-probs carry bf16 rounding (up to ~1 ulp of
    the logsumexp, ~0.06 at |lse| 8-16). Training upcasts to fp32 before the
    log-softmax (fp32_softmax=True), the usual choice in RL so that ratios are
    not dominated by bf16 noise; the remaining sampler/trainer gap is a small
    training-inference mismatch (measured in tests/test_rl_logprob.py).
    fp32_softmax=False mirrors the sampler bit for bit and is used to test the
    logic (same tokens, rounds, temperature, mask handling);
  - last round: it reveals every remaining position; those are actions too
    (they appear in reveal_step like any other round);
  - D20: PAD positions are revealed by the policy like any other token (and
    where PAD goes decides the answer length), so they are actions and are
    included. count_pad=False is available for an ablation.
"""
import torch

import fast_sampler


def canvas_at(tokens, reveal_step, rounds, mask_index):
  """tokens, reveal_step: (R, L); rounds: (R,). Canvas before round `rounds`."""
  hidden = reveal_step >= rounds[:, None]
  return torch.where(hidden, torch.full_like(tokens, mask_index), tokens)


def select_rounds(reveal_step, m, generator=None):
  """Stratified rounds per trajectory.

  Returns rounds (N, m) long (-1 = unused slot) and weights (N, m) float.
  If a trajectory has <= m non-empty rounds, all of them are taken (weight 1).
  """
  N = reveal_step.shape[0]
  rounds = torch.full((N, m), -1, dtype=torch.long)
  weights = torch.zeros((N, m), dtype=torch.float32)
  rs = reveal_step.cpu()
  for n in range(N):
    steps = torch.unique(rs[n][rs[n] >= 0])  # sorted non-empty rounds
    if len(steps) <= m:
      rounds[n, :len(steps)] = steps
      weights[n, :len(steps)] = 1.0
      continue
    for j, stratum in enumerate(torch.tensor_split(steps, m)):
      pick = torch.randint(len(stratum), (1,), generator=generator)
      rounds[n, j] = stratum[pick]
      weights[n, j] = float(len(stratum))
  return rounds.to(reveal_step.device), weights.to(reveal_step.device)


def all_rounds(reveal_step):
  """Every non-empty round with weight 1 (exact sum; tests and diagnostics)."""
  m = max(len(torch.unique(r[r >= 0])) for r in reveal_step.cpu())
  return select_rounds(reveal_step, m)


def logp_at_rounds(model, tokens, reveal_step, rounds, temperature,
                   count_pad=True, chunk=64, fp32_softmax=True):
  """Per-token log-probs of the positions revealed in the given rounds.

  tokens, reveal_step: (N, L); rounds: (N, m) from select_rounds.
  Returns logp (N, m, L) float32 (0 where not an action of that round) and
  act (N, m, L) bool. Gradient flows to the model unless called under
  torch.no_grad(). Uses the sampler's own forward (backbone up to the final
  hidden states, output layer only on the needed rows), so values match the
  rollout's distribution up to kernel-level rounding.
  """
  assert not model.time_conditioning, 'canvas-only forward needs no time input'
  N, m = rounds.shape
  L = tokens.shape[1]
  pad_id = model.tokenizer.pad_token_id
  act = (reveal_step[:, None, :] == rounds[:, :, None]) & (rounds[:, :, None] >= 0)
  if not count_pad:
    act = act & (tokens[:, None, :] != pad_id)
  logp = torch.zeros((N, m, L), dtype=torch.float32, device=tokens.device)
  pairs = act.flatten(0, 1).any(dim=1).nonzero()[:, 0]  # (trajectory, slot) with actions
  backbone = model.backbone
  for s in range(0, len(pairs), chunk):
    pid = pairs[s:s + chunk]
    n_idx, j_idx = pid // m, pid % m
    xt = canvas_at(tokens[n_idx], reveal_step[n_idx], rounds[n_idx, j_idx],
                   model.mask_index)
    sigma = model._process_sigma(torch.zeros(len(pid), 1, device=tokens.device))
    with torch.amp.autocast('cuda', dtype=torch.float32):
      hidden, _ = fast_sampler._backbone_hidden(backbone, xt, sigma)
    sel = act[n_idx, j_idx]                      # (c, L)
    with torch.amp.autocast('cuda', dtype=torch.float32):
      logits = fast_sampler._output_logits(backbone, hidden[sel])
    if fp32_softmax:
      logits = logits.float() / temperature
    elif temperature != 1.0:  # exactly the sampler's arithmetic (bf16)
      logits = logits / temperature
    rows = torch.full((logits.shape[0], 1), model.mask_index,
                      dtype=torch.long, device=logits.device)
    lp = model._process_model_output(
      model_output=logits[:, None, :], xt=rows, sigma=None)[:, 0]
    tok = tokens[n_idx][sel]
    vals = lp.gather(-1, tok[:, None])[:, 0]
    b, l = sel.nonzero(as_tuple=True)
    logp = logp.index_put((n_idx[b], j_idx[b], l), vals.float())
  return logp, act
