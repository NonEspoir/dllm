"""GRPO with DAPO tricks on MDLM trajectories (D11).

Advantage: A = r - mean_group(r), optionally / std_group(r) (standard GRPO;
Dr. GRPO drops the std, kept as a switch for the ablation).
Dynamic sampling: groups whose rewards are all equal have A = 0 everywhere and
give no gradient; they are dropped before the expensive log-prob forwards.
Loss (token level, clip-higher):
  per action i of trajectory n in sampled round j:
    ratio = exp(logp_new - logp_old)
    s = min(ratio * A_n, clip(ratio, 1 - eps_low, 1 + eps_high) * A_n)
  loss = - sum_{n,j,i} w_{n,j} * s / (number of actions of the kept trajectories)
w_{n,j} is the stratum weight from rl.logprob.select_rounds, so the sum over
sampled rounds is an unbiased estimate of the sum over all rounds, and the
denominator is the exact action count (known from reveal_step), as in DAPO's
token-level normalization.
With one update per batch (on-policy), ratio == 1 and the clip never fires;
it only matters when a batch is reused.
"""
import torch


def group_advantages(rewards, group, scale_std=True, eps=1e-6):
  adv = torch.zeros_like(rewards)
  for g in torch.unique(group):
    idx = group == g
    r = rewards[idx]
    a = r - r.mean()
    if scale_std:
      a = a / (r.std(unbiased=False) + eps)
    adv[idx] = a
  return adv


def mixed_groups(rewards, group):
  """Rows whose group has both correct and wrong answers (DAPO dynamic sampling)."""
  keep = torch.zeros_like(rewards, dtype=torch.bool)
  for g in torch.unique(group):
    idx = group == g
    r = rewards[idx]
    keep[idx] = bool(r.max() > r.min())
  return keep


def action_counts(reveal_step, tokens=None, pad_id=None, count_pad=True):
  """Exact number of actions per trajectory (all revealed positions)."""
  act = reveal_step >= 0
  if not count_pad:
    act = act & (tokens != pad_id)
  return act.sum(dim=1)


def grpo_loss(logp_new, logp_old, act, weights, adv, n_actions,
              eps_low=0.2, eps_high=0.28):
  """logp_*: (N, m, L); act: (N, m, L) bool; weights: (N, m); adv: (N,);
  n_actions: (N,) exact action counts. Returns loss and diagnostics."""
  ratio = torch.exp(logp_new - logp_old)
  a = adv[:, None, None]
  s = torch.minimum(ratio * a, torch.clamp(ratio, 1 - eps_low, 1 + eps_high) * a)
  w = weights[:, :, None] * act
  loss = -(w * s).sum() / n_actions.sum().clamp(min=1)
  with torch.no_grad():
    on = act
    clipped = ((ratio < 1 - eps_low) | (ratio > 1 + eps_high)) & on
    stats = {'ratio_mean': float(ratio[on].mean()) if on.any() else 1.0,
             'clip_frac': float(clipped.sum() / on.sum().clamp(min=1)),
             'logp_mean': float(logp_new[on].mean()) if on.any() else 0.0}
  return loss, stats
