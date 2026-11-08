"""R50: ELBO loss normalization and the weights/act shape guard.

The ELBO log-prob has one slot per trajectory, act (N, 1, L), while the
trajectory log-prob has m slots, act (N, m, L). Before R50 the script passed
the per-round stratum weights (N, m) to both, so `weights[:, :, None] * act`
broadcast to (N, m, L) and multiplied every ELBO action by sum_j w_j (the
number of non-empty reveal rounds, ~120). This test pins the intended
behaviour down; it needs only CPU torch.

  cd code && python -m tests.test_grpo_elbo_loss
"""
import torch

from rl import grpo


def manual_loss(logp_new, logp_old, act, adv, n_actions):
  """Token-level GRPO loss with weights=1 and ratio=1 (logp_new == logp_old)."""
  return -((act * adv[:, None, None]).sum() / n_actions.sum().clamp(min=1))


def main():
  N, L = 2, 4
  act = torch.ones(N, 1, L, dtype=torch.bool)
  lp = torch.full((N, 1, L), -0.5)
  adv = torch.tensor([1.0, -1.0])
  n_actions = act.sum(dim=(1, 2))

  ones = torch.ones(N, 1)
  loss, _ = grpo.grpo_loss(lp, lp.clone(), act, ones, adv, n_actions)
  expected = manual_loss(lp, lp, act, adv, n_actions)
  assert torch.allclose(loss, expected), (loss.item(), expected.item())
  print(f'[elbo-loss] ones weights: {loss.item():.6f} == {expected.item():.6f}')

  # The old bug: (N, m) weights with (N, 1, L) act must now raise, not broadcast.
  bad = torch.full((N, 8), 16.0)
  try:
    grpo.grpo_loss(lp, lp.clone(), act, bad, adv, n_actions)
  except AssertionError as e:
    print(f'[shape-guard] raised as intended: {str(e).splitlines()[0]}')
  else:
    raise SystemExit('[shape-guard] FAILED: (N,8) weights with (N,1,L) act did not raise')

  # m+1 weights are fine when act has m slots (trajectory path).
  m = 8
  act_m = torch.ones(N, m, L, dtype=torch.bool)
  lp_m = torch.full((N, m, L), -0.5)
  w_m = torch.ones(N, m)
  grpo.grpo_loss(lp_m, lp_m.clone(), act_m, w_m, adv, act_m.sum(dim=(1, 2)))
  print('[trajectory] (N,m,L) act with (N,m) weights: ok')

  print('ALL R50 TESTS PASSED')


if __name__ == '__main__':
  main()
