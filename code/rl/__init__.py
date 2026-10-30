"""RL on MDLM (D11 GRPO + DAPO tricks, D12 trajectory log-probs).

  rollout.py  sample G answers per question, keep the trajectory, score them
              with the GSM8K eval's own decode + scorer
  logprob.py  exact per-round log-probs of a trajectory, stratified round
              sampling (unbiased estimate of log pi(tau))
  grpo.py     group advantages, dynamic sampling, clipped token-level loss
"""
