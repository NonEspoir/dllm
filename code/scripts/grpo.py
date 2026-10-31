"""GRPO on MDLM (D11 GRPO + DAPO tricks, D12 trajectory log-probs, D31 settings).

One step:
  1. 32 train questions x G=8 rollouts (rl.rollout.generate: 128 steps, T=1,
     random reveal order), scored by the GSM8K eval's own scorer;
  2. drop groups that are all right or all wrong (zero advantage);
  3. advantages: group mean removed, divided by group std (switchable);
  4. per kept trajectory, 8 rounds stratified over its non-empty rounds;
     log-probs recomputed (fp32 log-softmax) with gradient, chunk by chunk,
     backward per chunk (the loss is a sum over actions divided by the exact
     total action count, so chunked backward gives the same gradient);
  5. one on-policy update (old = new.detach(): ratio is 1, clip inactive), or
     with --updates K > 1 (D32): old log-probs computed once for the batch,
     then K updates on the same rollouts and rounds; the ratio moves away from
     1 and the clip [1 - eps_low, 1 + eps_high] bounds each batch's change.
Policy = eval-mode network (no dropout), the one that is evaluated.

Every --eval_every steps: save a checkpoint (scripts/sft.py layout) and run
scripts/eval_gsm8k.sh on dev200 (512 steps + skip, T in {1, 0.1}, seed 1,
batch 200): the same protocol as the base runs, so results pair by question.

  python -m scripts.grpo --run_name 1030-grpo-base-lr3e-6 --lr 3e-6
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts import sft  # noqa: E402

REPO = sft.REPO
CODE = os.path.join(REPO, 'code')
CKPT_ROOT = '/root/autodl-tmp/rl_ckpt'


def question_pool(seed):
  """GSM8K train minus dev (already) minus rl_exclude (data/sft/README.md)."""
  with open('/root/autodl-tmp/data/gsm8k_train.json') as f:
    train = json.load(f)
  with open(sft.DATA) as f:
    excl = {r['gsm8k_index'] for r in map(json.loads, f) if r['rl_exclude']}
  with open(sft.DEV_IDS) as f:
    dev = set(json.load(f)['dev_ids'])
  pool = [r for r in train if r['gsm8k_index'] not in excl]
  assert not dev & {r['gsm8k_index'] for r in pool}
  g = torch.Generator().manual_seed(seed)
  return [pool[i] for i in torch.randperm(len(pool), generator=g).tolist()], len(excl)


def run_eval(ckpt, run_name, temp, steps, split):
  env = dict(os.environ, SPLIT=split, SAMPLER='fast_ancestral', STEPS=str(steps),
             TEMP=str(temp), BATCH='200', CKPT_PATH=ckpt, RUN_NAME=run_name)
  p = subprocess.run(['bash', 'scripts/eval_gsm8k.sh', 'sampler.skip_unchanged=true'],
                     cwd=CODE, env=env, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)
  path = os.path.join(REPO, 'runs', run_name, 'metrics.json')
  if p.returncode != 0 or not os.path.exists(path):
    return None
  with open(path) as f:
    return json.load(f)['accuracy']


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--run_name', required=True)
  p.add_argument('--init_ckpt', default=sft.BASE_CKPT)
  p.add_argument('--lr', type=float, required=True)
  p.add_argument('--steps', type=int, default=50)
  p.add_argument('--warmup', type=int, default=5)
  p.add_argument('--questions', type=int, default=32)
  p.add_argument('--group', type=int, default=8)
  p.add_argument('--rollout_steps', type=int, default=128)
  p.add_argument('--temperature', type=float, default=1.0)
  p.add_argument('--rounds', type=int, default=8)
  p.add_argument('--eps_low', type=float, default=0.2)
  p.add_argument('--eps_high', type=float, default=0.28)
  p.add_argument('--no_std', action='store_true', help='Dr. GRPO advantages')
  p.add_argument('--no_pad', action='store_true', help='D20 ablation')
  p.add_argument('--format_reward', type=float, default=0.0,
                 help='ablation (R36): reward for code that runs and returns a '
                      'number but is wrong; 0 = binary reward (main line). Use '
                      'with --no_std, or the group std rescales it to full strength')
  p.add_argument('--grad_clip', type=float, default=1.0)
  p.add_argument('--chunk', type=int, default=8, help='trajectories per backward')
  p.add_argument('--eval_every', type=int, default=10)
  p.add_argument('--eval_split', default='dev200')
  p.add_argument('--eval_steps', type=int, default=512)
  p.add_argument('--no_eval', action='store_true')
  p.add_argument('--seed', type=int, default=1)
  p.add_argument('--updates', type=int, default=1,
                 help='updates per rollout batch (D32); 1 = on-policy, clip inactive')
  p.add_argument('--keep_ckpts', choices=['all', 'best_last'], default='all',
                 help='best_last: keep only the best-so-far (mean dev acc over '
                      'both temperatures, ties -> earlier) and the latest checkpoint')
  p.add_argument('--stop_logp', type=float, default=None,
                 help='brake (R39): stop when the mean action log-prob of the last '
                      '--stop_window steps falls below this (D5 binary arm diverged '
                      'from about -0.07 to -1.8); the stopping step is saved and evaluated')
  p.add_argument('--stop_ratio', type=float, default=None,
                 help='brake relative to the start: stop when the last-window mean '
                      'action log-prob < ratio x the mean of the first window')
  p.add_argument('--stop_window', type=int, default=10)
  p.add_argument('--kl', type=float, default=0.0,
                 help='KL penalty to the init checkpoint (R41): + beta * mean over '
                      'sampled rounds of (log p_theta - log p_ref) on the actions. '
                      'k3 estimator (Schulman): exp(log p_ref - log p_theta) - 1 - '
                      '(log p_ref - log p_theta), unbiased, non-negative. The '
                      'reference is the same tracer with parameters restored from '
                      'the init checkpoint.')
  p.add_argument('--pool_size', type=int, default=0,
                 help='sanity check: train on the first N questions only (cycling)')
  args = p.parse_args()

  from rl import grpo, logprob, rollout
  import dataloader
  import omegaconf
  torch.manual_seed(args.seed)
  tok = dataloader.get_tokenizer(omegaconf.OmegaConf.create(
    {'data': {'tokenizer_name_or_path': 'HuggingFaceTB/SmolLM-135M'}}))
  config = sft.build_config(args.init_ckpt)
  sep = config.data.separator
  base = torch.load(args.init_ckpt, map_location='cpu', weights_only=False, mmap=True)
  assert sep == sft.SEP == base['hyper_parameters']['config'].data.separator
  model = sft.load_model(config, tok, args.init_ckpt).to('cuda')
  model.backbone.eval()  # policy = eval-mode network (no dropout)
  params = [q for q in model._get_parameters() if q.requires_grad]
  if args.kl:
    # Reference = the policy as loaded, i.e. the init checkpoint's EMA weights
    # (what every evaluation of the start model uses), frozen. Deep copy after
    # sft.load_model so the reference is never updated by the optimizer.
    import copy
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for q in ref_model.parameters():
      q.requires_grad_(False)
    assert all(x.data_ptr() != y.data_ptr()
               for x, y in zip(ref_model.parameters(), model.parameters()))
  else:
    ref_model = None
  opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
                          weight_decay=0.0)
  pool, n_excl = question_pool(args.seed)
  if args.pool_size:
    pool = pool[:args.pool_size]

  run_dir = os.path.join(REPO, 'runs', args.run_name)
  ckpt_dir = os.path.join(CKPT_ROOT, args.run_name)
  assert not os.path.exists(os.path.join(run_dir, 'metrics.json')), run_dir
  os.makedirs(os.path.join(run_dir, 'samples'), exist_ok=True)
  info = {'args': vars(args), 'pool': len(pool), 'rl_exclude': n_excl,
          'separator': sep,
          'git_commit': subprocess.run(['git', '-C', REPO, 'rev-parse', 'HEAD'],
                                       capture_output=True, text=True).stdout.strip(),
          'git_dirty': len(subprocess.run(['git', '-C', REPO, 'status', '--porcelain'],
                                          capture_output=True, text=True).stdout.splitlines()),
          'gpu': torch.cuda.get_device_name(0), 'start': time.strftime('%Y-%m-%dT%H:%M:%S')}
  with open(os.path.join(run_dir, 'run_info.json'), 'w') as f:
    json.dump(info, f, indent=1)
  log = open(os.path.join(run_dir, 'train_log.jsonl'), 'w')
  evals = []
  recent_logp, start_logp = [], []
  stopped = None
  gen = torch.Generator().manual_seed(args.seed)
  cursor, t_start = 0, time.time()

  for step in range(1, args.steps + 1):
    lr = args.lr * min(1.0, step / max(1, args.warmup))
    for q in opt.param_groups:
      q['lr'] = lr
    if cursor + args.questions > len(pool):
      cursor = 0
    batch = pool[cursor:cursor + args.questions]
    cursor += args.questions

    # 1. rollouts
    t0 = time.time()
    ro = rollout.generate(model, tok, [r['prompt'] for r in batch],
                          [r['response_ground_truth'] for r in batch],
                          group_size=args.group, num_steps=args.rollout_steps,
                          temperature=args.temperature, separator=sep,
                          batch_size=args.questions * args.group,
                          format_reward=args.format_reward)
    t_roll = time.time() - t0

    # 2-3. advantages, dynamic sampling
    adv = grpo.group_advantages(ro.rewards, ro.group, scale_std=not args.no_std)
    keep = grpo.mixed_groups(ro.rewards, ro.group)
    idx = keep.nonzero()[:, 0]
    correct = (ro.rewards == 1.0).float()
    rec = {'step': step, 'lr': lr, 'reward': float(ro.rewards.mean()),
           'correct': float(correct.mean()),
           'solved_any': float(torch.stack([correct[ro.group == g].max()
                                            for g in ro.group.unique()]).mean()),
           'kept_rows': int(len(idx)), 'rollout_s': round(t_roll, 1),
           'resp_chars': sum(map(len, ro.responses)) / len(ro.responses)}
    if ro.categories is not None:
      for cat in ('wrong', 'no_number', 'runtime', 'syntax', 'no_fn'):
        rec[f'cat_{cat}'] = ro.categories.count(cat) / len(ro.categories)

    # 4-5. loss and update
    t0 = time.time()
    opt.zero_grad(set_to_none=True)
    if len(idx) > 0:
      tokens, rs = ro.tokens[idx], ro.reveal_step[idx]
      a = adv[idx]
      n_act = grpo.action_counts(rs, tokens, tok.pad_token_id,
                                 count_pad=not args.no_pad).sum().reshape(1)
      rounds, w = logprob.select_rounds(rs, args.rounds, generator=gen)
      chunks = [slice(s, s + args.chunk) for s in range(0, len(idx), args.chunk)]
      old = None
      if args.updates > 1:  # old log-probs of this batch, same code path and rounds
        with torch.no_grad():
          old = [logprob.logp_at_rounds(model, tokens[c], rs[c], rounds[c],
                                        args.temperature, count_pad=not args.no_pad,
                                        chunk=args.chunk * args.rounds)[0]
                 for c in chunks]
      gnorms, clip_fracs, ratio_means = [], [], []
      for u in range(args.updates):
        opt.zero_grad(set_to_none=True)
        tot_loss, lp_sum, lp_n, n_clip, r_sum, kl_sum = 0.0, 0.0, 0, 0.0, 0.0, 0.0
        for ci, c in enumerate(chunks):
          lp, act = logprob.logp_at_rounds(model, tokens[c], rs[c], rounds[c],
                                           args.temperature,
                                           count_pad=not args.no_pad,
                                           chunk=args.chunk * args.rounds)
          ref = lp.detach() if old is None else old[ci]
          loss, st = grpo.grpo_loss(lp, ref, act, w[c], a[c], n_act,
                                    args.eps_low, args.eps_high)
          k = int(act.sum())
          if ref_model is not None:
            # same canvas/rounds: the trace only differs in the weights
            with torch.no_grad():
              lp_ref, _ = logprob.logp_at_rounds(ref_model, tokens[c], rs[c], rounds[c],
                                                 args.temperature,
                                                 count_pad=not args.no_pad,
                                                 chunk=args.chunk * args.rounds)
            d = lp_ref - lp
            kl_tok = (torch.exp(d) - 1 - d)
            kl = (w[c][:, :, None] * act * kl_tok).sum() / n_act.sum().clamp(min=1)
            loss = loss + args.kl * kl
            kl_sum += float(kl.detach()) * k
          loss.backward()
          tot_loss += float(loss.detach())
          lp_sum += float(lp.detach()[act].sum())
          lp_n += k
          n_clip += st['clip_frac'] * k
          r_sum += st['ratio_mean'] * k
        gnorms.append(float(torch.nn.utils.clip_grad_norm_(params, args.grad_clip)))
        opt.step()
        clip_fracs.append(n_clip / max(lp_n, 1))
        ratio_means.append(r_sum / max(lp_n, 1))
        if u == 0:
          first_loss, first_logp = tot_loss, lp_sum / max(lp_n, 1)
          first_kl = kl_sum / max(lp_n, 1)
      rec.update({'loss': first_loss, 'grad_norm': sum(gnorms) / len(gnorms),
                  'action_logp': first_logp,
                  **({'kl': first_kl} if ref_model is not None else {}),
                  'clip_frac_last': clip_fracs[-1], 'ratio_mean_last': ratio_means[-1]})
    else:
      rec.update({'loss': 0.0, 'grad_norm': 0.0, 'action_logp': None, 'skipped': True})
    rec['train_s'] = round(time.time() - t0, 1)
    rec['elapsed_s'] = round(time.time() - t_start, 1)
    log.write(json.dumps(rec) + '\n')
    log.flush()
    print(json.dumps(rec), flush=True)
    if step % args.eval_every == 0 or step == 1:
      with open(os.path.join(run_dir, 'samples', f'step{step}.json'), 'w') as f:
        json.dump([{'q': batch[int(g)]['prompt'], 'r': r, 'reward': float(x),
                    'cat': None if ro.categories is None else ro.categories[i]}
                   for i, (g, r, x) in enumerate(zip(ro.group.tolist(), ro.responses,
                                                     ro.rewards.tolist()))], f)
    del ro
    torch.cuda.empty_cache()

    # brake: diverging policy (action log-prob falling)
    if rec.get('action_logp') is not None:
      recent_logp = (recent_logp + [rec['action_logp']])[-args.stop_window:]
      if len(start_logp) < args.stop_window:
        start_logp.append(rec['action_logp'])
    threshold = args.stop_logp
    if args.stop_ratio is not None and len(start_logp) == args.stop_window:
      threshold = args.stop_ratio * sum(start_logp) / args.stop_window
    if (threshold is not None and len(recent_logp) == args.stop_window
        and sum(recent_logp) / args.stop_window < threshold):
      stopped = {'step': step, 'mean_action_logp': sum(recent_logp) / args.stop_window,
                 'threshold': threshold}
      print('STOP', json.dumps(stopped), flush=True)

    # periodic eval (same protocol as the base runs)
    if not args.no_eval and (step % args.eval_every == 0 or step == args.steps
                             or stopped is not None):
      path = os.path.join(ckpt_dir, f'step{step}.ckpt')
      sft.save_ckpt(model, base, path)
      e = {'step': step, 'ckpt': path}
      for T in (1.0, 0.1):
        name = f'{args.run_name}-s{step}-{args.eval_split}-e{args.eval_steps}-T{T}'
        e[f'acc_T{T}'] = run_eval(path, name, T, args.eval_steps, args.eval_split)
      evals.append(e)
      print('EVAL', json.dumps(e), flush=True)
      if args.keep_ckpts == 'best_last':
        def score(x):
          v = [x.get(f'acc_T{T}') for T in (1.0, 0.1)]
          return None if None in v else sum(v) / 2
        scored = [x for x in evals if score(x) is not None]
        best = max(scored, key=lambda x: (score(x), -x['step'])) if scored else None
        for x in evals[:-1]:
          if x is not best and os.path.exists(x['ckpt']):
            os.remove(x['ckpt'])
            x['ckpt_deleted'] = True
    if stopped is not None:
      break

  log.close()
  with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
    json.dump({'evals': evals, 'stopped_early': stopped,
               'train_seconds': round(time.time() - t_start, 1),
               'peak_gpu_mem_mib': torch.cuda.max_memory_allocated() // 2**20},
              f, indent=1)


if __name__ == '__main__':
  main()
