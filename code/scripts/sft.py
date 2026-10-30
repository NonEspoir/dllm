"""SFT for MDLM on the DeepSeek-written GSM8K data (data/sft/, D09).

Why not upstream `main.py mode=train training.finetune_path=...`:
  1. `load_from_checkpoint` restores the raw training weights, but every
     evaluation (including our base numbers) uses the EMA weights, and the two
     differ by ~10% relative L2 in the released checkpoint. SFT would not start
     from the model we measured.
  2. The restored EMA has decay 0.9999 and num_updates=250k, so a few hundred
     SFT steps would move the evaluated (EMA) weights by only a few percent.
  3. Lightning would also restore the pre-training dataloader position and the
     2500-step warmup schedule.
So this script runs a plain PyTorch loop and reuses the upstream loss
(`MDLM._loss`, the NELBO used in pre-training) unchanged.

Data format is identical to pre-training (dataloader.get_tiny_gsm_dataset with
train_on_prompt=False, train_on_pad=True, filter_too_long=True):
  [BOS] question.strip() SEP code.strip() [EOS] [PAD]*  (length 512)
  loss mask: 0 on [BOS] question SEP, 1 on code, EOS and every PAD.
SEP is the two characters backslash + 'n' (tokens [76, 94]), not a newline:
upstream's yaml writes `separator: '\\n'` in single quotes, which YAML does not
unescape, and both pre-training (saved config) and GSM8K eval use it. A real
newline here made SFT learn a layout the model is never evaluated with.

Checkpoints keep the upstream Lightning layout (so scripts/eval_gsm8k.sh can
load them with CKPT_PATH=...), weights only: `state_dict` and `ema.shadow_params`
point to the same tensors, so evaluation uses exactly the trained weights.

  python -m scripts.sft --run_name 1031-sft-v1b-lr3e-5 --select sft_v1b --lr 3e-5
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

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
DATA = os.path.join(REPO, 'data/sft/deepseek_v1.jsonl')
DEV_IDS = os.path.join(REPO, 'data/splits/gsm8k_dev_ids.json')
BASE_CKPT = '/root/autodl-tmp/checkpoints/tinygsm/mdlm.ckpt'
CKPT_ROOT = '/root/autodl-tmp/sft_ckpt'
BLOCK = 512
SEP = '\\n'  # literal backslash-n, see module docstring

SELECT = {
  # v1: hard checks only (format, exec == gold, not a literal, <= 512 tokens)
  'hard_pass': lambda r: r['hard_pass'],
  # v1b: v1 after the semantic check (data/sft/README.md "语义检查")
  'sft_v1b': lambda r: r['sft_v1b'],
  # D10 ablation: no correctness filter at all
  'unfiltered': lambda r: r['checks']['length_ok'],
  # optional ablation: drop samples that trip a soft check
  'clean': lambda r: r['hard_pass'] and not r['suspicious'],
}


def load_records(select):
  with open(DEV_IDS) as f:
    dev = set(json.load(f)['dev_ids'])
  with open(DATA) as f:
    rows = [json.loads(l) for l in f]
  rows = [r for r in rows if SELECT[select](r)]
  assert not dev & {r['gsm8k_index'] for r in rows}, 'dev leaked into SFT data'
  return rows


def tokenize(rows, tokenizer):
  """Same layout as dataloader.get_tiny_gsm_dataset (no wrap, pad to 512)."""
  bos, eos, pad = (tokenizer.bos_token_id, tokenizer.eos_token_id,
                   tokenizer.pad_token_id)
  sep = tokenizer(SEP, add_special_tokens=False).input_ids
  ids, masks, kept, too_long = [], [], [], 0
  for r in rows:
    q = tokenizer(r['question'].strip(), add_special_tokens=False).input_ids
    a = tokenizer(r['code'].strip(), add_special_tokens=False).input_ids
    x = [bos] + q + sep + a + [eos]
    if len(x) > BLOCK:  # filter_too_long=True in pre-training
      too_long += 1
      continue
    prompt_len = 1 + len(q) + len(sep)
    ids.append(x + [pad] * (BLOCK - len(x)))
    masks.append([0] * prompt_len + [1] * (BLOCK - prompt_len))
    kept.append(r['gsm8k_index'])
  return (torch.tensor(ids, dtype=torch.long),
          torch.tensor(masks, dtype=torch.long), kept, too_long)


def build_config(ckpt):
  import hydra
  import main as upstream_main  # noqa: F401  (registers OmegaConf resolvers)
  with hydra.initialize(version_base=None, config_path='../configs'):
    return hydra.compose(config_name='config', overrides=[
      'mode=train', 'data=tiny-gsm',
      'data.tokenizer_name_or_path=HuggingFaceTB/SmolLM-135M',
      'model=small', 'model.length=512', 'algo=mdlm',
      'sampler=fast_ancestral', f'eval.checkpoint_path={ckpt}',
      'trainer.devices=1', 'trainer.num_nodes=1'])


def load_model(config, tokenizer, ckpt):
  import algo
  model = algo.MDLM.load_from_checkpoint(
    ckpt, tokenizer=tokenizer, config=config, strict=True)
  # Start from the EMA weights (the ones every evaluation uses), then drop
  # the 0.9999 EMA so the trained weights are what gets evaluated.
  model.ema.copy_to(model._get_parameters())
  model.ema = None
  return model


def lr_at(step, total, args):
  warm = max(1, int(args.warmup_frac * total))
  if step < warm:
    return args.lr * (step + 1) / warm
  if args.schedule == 'constant':
    return args.lr
  p = (step - warm) / max(1, total - warm)
  return args.lr * (args.min_lr_frac + (1 - args.min_lr_frac)
                    * 0.5 * (1 + math.cos(math.pi * p)))


@torch.no_grad()
def val_loss(model, ids, masks, batch, device):
  """Mean NELBO per loss token on the held-out SFT split. Fixed noise seed so
  values are comparable across epochs."""
  model.backbone.eval()
  tot, n = 0.0, 0
  with torch.random.fork_rng(devices=[device]):
    torch.manual_seed(1234)
    for i in range(0, len(ids), batch):
      x, m = ids[i:i + batch].to(device), masks[i:i + batch].to(device)
      out = model._loss(x, m, train_mode=False)
      tot += float(out.nlls)
      n += int(out.num_tokens)
  model.backbone.train()
  return tot / max(n, 1)


def save_ckpt(model, base, path):
  sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        if k in base['state_dict']}
  assert sd.keys() == base['state_dict'].keys()
  shadow = [sd['backbone.' + n] for n, p in model.backbone.named_parameters()
            if p.requires_grad]
  assert len(shadow) == len(base['ema']['shadow_params'])
  ck = {k: base[k] for k in ['epoch', 'global_step', 'pytorch-lightning_version',
                             'loops', 'callbacks', 'hparams_name',
                             'hyper_parameters', 'sampler'] if k in base}
  ck['state_dict'] = sd
  # Same tensor objects: torch.save stores them once; at eval time the EMA
  # copy is a no-op.
  ck['ema'] = {'decay': 0.0, 'num_updates': 0, 'shadow_params': shadow}
  ck['optimizer_states'] = []
  ck['lr_schedulers'] = []
  os.makedirs(os.path.dirname(path), exist_ok=True)
  torch.save(ck, path)


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--run_name', required=True)
  p.add_argument('--select', choices=sorted(SELECT), default='sft_v1b')  # R29
  p.add_argument('--init_ckpt', default=BASE_CKPT)
  p.add_argument('--lr', type=float, default=3e-5)
  p.add_argument('--schedule', choices=['cosine', 'constant'], default='cosine')
  p.add_argument('--warmup_frac', type=float, default=0.05)
  p.add_argument('--min_lr_frac', type=float, default=0.1)
  p.add_argument('--weight_decay', type=float, default=0.0)  # upstream: 0
  p.add_argument('--epochs', type=int, default=5)
  p.add_argument('--batch', type=int, default=64)
  p.add_argument('--accum', type=int, default=1)
  p.add_argument('--grad_clip', type=float, default=1.0)  # upstream: 1.0
  p.add_argument('--val_frac', type=float, default=0.02)
  p.add_argument('--save_every_epoch', type=int, default=1)
  p.add_argument('--seed', type=int, default=1)
  p.add_argument('--max_steps', type=int, default=0,
                 help='stop early (smoke test); 0 = full schedule')
  p.add_argument('--dry_run', action='store_true',
                 help='build the data and print stats, no model / GPU')
  args = p.parse_args()

  import dataloader
  import omegaconf
  tokenizer = dataloader.get_tokenizer(omegaconf.OmegaConf.create(
    {'data': {'tokenizer_name_or_path': 'HuggingFaceTB/SmolLM-135M'}}))

  rows = load_records(args.select)
  ids, masks, kept, too_long = tokenize(rows, tokenizer)
  g = torch.Generator().manual_seed(args.seed)
  perm = torch.randperm(len(ids), generator=g)
  n_val = int(round(args.val_frac * len(ids)))
  val_idx, tr_idx = perm[:n_val], perm[n_val:]
  tr_ids, tr_masks = ids[tr_idx], masks[tr_idx]
  va_ids, va_masks = ids[val_idx], masks[val_idx]
  steps_per_epoch = len(tr_ids) // (args.batch * args.accum)
  if args.max_steps:
    steps_per_epoch = min(steps_per_epoch, args.max_steps)
    args.epochs = 1
  total_steps = steps_per_epoch * args.epochs
  stats = {'select': args.select, 'records': len(rows), 'too_long': too_long,
           'train': len(tr_ids), 'val': len(va_ids),
           'loss_tokens_per_example': float(masks.sum(1).float().mean()),
           'answer_tokens_per_example': float(
             ((ids != tokenizer.pad_token_id) & (masks == 1)).sum(1).float().mean()),
           'steps_per_epoch': steps_per_epoch, 'total_steps': total_steps}
  print(json.dumps(stats, indent=1))
  if args.dry_run:
    i = 0
    plen = int((masks[i] == 0).sum())
    print('--- prompt (no loss) ---')
    print(tokenizer.decode(ids[i, :plen]))
    print('--- target (loss), first 120 tokens ---')
    print(repr(tokenizer.decode(ids[i, plen:plen + 120])))
    print('pad from', int((ids[i] == tokenizer.pad_token_id).nonzero()[0]),
          'mask sum', int(masks[i].sum()))
    return

  run_dir = os.path.join(REPO, 'runs', args.run_name)
  ckpt_dir = os.path.join(CKPT_ROOT, args.run_name)
  assert not os.path.exists(os.path.join(run_dir, 'metrics.json')), run_dir
  os.makedirs(run_dir, exist_ok=True)
  info = {'args': vars(args), 'data_stats': stats,
          'git_commit': subprocess.run(['git', '-C', REPO, 'rev-parse', 'HEAD'],
                                       capture_output=True, text=True).stdout.strip(),
          'git_dirty': len(subprocess.run(['git', '-C', REPO, 'status', '--porcelain'],
                                          capture_output=True, text=True).stdout.splitlines()),
          'gpu': torch.cuda.get_device_name(0), 'start': time.strftime('%Y-%m-%dT%H:%M:%S'),
          'ckpt_dir': ckpt_dir}
  with open(os.path.join(run_dir, 'run_info.json'), 'w') as f:
    json.dump(info, f, indent=1)

  torch.manual_seed(args.seed)
  torch.backends.cuda.matmul.allow_tf32 = True
  device = torch.device('cuda')
  config = build_config(args.init_ckpt)
  base = torch.load(args.init_ckpt, map_location='cpu', weights_only=False, mmap=True)
  # Same separator as pre-training and as the GSM8K eval prompt.
  assert SEP == config.data.separator == base['hyper_parameters']['config'].data.separator
  model = load_model(config, tokenizer, args.init_ckpt).to(device)
  model.backbone.train()
  params = [q for q in model._get_parameters() if q.requires_grad]
  opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
                          weight_decay=args.weight_decay)

  log = open(os.path.join(run_dir, 'train_log.jsonl'), 'w')
  epochs_out = []
  v0 = val_loss(model, va_ids, va_masks, args.batch, device)
  print(f'epoch 0 val_loss {v0:.4f}')
  epochs_out.append({'epoch': 0, 'step': 0, 'val_loss': v0})
  step, t0 = 0, time.time()
  for epoch in range(1, args.epochs + 1):
    order = torch.randperm(len(tr_ids), generator=g)
    run_loss, run_n = 0.0, 0
    for s in range(steps_per_epoch):
      for q in opt.param_groups:
        q['lr'] = lr_at(step, total_steps, args)
      opt.zero_grad(set_to_none=True)
      for a in range(args.accum):
        j = order[(s * args.accum + a) * args.batch:(s * args.accum + a + 1) * args.batch]
        x, m = tr_ids[j].to(device), tr_masks[j].to(device)
        loss = model._loss(x, m, train_mode=True).loss / args.accum
        loss.backward()
        run_loss += float(loss) * args.accum
        run_n += 1
      gn = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
      opt.step()
      step += 1
      if step % 10 == 0 or step == total_steps:
        rec = {'step': step, 'epoch': epoch, 'loss': run_loss / run_n,
               'lr': opt.param_groups[0]['lr'], 'grad_norm': float(gn),
               'elapsed_s': round(time.time() - t0, 1)}
        log.write(json.dumps(rec) + '\n')
        log.flush()
        run_loss, run_n = 0.0, 0
    v = val_loss(model, va_ids, va_masks, args.batch, device)
    rec = {'epoch': epoch, 'step': step, 'val_loss': v,
           'elapsed_s': round(time.time() - t0, 1)}
    if epoch % args.save_every_epoch == 0 or epoch == args.epochs:
      path = os.path.join(ckpt_dir, f'epoch{epoch}.ckpt')
      save_ckpt(model, base, path)
      rec['ckpt'] = path
    epochs_out.append(rec)
    print(json.dumps(rec))
  log.close()
  with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
    json.dump({'epochs': epochs_out, 'train_seconds': round(time.time() - t0, 1),
               'peak_gpu_mem_mib': torch.cuda.max_memory_allocated() // 2**20},
              f, indent=1)


if __name__ == '__main__':
  main()
