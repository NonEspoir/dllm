"""Summary of the D2 SFT sweep: dev accuracy per (data, lr, epoch, T), paired
McNemar against base (same steps / T / seed / batch) and between SFT data
variants at the same lr and epoch; optionally picks the lr for the D10
ablation.

  python scripts/summarize_sft.py --prefix 1030 --steps 512 [--select_out f.json]

Run names (scripts/pipeline_d2_sft.sh):
  runs/<prefix>-sft-<data>-lr<lr>                  training (metrics.json: val_loss)
  (<data>: v1b for --select sft_v1b, unfiltered)
  runs/<prefix>-eval-sft-<data>-lr<lr>-e<ep>-s<steps>-T<T>   dev 1000 eval
Base: runs/<prefix>-eval-base-s<steps>-T<T> (same protocol, incl. the skip
flag); falls back to runs/1030-sweep-fast-dev-s<steps>-T<T> (D1 sweep, no skip).

lr selection rule (fixed before the sweep): for each lr take the epoch with
the highest mean dev accuracy over T=1.0 and T=0.1; pick the lr with the
highest such mean. Ties go to the smaller lr.
"""
import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.summarize_runs import load_metrics, mcnemar  # noqa: E402

RUNS = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../runs'))
TEMPS = ('1.0', '0.1')


def collect(prefix, steps):
  pat = re.compile(rf'^{prefix}-eval-sft-(?P<data>[a-z0-9_]+)-lr(?P<lr>[0-9.]+e-?\d+)'
                   rf'-e(?P<ep>\d+)-s{steps}-T(?P<T>[0-9.]+)$')
  acc = collections.defaultdict(dict)  # (data, lr, ep) -> T -> metrics
  for name in sorted(os.listdir(RUNS)):
    m = pat.match(name)
    if m and os.path.exists(os.path.join(RUNS, name, 'metrics.json')):
      acc[m['data'], m['lr'], int(m['ep'])][m['T']] = load_metrics(os.path.join(RUNS, name))
  return acc


def val_losses(prefix, data, lr):
  path = os.path.join(RUNS, f'{prefix}-sft-{data}-lr{lr}', 'metrics.json')
  if not os.path.exists(path):
    return {}
  with open(path) as f:
    return {e['epoch']: e['val_loss'] for e in json.load(f)['epochs']}


def pct(m):
  return f"{100 * m['accuracy']:.1f}" if m else '—'


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--prefix', required=True)
  p.add_argument('--steps', type=int, required=True)
  p.add_argument('--select_data', default='v1b')
  p.add_argument('--select_out')
  args = p.parse_args()

  acc = collect(args.prefix, args.steps)
  base, base_name, d1 = {}, {}, {}
  for T in TEMPS:
    base[T] = None
    for name in (f'{args.prefix}-eval-base-s{args.steps}-T{T}',
                 f'1030-sweep-fast-dev-s{args.steps}-T{T}'):
      d = os.path.join(RUNS, name)
      if os.path.exists(os.path.join(d, 'metrics.json')):
        if base[T] is None:
          base[T], base_name[T] = load_metrics(d), name
        else:
          d1[T] = load_metrics(d)

  print(f'## SFT sweep, dev 1000, {args.steps} steps (accuracy %, McNemar vs base)\n')
  print('base: ' + ', '.join(f'T={T} {pct(base[T])} ({base_name.get(T)})' for T in TEMPS))
  for T in d1:  # same model, skip on vs off (D1 sweep)
    n01, n10, pv = mcnemar(d1[T]['per_question_correct'], base[T]['per_question_correct'])
    print(f'- base T={T}: this protocol {pct(base[T])} vs D1 sweep {pct(d1[T])}, '
          f'p={pv:.3g} (+{n01}/-{n10})')
  print()
  print('| data | lr | epoch | val loss | T=1.0 | p vs base | T=0.1 | p vs base | mean |')
  print('|---|---|---|---|---|---|---|---|---|')
  for (data, lr, ep) in sorted(acc, key=lambda k: (k[0], float(k[1]), k[2])):
    row = acc[data, lr, ep]
    cells = []
    for T in TEMPS:
      m = row.get(T)
      if m and base[T]:
        n01, n10, pv = mcnemar(base[T]['per_question_correct'], m['per_question_correct'])
        cells += [pct(m), f'{pv:.3g} (+{n01}/-{n10})']
      else:
        cells += [pct(m), '—']
    mean = (f"{50 * sum(row[T]['accuracy'] for T in TEMPS):.1f}"
            if all(T in row for T in TEMPS) else '—')
    vl = val_losses(args.prefix, data, lr).get(ep)
    print(f"| {data} | {lr} | {ep} | {vl:.4f} | " if vl is not None else f'| {data} | {lr} | {ep} | — | ',
          end='')
    print(' | '.join(cells) + f' | {mean} |')

  # Data ablation (D10): same lr and epoch, different data.
  datas = sorted({k[0] for k in acc})
  if len(datas) > 1:
    print('\n## Data ablation (paired, same lr / epoch / T)\n')
    for (data, lr, ep), row in sorted(acc.items()):
      if data != args.select_data:
        continue
      for other in datas:
        if other == data or (other, lr, ep) not in acc:
          continue
        for T in TEMPS:
          a, b = row.get(T), acc[other, lr, ep].get(T)
          if a and b:
            n01, n10, pv = mcnemar(a['per_question_correct'], b['per_question_correct'])
            print(f'- lr {lr} epoch {ep} T={T}: {data} {pct(a)} vs {other} {pct(b)}; '
                  f'only-{other}-correct={n01}, only-{data}-correct={n10}, p={pv:.3g}')

  # lr selection for the ablation.
  best = {}
  for (data, lr, ep), row in acc.items():
    if data != args.select_data or not all(T in row for T in TEMPS):
      continue
    mean = sum(row[T]['accuracy'] for T in TEMPS) / len(TEMPS)
    if lr not in best or mean > best[lr][0]:
      best[lr] = (mean, ep)
  if best:
    lr = max(sorted(best, key=float), key=lambda k: best[k][0])  # ties -> smaller lr
    sel = {'rule': 'best-epoch mean dev accuracy over T=1.0 and T=0.1; ties -> smaller lr',
           'data': args.select_data, 'steps': args.steps, 'lr': lr,
           'best_epoch': best[lr][1], 'mean_accuracy': best[lr][0],
           'per_lr': {k: {'mean_accuracy': v[0], 'best_epoch': v[1]} for k, v in best.items()}}
    print('\n## lr selection\n')
    print(json.dumps(sel, indent=1))
    if args.select_out:
      with open(args.select_out, 'w') as f:
        json.dump(sel, f, indent=1)


if __name__ == '__main__':
  main()
