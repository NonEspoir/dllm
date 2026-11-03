"""Figures for the report and slides (R47). Writes PNGs to docs/figures/.

Every figure is regenerated from runs/ on each call, so it can be re-run after a
new run finishes; runs that do not exist yet are skipped and the figure is drawn
with what is available.

  python scripts/make_figures.py            # all figures
  python scripts/make_figures.py fig_dev200_curves

Colour: the reference categorical palette (fixed order, never cycled), light
surface. Colour follows the run (entity), the same order everywhere. One y-axis
per panel; a legend is present whenever there are two or more series. The report
carries the same numbers as tables, so no figure is the only place a number
appears.
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
RUNS = os.path.join(REPO, 'runs')
OUT = os.path.join(REPO, 'docs', 'figures')

# reference palette, light column, fixed slot order
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({
  'figure.dpi': 200, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
  'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
  'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'text.color': INK,
  'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True,
  'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
  'axes.spines.top': False, 'axes.spines.right': False,
  'legend.frameon': False, 'legend.fontsize': 7,
})

# the runs shown in the RL figures, in the fixed order (colour follows the run)
ARMS = [
  ('base (no RL)', None, None),
  ('binary reward', '1030-grpo-base-d5-binary', 'binary'),
  ('format reward', '1030-grpo-base-d5-fmt0.1', 'format'),
  ('format + KL', '1031-grpo-base-fmt0.1-kl0.04', 'fmt+KL'),
  ('format, 300 steps', '1031-grpo-base-fmt0.1-s300', 'format 300'),
  ('SFT -> RL', '1031-grpo-sft-e1-fmt0.1', 'SFT->RL'),
  ('convergence (KL+cos)', '1101-grpo-conv-fmt0.1-kl0.04', 'conv'),
  ('ELBO, no KL', '1103-grpo-elbo-conv-fmt0.1-nokl', 'ELBO'),
]
BASE_DEV200 = {'1.0': 0.215, '0.1': 0.390}


def evals(name):
  p = os.path.join(RUNS, name, 'stdout.txt')
  if not os.path.exists(p):
    return []
  with open(p) as f:
    return [json.loads(l[5:]) for l in f if l.startswith('EVAL ')]


def logs(name):
  p = os.path.join(RUNS, name, 'train_log.jsonl')
  if not os.path.exists(p):
    return []
  with open(p) as f:
    return [json.loads(l) for l in f]


def ma(rows, key, w=25):
  out = []
  for i in range(len(rows)):
    v = [r[key] for r in rows[max(0, i - w + 1):i + 1] if isinstance(r.get(key), (int, float))]
    out.append((rows[i]['step'], sum(v) / len(v)))
  return out


def save(fig, name):
  path = os.path.join(OUT, name + '.png')
  fig.savefig(path)
  plt.close(fig)
  print('wrote', os.path.relpath(path, REPO))


def fig_dev200_curves():
  fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
  for ax, T in zip(axes, ('1.0', '0.1')):
    for slot, (label, name, short) in enumerate(ARMS):
      if name is None:
        ax.axhline(BASE_DEV200[T], color=INK2, lw=1.0, ls=(0, (4, 3)))
        ax.annotate('base (no RL)', (5, BASE_DEV200[T]), xytext=(0, 4),
                    textcoords='offset points', color=INK2, fontsize=7)
        continue
      e = [(x['step'], x[f'acc_T{T}']) for x in evals(name) if x.get(f'acc_T{T}') is not None]
      if not e:
        continue
      xs, ys = zip(*e)
      ax.plot(xs, ys, color=C[slot], lw=1.6, marker='o', ms=3.2, label=short)
    ax.set_title(f'dev200, T={T}')
    ax.set_xlabel('training step')
    ax.set_ylim(0.15, 0.6)
  axes[0].set_ylabel('accuracy')
  h, l = axes[0].get_legend_handles_labels()
  fig.tight_layout()
  fig.subplots_adjust(bottom=0.32)
  fig.legend(h, l, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.03))
  save(fig, 'fig_dev200_curves')


def fig_health_logp():
  fig, ax = plt.subplots(figsize=(4.6, 2.5))
  for slot, (label, name, short) in enumerate(ARMS):
    if name is None:
      continue
    rows = logs(name)
    if not rows:
      continue
    m = ma([r for r in rows if r.get('action_logp') is not None], 'action_logp')
    xs, ys = zip(*m)
    ax.plot(xs, ys, color=C[slot], lw=1.6, label=short)
  ax.set_xlabel('training step')
  ax.set_ylabel('mean action log-prob (25-step mean)')
  ax.set_title('Policy drift: falling curves precede divergence')
  ax.legend(loc='lower left', ncol=2)
  save(fig, 'fig_health_logp')


def fig_train_correct():
  fig, ax = plt.subplots(figsize=(4.6, 2.5))
  for slot, (label, name, short) in enumerate(ARMS):
    if name is None:
      continue
    rows = logs(name)
    if not rows:
      continue
    xs, ys = zip(*ma(rows, 'correct'))
    ax.plot(xs, ys, color=C[slot], lw=1.6, label=short)
  ax.set_xlabel('training step')
  ax.set_ylabel('rollout accuracy (25-step mean)')
  ax.set_title('Training-batch accuracy')
  ax.legend(loc='upper left', ncol=2)
  save(fig, 'fig_train_correct')


def fig_sft():
  data = [('v1b', '1e-5', '1030-sft-v1b-lr1e-5'), ('v1b', '3e-5', '1030-sft-v1b-lr3e-5'),
          ('v1b', '1e-4', '1030-sft-v1b-lr1e-4'), ('unfiltered', '1e-5', '1030-sft-unfiltered-lr1e-5')]
  fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
  for slot, (data_name, lr, run) in enumerate(data):
    rows = logs(run)
    if not rows:
      continue
    # val loss: last logged loss of each epoch
    ep, loss = [], []
    for r in rows:
      if not ep or r['epoch'] != ep[-1]:
        ep.append(r['epoch'])
        loss.append(r['loss'])
      else:
        loss[-1] = r['loss']
    axes[0].plot(ep, loss, color=C[slot], lw=1.6, marker='o', ms=3.2, label=f'{data_name}, lr {lr}')
    xs, ys = [], []
    for e in (1, 2, 3, 4, 5):
      p = os.path.join(RUNS, f'1030-eval-sft-{data_name}-lr{lr}-e{e}-s512-T0.1', 'metrics.json')
      if os.path.exists(p):
        xs.append(e)
        ys.append(100 * json.load(open(p))['accuracy'])
    if xs:
      axes[1].plot(xs, ys, color=C[slot], lw=1.6, marker='o', ms=3.2, label=f'{data_name}, lr {lr}')
  axes[0].set_xlabel('epoch')
  axes[0].set_ylabel('training loss')
  axes[0].set_title('SFT loss keeps falling')
  axes[1].axhline(38.6, color=INK2, lw=1.0, ls=(0, (4, 3)))
  axes[1].annotate('base 38.6', (1, 38.6), xytext=(0, 3), textcoords='offset points',
                   color=INK2, fontsize=7)
  axes[1].set_xlabel('epoch')
  axes[1].set_ylabel('dev1000 accuracy, T=0.1 (%)')
  axes[1].set_title('... while accuracy does not improve')
  axes[0].legend(loc='upper right')
  save(fig, 'fig_sft')


def fig_kl_and_grad():
  pairs = [('convergence (trajectory + KL)', '1101-grpo-conv-fmt0.1-kl0.04'),
           ('ELBO, no KL', '1103-grpo-elbo-conv-fmt0.1-nokl'),
           ('ELBO + KL (aborted)', '1103-grpo-elbo-conv-fmt0.1-kl0.04-aborted')]
  fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
  for slot, (label, name) in enumerate(pairs):
    rows = logs(name)
    if not rows:
      continue
    kl = [(r['step'], r['kl']) for r in rows if r.get('kl')]  # 0 = step 1, no data
    axes[0].plot(*zip(*kl), color=C[slot], lw=1.6, label=label)
    g = ma([r for r in rows if r.get('grad_norm_max')], 'grad_norm_max', 10)
    axes[1].plot(*zip(*g), color=C[slot], lw=1.6, label=label)
  axes[0].set_yscale('log')
  axes[0].set_xlabel('training step')
  axes[0].set_ylabel('KL vs start model (nats/action)')
  axes[0].set_title('KL differs ~1000x between estimators')
  axes[0].legend(loc='lower right')
  axes[1].set_yscale('log')
  axes[1].set_xlabel('training step')
  axes[1].set_ylabel('max grad norm (10-step mean)')
  axes[1].set_title('Gradient scale (clipped to 1.0)')
  save(fig, 'fig_kl_and_grad')


def fig_test_main():
  models = [('base', 'base'), ('SFT', 'sft'), ('RL (convergence)', 'rl')]
  settings = [('512 / T=1', 's512-T1.0'), ('512 / T=0.1', 's512-T0.1'), ('1024 / T=1', 's1024-T1.0')]
  fig, ax = plt.subplots(figsize=(4.8, 2.4))
  width = 0.26
  for mi, (label, tag) in enumerate(models):
    xs, ys = [], []
    for si, (slabel, s) in enumerate(settings):
      p = os.path.join(RUNS, f'1103-test-{tag}-{s}', 'metrics.json')
      if os.path.exists(p):
        xs.append(si + (mi - 1) * width)
        ys.append(100 * json.load(open(p))['accuracy'])
    if xs:
      ax.bar(xs, ys, width=width - 0.02, color=C[mi], label=label)
      for x, y in zip(xs, ys):
        ax.annotate(f'{y:.1f}', (x, y), xytext=(0, 2), textcoords='offset points',
                    ha='center', fontsize=6.5, color=INK2)
  ax.set_xticks(range(len(settings)), [s[0] for s in settings])
  ax.set_xlabel('steps / temperature')
  ax.set_ylabel('GSM8K test accuracy (%)')
  ax.set_title('Test set (1319 questions, one seed each)')
  ax.set_ylim(0, 46)
  ax.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, 1.02))
  save(fig, 'fig_test_main')


def fig_vote():
  fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=True)
  for ax, T in zip(axes, ('1.0', '0.1')):
    for slot, (label, prefix) in enumerate([('base', f'1103-dev-base-mv-T{T}'),
                                            ('RL (convergence)', f'1103-dev-rl-mv-T{T}')]):
      rows, cols = [], []
      k = 0
      while True:
        p = os.path.join(RUNS, f'{prefix}-seed{k + 1}', 'samples', 'results.json')
        if not os.path.exists(p):
          break
        rows.append(json.load(open(p))['records'])
        k += 1
      if not rows:
        continue
      sys.path.insert(0, os.path.dirname(__file__))
      import majority_vote as mv
      assert all(len(r) == len(rows[0]) for r in rows)
      golds = [r['response_ground_truth'] for r in rows[0]]
      vals = [[mv.value(rows[j][i]['response']) for j in range(len(rows))] for i in range(len(rows[0]))]
      maj, pas = [], []
      for kk in range(1, len(rows) + 1):
        maj.append(100 * sum(mv.correct(mv.vote(v[:kk]), g) for v, g in zip(vals, golds)) / len(golds))
        pas.append(100 * sum(any(mv.correct(x, g) for x in v[:kk]) for v, g in zip(vals, golds)) / len(golds))
      ks = list(range(1, len(rows) + 1))
      ax.plot(ks, maj, color=C[slot], lw=1.6, marker='o', ms=3.2,
              label=f'{label}: majority vote')
      ax.plot(ks, pas, color=C[slot], lw=1.2, ls=(0, (3, 2)),
              label=f'{label}: pass@k')
    ax.set_title(f'Majority vote, T={T}')
    ax.set_xlabel('k samples per question')
  axes[0].set_ylabel('dev1000 (%)')
  axes[0].legend(loc='lower right')
  save(fig, 'fig_vote')


def fig_steps():
  fig, ax = plt.subplots(figsize=(4.2, 2.3))
  def series(prefix, slot, label, pattern='1103-dev-rl-s{}-T1.0'):
    xs, ys = [], []
    for steps in (128, 256, 512, 1024):
      p = os.path.join(RUNS, pattern.format(steps), 'metrics.json')
      if os.path.exists(p):
        xs.append(steps)
        ys.append(100 * json.load(open(p))['accuracy'])
    ax.plot(xs, ys, color=C[slot], lw=1.6, marker='o', ms=3.2, label=label)
  series(None, 0, 'RL (convergence)')
  base = {64: 10.4, 128: 14.3, 256: 15.6, 512: 17.8, 1024: 17.5}
  xs = sorted(base)
  ax.plot(xs, [base[k] for k in xs], color=INK2, lw=1.0, ls=(0, (4, 3)), marker='o',
          ms=2.5, label='base (D1 sweep)')
  ax.set_xscale('log', base=2)
  ax.set_xticks([64, 128, 256, 512, 1024], ['64', '128', '256', '512', '1024'])
  ax.set_xlabel('sampling steps')
  ax.set_ylabel('dev1000 accuracy, T=1 (%)')
  ax.set_title('Step robustness')
  ax.legend(loc='lower right')
  save(fig, 'fig_steps')


FIGS = {f.__name__: f for f in (fig_dev200_curves, fig_health_logp, fig_train_correct, fig_sft,
                                fig_kl_and_grad, fig_test_main, fig_vote, fig_steps)}


def main():
  os.makedirs(OUT, exist_ok=True)
  want = sys.argv[1:] or list(FIGS)
  for name in want:
    FIGS[name]()


if __name__ == '__main__':
  main()
