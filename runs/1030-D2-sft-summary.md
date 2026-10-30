## SFT sweep, dev 1000, 512 steps (accuracy %, McNemar vs base)

base: T=1.0 18.0 (1030-eval-base-s512-T1.0), T=0.1 38.6 (1030-eval-base-s512-T0.1)
- base T=1.0: this protocol 18.0 vs D1 sweep 17.8, p=0.754 (+6/-4)
- base T=0.1: this protocol 38.6 vs D1 sweep 38.8, p=0.688 (+2/-4)

| data | lr | epoch | val loss | T=1.0 | p vs base | T=0.1 | p vs base | mean |
|---|---|---|---|---|---|---|---|---|
| unfiltered | 1e-5 | 1 | 0.0829 | 15.9 | 0.0527 (+43/-64) | 38.9 | 0.87 (+76/-73) | 27.4 |
| unfiltered | 1e-5 | 2 | 0.0774 | 14.0 | 0.00033 (+40/-80) | 36.6 | 0.157 (+80/-100) | 25.3 |
| unfiltered | 1e-5 | 3 | 0.0749 | 14.0 | 0.00108 (+52/-92) | 37.9 | 0.666 (+93/-100) | 25.9 |
| unfiltered | 1e-5 | 4 | 0.0738 | 14.0 | 0.00117 (+53/-93) | 37.2 | 0.365 (+96/-110) | 25.6 |
| unfiltered | 1e-5 | 5 | 0.0733 | 13.7 | 0.000406 (+50/-93) | 37.7 | 0.573 (+96/-105) | 25.7 |
| v1b | 1e-5 | 1 | 0.0699 | 16.2 | 0.095 (+43/-61) | 37.8 | 0.568 (+71/-79) | 27.0 |
| v1b | 1e-5 | 2 | 0.0638 | 14.6 | 0.00337 (+47/-81) | 37.5 | 0.45 (+82/-93) | 26.1 |
| v1b | 1e-5 | 3 | 0.0612 | 13.5 | 0.000105 (+43/-88) | 37.2 | 0.356 (+92/-106) | 25.4 |
| v1b | 1e-5 | 4 | 0.0603 | 13.5 | 0.000134 (+45/-90) | 38.1 | 0.785 (+105/-110) | 25.8 |
| v1b | 1e-5 | 5 | 0.0598 | 14.7 | 0.00683 (+54/-87) | 38.1 | 0.783 (+103/-108) | 26.4 |
| v1b | 3e-5 | 1 | 0.0601 | 13.3 | 0.000105 (+48/-95) | 37.8 | 0.632 (+103/-111) | 25.6 |
| v1b | 3e-5 | 2 | 0.0555 | 13.7 | 0.000692 (+56/-99) | 38.5 | 1 (+119/-120) | 26.1 |
| v1b | 3e-5 | 3 | 0.0540 | 13.5 | 0.000619 (+61/-106) | 36.9 | 0.327 (+125/-142) | 25.2 |
| v1b | 3e-5 | 4 | 0.0534 | 13.6 | 0.000736 (+60/-104) | 36.6 | 0.246 (+124/-144) | 25.1 |
| v1b | 3e-5 | 5 | 0.0530 | 13.7 | 0.00109 (+62/-105) | 36.4 | 0.199 (+123/-145) | 25.1 |
| v1b | 1e-4 | 1 | 0.0531 | 14.3 | 0.00492 (+64/-101) | 35.8 | 0.0951 (+117/-145) | 25.1 |
| v1b | 1e-4 | 2 | 0.0508 | 14.7 | 0.0159 (+72/-105) | 32.8 | 0.000662 (+112/-170) | 23.8 |
| v1b | 1e-4 | 3 | 0.0499 | 12.7 | 8.3e-05 (+62/-115) | 32.2 | 0.000111 (+102/-166) | 22.4 |
| v1b | 1e-4 | 4 | 0.0495 | 13.7 | 0.00172 (+69/-112) | 32.4 | 0.000216 (+106/-168) | 23.1 |
| v1b | 1e-4 | 5 | 0.0494 | 14.1 | 0.00557 (+75/-114) | 32.8 | 0.000525 (+107/-165) | 23.4 |

## Data ablation (paired, same lr / epoch / T)

- lr 1e-5 epoch 1 T=1.0: v1b 16.2 vs unfiltered 15.9; only-unfiltered-correct=13, only-v1b-correct=16, p=0.711
- lr 1e-5 epoch 1 T=0.1: v1b 37.8 vs unfiltered 38.9; only-unfiltered-correct=28, only-v1b-correct=17, p=0.135
- lr 1e-5 epoch 2 T=1.0: v1b 14.6 vs unfiltered 14.0; only-unfiltered-correct=16, only-v1b-correct=22, p=0.418
- lr 1e-5 epoch 2 T=0.1: v1b 37.5 vs unfiltered 36.6; only-unfiltered-correct=24, only-v1b-correct=33, p=0.289
- lr 1e-5 epoch 3 T=1.0: v1b 13.5 vs unfiltered 14.0; only-unfiltered-correct=23, only-v1b-correct=18, p=0.533
- lr 1e-5 epoch 3 T=0.1: v1b 37.2 vs unfiltered 37.9; only-unfiltered-correct=34, only-v1b-correct=27, p=0.443
- lr 1e-5 epoch 4 T=1.0: v1b 13.5 vs unfiltered 14.0; only-unfiltered-correct=18, only-v1b-correct=13, p=0.473
- lr 1e-5 epoch 4 T=0.1: v1b 38.1 vs unfiltered 37.2; only-unfiltered-correct=25, only-v1b-correct=34, p=0.298
- lr 1e-5 epoch 5 T=1.0: v1b 14.7 vs unfiltered 13.7; only-unfiltered-correct=11, only-v1b-correct=21, p=0.11
- lr 1e-5 epoch 5 T=0.1: v1b 38.1 vs unfiltered 37.7; only-unfiltered-correct=27, only-v1b-correct=31, p=0.694

## lr selection

{
 "rule": "best-epoch mean dev accuracy over T=1.0 and T=0.1; ties -> smaller lr",
 "data": "v1b",
 "steps": 512,
 "lr": "1e-5",
 "best_epoch": 1,
 "mean_accuracy": 0.27,
 "per_lr": {
  "1e-4": {
   "mean_accuracy": 0.2505,
   "best_epoch": 1
  },
  "1e-5": {
   "mean_accuracy": 0.27,
   "best_epoch": 1
  },
  "3e-5": {
   "mean_accuracy": 0.261,
   "best_epoch": 2
  }
 }
}
