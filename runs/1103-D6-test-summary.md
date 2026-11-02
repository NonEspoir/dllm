# D6 final evaluation (1103)

## test set, three models x three settings

| model | steps | T | accuracy | 95% CI | n | time |
|---|---|---|---|---|---|---|
| rl | 512 | 1.0 | 27.4 | [25.1, 29.7] | 1319 | 427 s |
| rl | 512 | 0.1 | 39.5 | [36.9, 42.1] | 1319 | 424 s |
| rl | 1024 | 1.0 | 29.4 | [27.0, 31.7] | 1319 | 523 s |
| base | 512 | 1.0 | 15.2 | [13.3, 17.2] | 1319 | 425 s |
| base | 512 | 0.1 | 32.4 | [29.9, 34.9] | 1319 | 425 s |
| base | 1024 | 1.0 | 16.0 | [14.1, 18.0] | 1319 | 531 s |
| sft | 512 | 1.0 | 15.2 | [13.2, 17.1] | 1319 | 429 s |
| sft | 512 | 0.1 | 33.5 | [31.0, 36.2] | 1319 | 423 s |
| sft | 1024 | 1.0 | 15.3 | [13.4, 17.3] | 1319 | 528 s |

## final model on dev: step robustness (T=1)

| steps | accuracy |
|---|---|
| 128 | 27.3 |
| 256 | 31.9 |
| 512 | 30.8 |
| 1024 | 33.0 |

## majority vote / pass@k on dev (8 different seeds)

### T=1.0
```
<sample>:13: SyntaxWarning: 'int' object is not callable; perhaps you missed a comma?
<sample>:10: SyntaxWarning: 'int' object is not callable; perhaps you missed a comma?
<sample>:9: SyntaxWarning: 'int' object is not callable; perhaps you missed a comma?
<sample>:7: SyntaxWarning: 'int' object is not subscriptable; perhaps you missed a comma?
<sample>:11: SyntaxWarning: 'float' object is not callable; perhaps you missed a comma?
<sample>:10: SyntaxWarning: 'int' object is not callable; perhaps you missed a comma?
<sample>:14: SyntaxWarning: invalid decimal literal
<sample>:6: SyntaxWarning: 'int' object is not callable; perhaps you missed a comma?
<sample>:13: SyntaxWarning: 'int' object is not callable; perhaps you missed a comma?
8 samples x 1000 questions
k=1: majority 30.8%   pass@1 30.8%
k=2: majority 38.7%   pass@2 45.7%
k=3: majority 44.2%   pass@3 53.7%
k=4: majority 48.0%   pass@4 58.9%
k=5: majority 50.9%   pass@5 62.7%
k=6: majority 53.5%   pass@6 65.8%
k=7: majority 54.9%   pass@7 67.9%
k=8: majority 55.9%   pass@8 70.0%
questions with no usable answer in any sample: 5 (0.5%)
```

### T=0.1
```
8 samples x 1000 questions
k=1: majority 46.3%   pass@1 46.3%
k=2: majority 50.5%   pass@2 62.4%
k=3: majority 55.9%   pass@3 68.6%
k=4: majority 58.6%   pass@4 72.4%
k=5: majority 60.9%   pass@5 74.4%
k=6: majority 62.1%   pass@6 76.7%
k=7: majority 63.6%   pass@7 77.6%
k=8: majority 64.6%   pass@8 78.8%
questions with no usable answer in any sample: 0 (0.0%)
```
