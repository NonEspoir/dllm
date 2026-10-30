"""Reward for RL rollouts (D11; format reward is an ablation switch, R36).

grade() mirrors sandbox_gsm8k.evaluate_samples step by step (same code
extraction, restricted exec, 5 s limit, number parsing, comparison) but also
says *why* an answer failed. grade(...) == 'correct' exactly when
evaluate_samples(...) is True (checked in tests/test_rl_reward.py on eval
generations).

Categories:
  correct    runs and returns / prints the gold number
  wrong      runs and returns / prints a number, not the gold one
  no_number  runs, but neither the return value nor stdout gives a number
  runtime    exception or timeout
  syntax     the extracted code does not compile
  no_fn      no simple_math_problem after exec

reward(): 1 for correct; format_reward (default 0 = binary, the main line)
for 'wrong' (code that runs and produces a number); 0 otherwise.
"""
import contextlib
import io

import sandbox_gsm8k as sb


def grade(sample, answer, timeout_s=5.0):
  code = sb._extract_code(sample)
  try:
    compile(code, '<sample>', 'exec')
  except (SyntaxError, ValueError):  # ValueError: e.g. null bytes; upstream also scores 0
    return 'syntax'
  try:
    with sb._time_limit(timeout_s):
      ns = sb._safe_exec_no_timer(code)
      fn = ns.get('simple_math_problem', None)
      if fn is None:
        return 'no_fn'
      stdout = io.StringIO()
      with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
        out = fn()
  except (sb._Timeout, Exception):
    return 'runtime'
  gold = sb._extract_gold_answer(answer)
  out = sb._to_number(out)
  if out is None:
    out = sb._extract_text_answer(stdout.getvalue())
  if out is None:
    return 'no_number'
  return 'correct' if sb._numbers_equal(out, gold) else 'wrong'


def reward(category, format_reward=0.0):
  if category == 'correct':
    return 1.0
  if category == 'wrong':
    return float(format_reward)
  return 0.0
