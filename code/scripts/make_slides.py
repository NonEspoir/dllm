"""Slides for the presentation (R49): docs/slides.pptx, 16:9, from the outline in
docs/12-slides与demo.md (14 pages + one demo page). Numbers are copied from
docs/11-报告.md; figures come from docs/figures/ (scripts/make_figures.py).
Speaker notes carry what to say. No GPU needed.

  python scripts/make_slides.py
"""
import os

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
FIG = os.path.join(REPO, 'docs', 'figures')
OUT = os.path.join(REPO, 'docs', 'slides.pptx')

FONT = 'Microsoft YaHei'
MONO = 'Consolas'
INK, INK2, ACCENT = RGBColor(0x0b, 0x0b, 0x0b), RGBColor(0x52, 0x51, 0x4e), RGBColor(0x2a, 0x78, 0xd6)
GOOD, BAD, RULE = RGBColor(0x1b, 0xaf, 0x7a), RGBColor(0xe3, 0x49, 0x48), RGBColor(0xe4, 0xe3, 0xdf)
W, H = Inches(13.333), Inches(7.5)
MARGIN = Inches(0.6)


def set_font(run, size, bold=False, color=INK, mono=False):
  f = run.font
  f.size, f.bold, f.color.rgb = Pt(size), bold, color
  f.name = MONO if mono else FONT
  rpr = run._r.get_or_add_rPr()
  for tag in ('a:ea', 'a:cs'):  # East Asian glyphs ignore font.name
    el = rpr.find(qn(tag))
    if el is None:
      el = rpr.makeelement(qn(tag), {})
      rpr.append(el)
    el.set('typeface', MONO if mono else FONT)


def text(slide, x, y, w, h, lines, size=18, color=INK, bold=False, mono=False, spacing=1.15):
  """lines: str or list of str / (str, dict) ; a leading '- ' makes a bullet,
  '  - ' a sub-bullet."""
  box = slide.shapes.add_textbox(x, y, w, h)
  tf = box.text_frame
  tf.word_wrap = True
  tf.margin_left = tf.margin_right = Inches(0.05)
  if isinstance(lines, str):
    lines = [lines]
  for i, line in enumerate(lines):
    opts = {}
    if isinstance(line, tuple):
      line, opts = line
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.line_spacing = spacing
    lvl = 0
    if line.startswith('  - '):
      line, lvl = '◦ ' + line[4:], 1
    elif line.startswith('- '):
      line = '• ' + line[2:]
    if lvl:
      p.level = 1
    p.space_after = Pt(opts.get('after', 6))
    r = p.add_run()
    r.text = line
    set_font(r, opts.get('size', size - 2 * lvl), opts.get('bold', bold),
             opts.get('color', color), opts.get('mono', mono))
  return box


def new_slide(prs, title, n, sub=None):
  s = prs.slides.add_slide(prs.slide_layouts[6])
  bar = s.shapes.add_shape(1, 0, 0, Inches(0.18), H)
  bar.fill.solid(); bar.fill.fore_color.rgb = ACCENT; bar.line.fill.background()
  text(s, MARGIN, Inches(0.35), W - 2 * MARGIN, Inches(0.8), title, size=30, bold=True)
  if sub:
    text(s, MARGIN, Inches(1.05), W - 2 * MARGIN, Inches(0.5), sub, size=16, color=INK2)
  text(s, W - Inches(1.2), H - Inches(0.5), Inches(0.8), Inches(0.35), str(n), size=11, color=INK2)
  return s


def image(slide, name, x, y, w, h):
  """Fit docs/figures/<name>.png inside the box, keep aspect, centre it."""
  path = os.path.join(FIG, name + '.png')
  iw, ih = Image.open(path).size
  scale = min(w / iw, h / ih)
  pw, ph = int(iw * scale), int(ih * scale)
  slide.shapes.add_picture(path, x + (w - pw) // 2, y + (h - ph) // 2, Emu(pw), Emu(ph))


def table(slide, x, y, w, rows, col_w=None, size=14, row_h=0.42, bold_rows=(), header=True):
  shp = slide.shapes.add_table(len(rows), len(rows[0]), x, y, w, Inches(row_h * len(rows)))
  tb = shp.table
  if col_w:
    tot = sum(col_w)
    for j, cw in enumerate(col_w):
      tb.columns[j].width = int(w * cw / tot)
  for i, row in enumerate(rows):
    tb.rows[i].height = Inches(row_h)
    for j, val in enumerate(row):
      cell = tb.cell(i, j)
      cell.fill.solid()
      cell.fill.fore_color.rgb = RGBColor(0xf3, 0xf2, 0xee) if (header and i == 0) else RGBColor(0xff, 0xff, 0xff)
      cell.margin_left = cell.margin_right = Inches(0.08)
      tf = cell.text_frame
      tf.word_wrap = True
      r = tf.paragraphs[0].add_run()
      r.text = str(val)
      set_font(r, size, bold=(header and i == 0) or i in bold_rows,
               color=ACCENT if i in bold_rows else INK)
  return shp


def notes(slide, s):
  slide.notes_slide.notes_text_frame.text = s


def build():
  prs = Presentation()
  prs.slide_width, prs.slide_height = W, H
  CW = W - 2 * MARGIN
  n = 0

  # 1 conclusion
  n += 1
  s = new_slide(prs, '模仿没用，试错有用', n, '170M 扩散语言模型（MDLM）在 GSM8K 上的 SFT 与 RL')
  text(s, MARGIN, Inches(1.8), Inches(5.6), Inches(4.8), [
    ('GSM8K test（1319 题）', {'size': 16, 'color': INK2}),
    ('15.2 → 42.8', {'size': 44, 'bold': True, 'color': ACCENT, 'after': 2}),
    ('512 步，T=1：原始 → RL', {'size': 16, 'color': INK2, 'after': 16}),
    ('32.4 → 45.4', {'size': 32, 'bold': True, 'after': 2}),
    ('T=0.1（降温后的原始模型也比不过）', {'size': 16, 'color': INK2, 'after': 16}),
    ('- SFT：15.2 → 15.2，没有提升', {'size': 18}),
    ('- 论文设置（1024 步）：16.0 → 43.1', {'size': 18}),
  ])
  image(s, 'fig_test_main', Inches(6.4), Inches(1.7), Inches(6.4), Inches(5.0))
  notes(s, '一句话：模仿学习没有用，让模型自己试、按对错打分有用。RL 后 test 15.2 到 42.8，配对检验 p=7e-76。'
           '降温是免费的对照组，T=0.1 的原始模型 32.4，RL 后 45.4，所以提升不是解码技巧。'
           '如实说明：定稿先是轨迹法（27.4），后来 ELBO 对照臂在 dev 上更好，按事先写好的规则改定稿，test 被两个 RL 模型各用过一次。')

  # 2 task and model
  n += 1
  s = new_slide(prs, '任务与模型', n)
  text(s, MARGIN, Inches(1.4), Inches(6.0), Inches(5.6), [
    ('MDLM（掩码扩散语言模型）', {'bold': True, 'size': 20}),
    '- 170M 参数，在 TinyGSM（1185 万条 GPT-3.5 合成的 Python 解）上预训练',
    '- 生成：从 512 个 [MASK] 开始，K 步里逐步"揭开"；揭开后不再改写',
    '- 同一步揭开的 token 相互独立 → 步数越少误差越大',
    ('为什么选 MDLM（同仓库 6 个模型）', {'bold': True, 'size': 20}),
    '- 起点准确率够高（GRPO 要组内有对有错）',
    '- 采样策略的 log π 能精确算（轨迹法的前提）',
    '- 有可借鉴的 RL 文献（d1 等）',
  ], size=17)
  text(s, Inches(7.0), Inches(1.5), Inches(5.7), Inches(0.4), '输入题目，输出可执行的 Python：', size=15, color=INK2)
  text(s, Inches(7.0), Inches(1.95), Inches(5.7), Inches(4.0), [
    'def simple_math_problem() -> int:',
    "    '''",
    '    Stella collects stamps. ...',
    "    '''",
    '    first_pages = 10 * 5 * 30',
    '    rest_pages = 40 * 50',
    '    result = first_pages + rest_pages',
    '    return result',
  ], size=14, mono=True, spacing=1.0)
  text(s, Inches(7.0), Inches(5.5), Inches(5.7), Inches(1.0),
       '执行返回值 == 标准答案 → 对。评测与奖励用同一个判分器。', size=15, color=INK2)
  notes(s, '方向二从头预训练被否决的理由是预算：一次 170M 预训练按官方训练量要数百卡时。'
           'FLM / CANDI 起点只有 0.2–0.3%，S-FLM 是确定性 ODE、没有随机动作，都不适合做 RL。')

  # 3 starting point
  n += 1
  s = new_slide(prs, '起点：18% 与 38.6%', n, 'dev1000，同一个原始模型，只改采样温度')
  text(s, MARGIN, Inches(2.0), Inches(4.0), Inches(2.5), [
    ('18.0%', {'size': 60, 'bold': True, 'after': 0}), ('T = 1', {'size': 20, 'color': INK2})])
  text(s, Inches(4.6), Inches(2.0), Inches(4.0), Inches(2.5), [
    ('38.6%', {'size': 60, 'bold': True, 'color': ACCENT, 'after': 0}), ('T = 0.1', {'size': 20, 'color': INK2})])
  text(s, MARGIN, Inches(4.6), CW, Inches(2.5), [
    '- 只调温度，准确率翻倍 → 很多题"本来就会"，只是采样时会写坏',
    '- 降温修好的主要是"代码跑不起来"（运行时错误 383 → 138），推理错误基本没少',
    '- 所以 RL 必须和"降温后的 base"比，否则会把降温的效果算到 RL 头上',
    '- 复现论文设置（test，1024 步，T=1）：16.0%（论文 18.0），在噪声边缘',
  ], size=18)
  notes(s, '这一页给后面所有比较定了基准：每个模型都报 T=1 和 T=0.1 两个数，比较都是同一协议、逐题配对。')

  # 4 sampler
  n += 1
  s = new_slide(prs, '先让考试变快：新采样器', n, '原版每步对所有位置做 float64 采样再丢弃；新版只对被揭开的位置算')
  table(s, MARGIN, Inches(1.8), Inches(7.2), [
    ['dev200，256 步', '原版', '新版'],
    ['耗时（batch 64）', '153 s', '76 s'],
    ['峰值显存', '47 GB', '5 GB'],
    ['dev1000 准确率', '15.0%', '15.6%（p=0.72）'],
  ], col_w=[3, 2, 2.5], size=16, row_h=0.55)
  text(s, Inches(8.3), Inches(1.8), Inches(4.4), Inches(4.8), [
    ('三层等价验证', {'bold': True, 'size': 20}),
    '- ① 同一随机数状态下 16 组 token 逐位相同',
    '- ② dev1000 统计一致（配对 McNemar）',
    '- ③ 快 2 倍、显存 1/9',
    ('原版保留，用配置切换', {'size': 16, 'color': INK2}),
  ], size=17)
  text(s, MARGIN, Inches(4.4), Inches(7.4), Inches(2.6), [
    '- 另加 skip_unchanged：画布没变的序列不重算',
    '- 副作用：TF32 下 cuBLAS 按子 batch 大小换 kernel，结果同分布但不逐位相同',
    '  - 所以每个对比都另跑同协议的 base，逐题配对',
  ], size=17)
  notes(s, '提速只有 2 倍，低于按 4090 估的 3–4 倍：A800 的 FP64 强，原版的 float64 开销本来不大。这条记进了"已更正的判断"。'
           '逐字测试第一次 16 组全"失败"，原因是 torch.compile 的随机数生成和 eager 不同，不是采样器的 bug。')

  # 5 SFT
  n += 1
  s = new_slide(prs, 'SFT：准备了很好的数据，但没有用', n)
  text(s, MARGIN, Inches(1.4), Inches(5.0), Inches(5.6), [
    ('数据', {'bold': True, 'size': 20}),
    '- DeepSeek 把 GSM8K train 改写成 TinyGSM 风格 Python',
    '- 执行结果必须等于标准答案 + 语义检查 + 人工复核',
    '- 6232 条，估计残余错误约 0.3%',
    ('结果（dev1000）', {'bold': True, 'size': 20}),
    '- 3 个学习率 × 5 epoch：T=0.1 最好也只是持平',
    '- T=1 几乎全部显著更差',
    '- test：15.2 vs 15.2，对的题数完全相同（+47 / −47）',
  ], size=16)
  image(s, 'fig_sft', Inches(5.7), Inches(1.5), Inches(7.1), Inches(4.6))
  text(s, Inches(5.7), Inches(6.2), Inches(7.1), Inches(0.8),
       'loss 一直降，准确率不涨；lr 越大 loss 越低、准确率越差', size=16, color=INK2)
  notes(s, '数据过滤消融（6232 条过滤 vs 6464 条不过滤）10 组配对全不显著。'
           'SFT 冒烟时发现过一个 bug：分隔符应是字面量 \\n（反斜杠加 n），一开始用了真换行，val loss 20 步降 3 倍才暴露。')

  # 6 why SFT fails
  n += 1
  s = new_slide(prs, '为什么 SFT 没用：学的是写法，不是解题', n)
  table(s, MARGIN, Inches(1.5), Inches(7.4), [
    ['dev1000（SFT = lr 3e-5，epoch 2）', 'base', 'SFT'],
    ['T=0.1 docstring 用 """ 的题数', '105', '977'],
    ['T=0.1 语法错误', '2', '0'],
    ['T=0.1 对的题', '386', '385'],
    ['T=1 NameError', '343', '393'],
    ['T=1 语法错误', '7', '33'],
  ], col_w=[3.6, 1.2, 1.2], size=15, row_h=0.5)
  text(s, Inches(8.4), Inches(1.5), Inches(4.4), Inches(5.4), [
    ('读法', {'bold': True, 'size': 20}),
    '- 整片换成 DeepSeek 的写法风格',
    '- 低温下更整齐，高温下更容易写坏',
    '- T=0.1 对的题总数不变，只是换了一批（+120 / −119）',
    '- val loss 越低，准确率越差',
    ('模仿一个更强模型的输出，学到的首先是表面形式', {'size': 16, 'color': ACCENT}),
  ], size=17)
  notes(s, '推论：要提高"做对"，需要直接以对错为信号的训练，也就是 RL。')

  # 7 RL method
  n += 1
  s = new_slide(prs, 'RL 怎么做：GRPO', n, '同一道题采 8 个答案，自己跟自己比')
  steps = ['rollout\n32 题 × 8 个答案\n128 步，T=1', '打分\n执行正确 = 1\n能跑但错 = 0.1',
           '组内优势\n减去组内平均\n全对 / 全错组丢掉', 'log π\n轨迹法 或 ELBO', '更新\nPPO clip\n每批 4 次']
  bw, gap = Inches(2.2), Inches(0.33)
  for i, st in enumerate(steps):
    x = MARGIN + i * (bw + gap)
    box = s.shapes.add_shape(1, x, Inches(1.9), bw, Inches(1.6))
    box.fill.solid(); box.fill.fore_color.rgb = RGBColor(0xee, 0xf4, 0xfc); box.line.color.rgb = ACCENT
    tf = box.text_frame; tf.word_wrap = True
    for k, ln in enumerate(st.split('\n')):
      p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
      r = p.add_run(); r.text = ln
      set_font(r, 16 if k == 0 else 13, bold=(k == 0))
    if i < len(steps) - 1:
      text(s, x + bw, Inches(2.4), gap, Inches(0.5), '→', size=22, color=INK2)
  text(s, MARGIN, Inches(3.9), CW, Inches(3.2), [
    '- 奖励直接用评测的判分器：做对就是目标，没有奖励模型',
    '- DAPO 技巧：clip-higher（0.2 / 0.28）、动态采样、token 级归一化；去掉组内 std',
    '- 稳定性：梯度尖峰跳过、cosine 学习率衰减、log-prob 下降刹车',
    '- 选 checkpoint：只看 dev200；规则在看到结果之前写定',
  ], size=18)
  notes(s, '调试时发现学习率 1e-6 到 1e-5 太小，50 步权重只动 0.045%，模型几乎没变；正式跑改为 lr 3e-5、每批更新 4 次。')

  # 8 log-prob
  n += 1
  s = new_slide(prs, 'log π 怎么算：精确轨迹 vs ELBO 近似', n, 'π(y|x) 要对所有揭开顺序求和，算不出')
  table(s, MARGIN, Inches(1.6), CW, [
    ['', '轨迹法（原主线）', 'ELBO（原对照 → 定稿）'],
    ['算的是什么', '这一条采样轨迹的概率 π(τ)：每轮新揭开位置的 log p 之和', '答案全遮、题目可见，一次前向，各位置 log p 之和'],
    ['与采样策略一致吗', '一致，精确（每条抽 8 轮，无偏）', '不一致，是近似'],
    ['训练部分每步', '约 40 s', '约 8 s'],
    ['test 512 / T=1', '27.4', '42.8'],
    ['test 512 / T=0.1', '39.5', '45.4'],
  ], col_w=[2, 4.5, 4.5], size=15, row_h=0.62, bold_rows=(4, 5))
  text(s, MARGIN, Inches(5.85), CW, Inches(1.3), [
    ('预期：ELBO 更便宜但更差。结果相反：ELBO 好得多。', {'bold': True, 'color': ACCENT}),
    '机制只有推测（未核实）：轨迹法的多数动作在画布快写满时揭开，梯度接近 0；ELBO 直接训练全遮盖时的第一步决策。',
  ], size=17)
  notes(s, '轨迹法能精确算的前提：MDLM 揭开后不再改写，所以可以由（最终 token，揭开轮次）还原每一轮的画布；哪些位置被揭开与 θ 无关。'
           '证据：轨迹法里 PAD 占动作 68%，平均 log p ≈ −0.002。ELBO 常态梯度范数 20–150，轨迹法约 0.02。'
           '"快约 5 倍"只指训练部分；rollout 两者一样，整步约快 1.8 倍。')

  # 9 results
  n += 1
  s = new_slide(prs, '结果：做不做 RL、怎么算 log π，才是大差别', n, 'dev1000，各配方按事先定的规则选点')
  image(s, 'fig_dev200_curves', MARGIN, Inches(1.5), Inches(7.6), Inches(3.4))
  table(s, Inches(8.4), Inches(1.5), Inches(4.4), [
    ['配方', 'T=1', 'T=0.1'],
    ['base', '18.0', '38.6'],
    ['轨迹法（5 种配方）', '26–32', '46–47'],
    ['ELBO 150 步', '48.1', '51.3'],
    ['ELBO 400 步（定稿）', '50.0', '50.5'],
  ], col_w=[2.6, 1, 1], size=14, row_h=0.5, bold_rows=(4,))
  text(s, MARGIN, Inches(5.1), CW, Inches(2.0), [
    '- 轨迹法内部：奖励设计、加不加 KL，两两配对都不显著（p = 0.14–1.00）',
    '- 只换 log-prob 算法：同配置 150 步，ELBO 高 16.5pp（T=1）/ 4.9pp（T=0.1）；两次 ELBO 运行只差 0.55pp',
    '- 全部 RL 配方都显著好于 base（p < 0.001）',
  ], size=17)
  notes(s, 'dev200 单点噪声约 ±3pp，曲线的折返不要逐点解读。dev1000 的后 800 题没参与单个配方内的选点，结论在后 800 题上一样。')

  # 10 demo
  n += 1
  s = new_slide(prs, '样例：同一道题，原始 vs 定稿', n, 'dev 第 3 / 17 / 42 / 88 / 137 题（固定编号，不是挑的）：base 2/5，定稿 3/5')
  card_w = (CW - Inches(0.3)) // 2
  text(s, MARGIN, Inches(1.5), card_w, Inches(0.5), 'dev #42　base：运行时错误', size=16, bold=True, color=BAD)
  text(s, MARGIN, Inches(1.95), card_w, Inches(2.3), [
    'stamps_in_first_10_pages = 5 * 30',
    'stamps_in_rest_of__pages = (50 * 50) - (50 - 10)',
    'total_stamps = stamps_in_first_10_pages + ...',
    'result = stamps_total      # NameError',
  ], size=12, mono=True, spacing=1.0)
  text(s, MARGIN + card_w + Inches(0.3), Inches(1.5), card_w, Inches(0.5), 'dev #42　定稿：对（3500）', size=16, bold=True, color=GOOD)
  text(s, MARGIN + card_w + Inches(0.3), Inches(1.95), card_w, Inches(2.3), [
    'first_pages = 10 * 5 * 30',
    'rest_pages = 40 * 50',
    'result = first_pages + rest_pages',
    'return result',
  ], size=12, mono=True, spacing=1.0)
  text(s, MARGIN, Inches(3.7), CW, Inches(0.5), 'dev #3（$100 每天花 $8，一周后换 $5 钞票，账户还剩多少）：两个都错', size=16, bold=True, color=BAD)
  text(s, MARGIN, Inches(4.15), CW, Inches(1.0), [
    'base:  result = remaining_balance      # 44，没扣掉取走的钞票',
    'final: result = bills                  # 8，返回了钞票张数',
  ], size=12, mono=True, spacing=1.0)
  table(s, MARGIN, Inches(5.2), CW, [
    ['dev1000，T=1', '对', '能跑但算错', '运行时错误'],
    ['base', '180', '429', '383'],
    ['定稿', '500', '448', '46'],
  ], col_w=[3, 2, 2, 2], size=14, row_h=0.4, bold_rows=(2,))
  notes(s, '#42 正是 RL 收益的主要类型：base 思路接近但变量名写错；定稿写得更短、变量自洽。'
           '#3 两个都没读懂"剩下多少"——推理错误 RL 没修好。下方表格是全量统计：多对的 320 题几乎全部来自运行时错误减少（−337），"能跑但算错"没有减少。'
           '完整卡片在 runs/1103-demo-final.txt。')

  # 11 divergence
  n += 1
  s = new_slide(prs, 'RL 会崩，以及怎么救', n)
  image(s, 'fig_health_logp', MARGIN, Inches(1.4), Inches(6.4), Inches(4.2))
  text(s, Inches(7.2), Inches(1.4), Inches(5.6), Inches(5.6), [
    ('症状', {'bold': True, 'size': 20}),
    '- 二值奖励约第 160 步后：log-prob −0.07 → −1.7，输出变乱码（熵爆炸，不是熵塌缩）',
    ('试过的', {'bold': True, 'size': 20}),
    '- 格式分：只把发散推迟到第 174 步',
    '- KL + 学习率衰减 + 尖峰跳过：轨迹法稳定到 400 步',
    '- ELBO：不加 KL、只加衰减，也稳定到 400 步',
    ('踩的坑', {'bold': True, 'size': 20}),
    '- 换成 ELBO 后照搬 KL 系数：KL 量级大约 1000 倍，罚项是策略梯度的 10 倍，第 24 步停掉、去 KL 重跑',
  ], size=16)
  notes(s, '加了相对阈值刹车：最近 10 步 log-prob 均值低于起点的 2 倍就存 checkpoint 并停。'
           'ELBO 的 log-prob 是在全遮盖画布上算的，绝对值不能和轨迹法比，只看形状是否平稳。')

  # 12 SFT then RL
  n += 1
  s = new_slide(prs, '反直觉：先 SFT 再 RL 更差', n, 'dev1000，同一 RL 配方（轨迹法、格式分），只换起点')
  table(s, MARGIN, Inches(1.8), Inches(7.6), [
    ['起点', 'T=1', 'T=0.1'],
    ['base → RL', '31.6', '46.4'],
    ['SFT → RL', '25.9', '46.8'],
    ['配对检验', 'p = 0.0005', 'p = 0.86'],
  ], col_w=[3, 2, 2], size=18, row_h=0.6)
  text(s, Inches(8.6), Inches(1.8), Inches(4.2), Inches(5.0), [
    ('推测（未核实）', {'bold': True, 'size': 20}),
    '- SFT 把高温分布压窄、错误更多，RL 在 T=1 下补不回来',
    '- SFT 起点 log-prob 更低，更早走进不稳定区（第 138 步刹车）',
    '- 训练正确率每个窗口都低 2–6pp',
  ], size=17)
  text(s, MARGIN, Inches(4.6), Inches(7.6), Inches(2.0),
       '题目要的"base → SFT → RL"路线，在这里不比"base → RL"好。', size=20, color=ACCENT, bold=True)

  # 13 ceiling
  n += 1
  s = new_slide(prs, '能力上限：为什么到不了 90%', n)
  image(s, 'fig_vote', MARGIN, Inches(1.4), Inches(7.2), Inches(3.4))
  text(s, Inches(8.0), Inches(1.4), Inches(4.8), Inches(5.6), [
    '- 同一份 TinyGSM、同规模的 AR 模型也只有 54–63%',
    '- 一次写完 512 token，没有草稿、没有自查',
    '- RL 只修好了"跑不起来"，"能跑但算错"始终约 45%',
    '- 多数投票：同样采 8 次，RL 仍领先（55.9 vs 40.2）',
    '- pass@8 远高于投票：能力在，但只能靠多试取出来',
  ], size=17)
  text(s, MARGIN, Inches(5.0), Inches(7.2), Inches(1.6),
       '图中 RL 是轨迹法原定稿；ELBO 定稿单次 T=1 就有 50.0，已高于 base 投 8 次（40.2）；它自己的投票没做（成本原因）。',
       size=14, color=INK2)

  # 14 reflection
  n += 1
  s = new_slide(prs, '过程反思', n)
  text(s, MARGIN, Inches(1.4), Inches(6.0), Inches(5.6), [
    ('无效的尝试', {'bold': True, 'size': 20}),
    '- SFT 3 lr × 5 epoch；数据过滤消融',
    '- 格式奖励：信号变密，方向不变',
    '- RL 学习率太小：约 4 卡时白跑',
    ('工程上的坑', {'bold': True, 'size': 20}),
    '- SFT 分隔符：val loss 降得太快才暴露',
    '- 刹车代码变量名冲突：第 1 步崩 → 改训练循环必须用正式参数冒烟',
    '- 选点挑到噪声尖峰 → 三点平滑 / 取最后一步',
  ], size=16)
  text(s, Inches(7.0), Inches(1.4), Inches(5.8), Inches(5.6), [
    ('被更正的判断（共记 31 条）', {'bold': True, 'size': 20}),
    '- 采样器"快 3–4 倍" → 实测 2 倍',
    '- 好数据 + SFT 应该涨分 → 不涨',
    '- KL 轮变慢是尖峰造成的 → 不是',
    '- ELBO 更差 → 好得多',
    ('方法上', {'bold': True, 'size': 20}),
    '- test 只用于复现和最终评测；被两个 RL 模型各用过一次，两行都保留',
    '- 约 55 A800 卡时，总花费约 ¥300',
  ], size=16)

  # 15 next
  n += 1
  s = new_slide(prs, '下一步', n)
  text(s, MARGIN, Inches(1.5), CW, Inches(5.5), [
    ('弄清 ELBO 为什么更好', {'bold': True, 'size': 22}),
    '- 按揭开轮次拆分梯度贡献；轨迹法只保留早期轮次；混合两种估计量；多 seed 复现',
    ('"能跑但算错"这一块', {'bold': True, 'size': 22}),
    '- 过程奖励（中间值与标准解对上给分，SFT 数据里已有标注）；带草稿的输出格式',
    ('多样性', {'bold': True, 'size': 22}),
    '- 定稿对温度几乎不敏感：测它的 pass@k，确认有没有损失可探索性',
    ('数据与规模', {'bold': True, 'size': 22}),
    '- 多轮 RFT；混入 TinyGSM 原数据；用 Duo 验证"可改写 token"是否让 RL 更有效',
  ], size=18)
  notes(s, '没做的部分：ELBO 定稿的投票、pass@k、步数鲁棒性，以及多 seed 复现。原因是结果在最后一天出来、不再开 GPU。')

  prs.save(OUT)
  print('wrote', OUT, n, 'slides')


if __name__ == '__main__':
  build()
