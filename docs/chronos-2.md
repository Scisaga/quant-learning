# Chronos-2 落地方案（形态学 + 宏观/板块协变量 → 区间/阈值概率 → Qlib 评估/回测）

目标：基于 **Chronos-2** 把「形态学（形态/走势结构）+ 宏观/板块协变量」映射为两类可交易语义的输出：

1. **未来 N 日（N 可变）的收益/涨跌幅区间（分位数区间）**
2. **未来 N 日累计涨/跌超过阈值 a（幅度可变）的概率**

本文按 **基本概念 → 模型机制设计 → 信号层 → 策略层 → 工程实现与验收 → 风险点 → TODO** 组织，避免把“口径/指标/工程细节”混在一起。

---

## 1. 基本概念（先统一口径）

### 1.1 预测对象与标签（推荐：log 价格 / log-return）

设标的 $i$ 在交易日 $t$ 的价格为 $P_{i,t}$：

- **目标序列（建模对象）**：$y_{i,t}=\log(P_{i,t})$
- **未来 N 日 log-return（标签/事件定义核心）**：
  $$r_{i,t,N}=y_{i,t+N}-y_{i,t}=\log\left(\frac{P_{i,t+N}}{P_{i,t}}\right)$$
- **未来 N 日简单收益（若策略/报表用百分比）**：
  $$\Delta_{i,t,N}=\exp(r_{i,t,N})-1$$

阈值事件（“累计涨/跌超过阈值”）在 log 空间很好写：

- 上涨阈值 $a>0$：$\Delta_{i,t,N}\ge a \iff r_{i,t,N}\ge u$，其中 $u=\log(1+a)$
- 下跌阈值 $a>0$：$\Delta_{i,t,N}\le -a \iff r_{i,t,N}\le d$，其中 $d=\log(1-a)$

> 原因：阈值事件关于“累计变化”，log 空间把乘法变化变成加法阈值，工程上更稳定。

### 1.2 Chronos-2 输出是什么（分位数预测）

Chronos-2 面向多步 ahead 预测，输出每个 horizon $h$ 的分位数（quantile forecasts），例如：

$$Q_{0.1}(y_{t+h}),\ Q_{0.5}(y_{t+h}),\ Q_{0.9}(y_{t+h})$$

并支持设置更密的 `quantile_levels`（如 0.01…0.99）。模型卡/README 也标注了其对协变量与长度的支持（如最大 `context_length=8192`、最大 `prediction_length=1024`，以具体模型/实现为准）。同时它强调通过 **group attention** 做组内信息共享（用于协变量融合/同板块联动）。参考：[Chronos-2 README][hf-chronos2-readme]、[Chronos-2 论文][arxiv-chronos2]。

将“未来水平值”的分位数转成“未来 N 日变化（log-return）”分位数：

$$Q_q(r_{t,N}) \approx Q_q(y_{t+N}) - y_t$$

再转成简单收益分位数：

$$Q_q(\Delta_{t,N}) \approx \exp(Q_q(r_{t,N}))-1$$

> N 可变的边界：只要 $N \le$ 你一次预测的 `prediction_length`，就能直接从对应 horizon 取值。

### 1.3 用分位数反推阈值概率（CDF 近似）

需要：

- $p_{\text{up}}(N,a)=\Pr(\Delta_{t,N}\ge a)$
- $p_{\text{down}}(N,a)=\Pr(\Delta_{t,N}\le -a)$

工程上最简单可控的方法是“分位数网格 + 插值”：

1. 设置密集 `quantile_levels`，得到 $Q_q(r_{t,N})$ 网格
2. 找到满足 $Q_{q^*}(r_{t,N})\approx u=\log(1+a)$ 的 $q^*$
3. 近似：$p_{\text{up}}(N,a)\approx 1-q^*$

下跌同理：若 $Q_{q^\downarrow}(r_{t,N})\approx d=\log(1-a)$，则 $p_{\text{down}}(N,a)\approx q^\downarrow$。

### 1.4 Qlib 评估口径：信号层 IR vs 组合层 IR

在 Qlib 风格里，“IR”至少分两类：

- **信号层（排序能力）**：IC / RankIC 及 ICIR / RankICIR  
  需要“每个交易日、每只股票一个 score”，再做横截面相关。
- **组合层（策略表现）**：Portfolio IR 等  
  需要回测后基于收益序列（或超额收益）计算。

结论：**ICIR/RankICIR**回答“score 是否稳定可排序”；**Portfolio IR**回答“按该 score 交易是否产生可用风险调整收益”。两者必须同时验收。参考：[Qlib Recorder/Records][qlib-recorder]。

---

## 2. 模型机制设计（输入组织 + 泄漏约束）

### 2.1 Chronos-2 能力点（与本任务相关）

落地时关键是把任务表达成 Chronos-2 的统一接口：

- 统一支持 **univariate / multivariate / covariate-informed**
- 协变量区分 **past-only** 与 **known-future**（实值/类别）
- **group attention**：组内共享信息（可用于协变量融合、同板块多标的联动）

参考：[Chronos-2 README][hf-chronos2-readme]、[Chronos-2 论文][arxiv-chronos2]。

### 2.2 协变量设计：Past-only vs Known-future（防信息泄漏）

**Past-only（只能用到 t 时刻真实已知的信息）**：

- 板块指数/宽基指数：log 价格 / log-return（可多维）
- 宏观/财务/公告类：必须做 “as-of（真实可得时间）” 对齐  
  例：CPI 在公布前不可用；可做“公布后阶梯填充”，但从真实公布日开始生效
- 形态学特征（历史 K 线）：
  - 实值：波动率、趋势强度、回撤、峰谷/分形统计等
  - 类别：形态类别（上升/盘整/下跌）、regime 标签等（categorical covariate）

**Known-future（未来在今天就能确定的表）**：

- 交易日历：星期几、月末/季末、节假日等
- 可预知事件：财报披露日、解禁日、分红除权登记日（前提：数据确实可得且可验证）

#### 2.2.1 Regime 标签（market regime）是什么？

Regime 标签 = 用**仅基于过去可得信息**把“市场/板块/个股所处状态”粗分成有限个类别（categorical covariate），让模型知道“现在是哪种行情/波动环境”。

- 标签粒度（先选一个跑通闭环）：
  - 全市场 regime（所有股票共享一列）
  - 板块 regime（同板块共享）
  - 个股 regime（每只股票自己算）
- 常见 regime 维度（最常用、也最易做无泄漏）：
  - **趋势**：上行 / 震荡 / 下行
  - **波动**：高波动 / 中性 / 低波动
  - （可选）流动性：高 / 低；相关性/分散度：高 / 低
- 代表性的“用于判定 regime 的特征”示例（都用滚动窗口、past-only）：
  - 趋势：`mom_20`（近 20 日累计收益）、`ma_slope`（MA 斜率）、`ma20>ma60`、`ADX` 等
  - 波动：`vol_20`（近 20 日收益标准差）、`ATR`、`bb_width`，以及 `vol_20` 的分位点（例如 >80% 视为高波动）
  - 回撤：`dd_60 = 1 - P_t / max(P_{t-60..t})`
  - 流动性：滚动成交额/换手的分位点（例如 <20% 视为低流动）
- 落地建议（先简单、再迭代）：
  - 先做 `trend ∈ {up, range, down}` × `vol ∈ {high, low}` → 6 类，足够用于 PoC
  - 标签编码固定（例如 `0..5`），缺失/未知用 `UNK`（不要动态增删编码）

### 2.3 Group 组装方式（两种常用工程形态）

**方案 A：单标的组（先跑通闭环）**  
target 为该股 $y_{i,t}$，加 past-only/known-future 协变量。

- 优点：实现快、调试清晰
- 缺点：不直接利用“同板块个股共振”

**方案 B：板块内多标的组（更贴近联动/共振）**  
同板块一篮子股票作为一个 group 输入。

- 优点：更贴近“板块协同”
- 缺点：对齐/缺失更复杂；需控制组大小与样本覆盖

---

## 3. 信号层（score 构造与 IC 评估）

这一层的目标是：把 Chronos-2 的分位数/概率输出变成横截面可排序的 `score`，并用 Qlib 的 IC/ICIR 评估“排序能力是否稳定”。

### 3.1 最小字段集合（`N` / `a` 是参数）

- `N`：预测 horizon（未来 N 个**交易日**）
- `a`：事件阈值（涨跌幅阈值，例如 `3%`、`5%`），只影响 `p_up/p_down` 的后处理（不需要重跑模型）

对每个交易日、每个股票，至少产出（以 N 日为例）：

- `q10_N`, `q50_N`, `q90_N`：$\Delta_{t,N}$ 的分位数（或对应的 log-return 分位数）
- `interval_width_N = q90_N - q10_N`（区间宽度：越宽表示越不确定）
- `p_up_N_a`, `p_down_N_a`（用分位数网格插值反推）

### 3.2 score 设计（先都产出，再用评测挑）

Qlib 需要“可排序的标量 score”。建议同时产出三种，对照跑 IC：

1. **score1：中位数收益**（“预测会涨多少”）  
   $s^{(1)}_{i,t}=Q_{0.5}(r_{i,t,N})$
2. **score2：不确定性折扣**（“同样预测涨幅，越没把握越打折”）  
   $s^{(2)}_{i,t}=Q_{0.5}(r_{i,t,N})/(w_{i,t}+\epsilon)$
3. **score3：事件概率差**（“涨过阈值的概率 - 跌过阈值的概率”）  
   $s^{(3)}_{i,t}=p_{\text{up}}(N,a)-p_{\text{down}}(N,a)$

> 说明：score 用来做“排序”。区间/概率的“可解释可靠性”（coverage/reliability）是另一类指标，需要单独评估与校准，见 5.4。

### 3.3 信号层评测（Qlib：IC / ICIR）

- 产出两张表（索引 `(datetime, instrument)`）：
  - `pred_df`：列为 `score`（也可多列：`score1/2/3`）
  - `label_df`：列为 `label`（未来 N 日收益或 log-return，口径与 score 的 N 对齐）
- 用 `SigAnaRecord` 或等价流程评估：IC、RankIC、ICIR、RankICIR
- 建议的对照维度：
  - `N ∈ {3,5,10,20}`（不同 horizon 信噪比差异很大）
  - score1 vs score2 vs score3（看哪种更稳）
  - 按 regime 切片（例如高波动 vs 低波动），定位“在哪些环境有效/失效”

---

## 4. 策略层（门控、仓位与回测）

这一层把“信号”转成“可交易规则”。建议把策略逻辑写成显式的 `static_gate` + `dynamic_gate`，并在回测里统一验收成本、回撤与组合 IR。

### 4.1 Static 门控（是否允许交易）

Static 的目的：把“不可交易/数据不可信/事件风险不可控”的样本先筛掉，避免污染信号评测与回测。

- 流动性：近 20 日成交额均值 ≥ L
- 可交易性：非停牌、非 ST、非一字板等
- 数据完整性：context window 覆盖率 ≥ 95%
- 事件风险：重大事件日（财报/解禁等）是否允许交易（可配置）

### 4.2 Dynamic 门控（示例：用概率 + 不确定性控制）

先把字母翻译成直觉：

- `p_up(N,a)`：未来 N 日**涨幅 ≥ a** 的概率（胜率视角）
- `p_down(N,a)`：未来 N 日**跌幅 ≤ -a** 的概率（下行风险视角）
- `q50_N = Q_{0.5}(\Delta_{t,N})`：中位数预测收益（“大概能赚多少”）
- `q10_N = Q_{0.1}(\Delta_{t,N})`：左尾情景（“更坏时可能亏多少”）
- `w_N = q90_N - q10_N`：区间宽度（不确定性/把握度）
- `p0,w0,m0,q0,s0`：需要通过回测/验证集调参的阈值（不是模型常数）

示例规则（可按“仅多/多空”选择解释）：

- 多头触发（示例）：上涨概率高 + 不确定性小 + 预期收益为正
  $$p_{\text{up}}(N,a)\ge p_0 \ \land\ w_N \le w_0 \ \land\ q50_N \ge m_0$$
- 风控/空头触发（示例）：下行概率高 或 左尾很差
  $$p_{\text{down}}(N,a)\ge q_0 \ \lor\ q10_N \le -s_0$$

对应的最小伪代码：

```text
if not static_pass:
    do_nothing
else:
    long = (p_up >= p0) and (width <= w0) and (q50 >= m0)
    risk_off = (p_down >= q0) or (q10 <= -s0)
```

- 若只做多：`risk_off` 可以解释为“不买/减仓”
- 若做多空：`risk_off` 可解释为“减多 + 允许做空（或加对冲）”

### 4.3 回测建议（Qlib）

- horizon 对齐：`N` 既是预测 horizon，也应与标签定义/策略持有周期保持一致
- 成本敏感性：同一套规则分别跑“0 成本/含成本”，看组合 IR 与回撤的变化
- 先固定最小规则，再扩展：先只用 `score1`（或 `score3`）跑通，再逐步引入 `w_N`、事件日过滤等

---

## 5. 工程实现与验收（把闭环跑起来）

### 5.1 输入数据表 schema（对齐 predict_df 形态）

为与 `Chronos2Pipeline.predict_df(...)`/AutoGluon 接口对齐，建议统一两张表：

- `context_df`（历史可得）：`item_id, timestamp, target, cov_past_*...`
- `future_df`（未来已知，可为空）：`item_id, timestamp, cov_future_*...`

关键工程约束：

- **as-of 对齐**：宏观/公告/财务必须用真实可得时间
- **缺失处理**：对齐到交易日索引后，缺失填充策略统一（ffill/0/缺失标记）
- **类别协变量**：编码与未知值策略固定（例如 `UNK`）

### 5.2 推理与派生字段（Chronos-2）

PoC 阶段建议：

- `prediction_length = max(N_set)`，覆盖所有要取的 `N`
- `quantile_levels`：至少 0.1/0.5/0.9；做阈值概率建议密集（0.01…0.99）
- 从 `forecast_df` 派生：  
  - $Q_q(r_{t,N})$、$Q_q(\Delta_{t,N})$  
  - `p_up_N_a/p_down_N_a`（分位数插值；改变 `a` 不需要重跑模型）  
  - `score1/2/3`

两条接入路径：

- 直接用 `chronos-forecasting`（`Chronos2Pipeline.predict_df`）：[PyPI][pypi-chronos-forecasting] / [README][hf-chronos2-readme]
- 用 AutoGluon `TimeSeriesPredictor`（带 backtesting 脚手架，适合快速比较/微调）：[教程][autogluon-chronos2-tutorial]

### 5.3 对接 Qlib（信号评测 + 回测）

工程侧只要保证两张 MultiIndex 表（索引 `(datetime, instrument)`）：

- `pred_df`：至少 `score`（也可多列：`score1/2/3`）
- `label_df`：`label`（未来 N 日收益或 log-return，口径与 score 的 N 对齐）

复用 Qlib 的 record 体系：

- `SigAnaRecord`：IC / RankIC / ICIR / RankICIR（信号层）
- `PortAnaRecord`：回测报告（组合层：年化、回撤、换手、成本敏感性、Portfolio IR 等）

参考：[Qlib Recorder/Records][qlib-recorder]。

### 5.4 校准（M5）：怎么做、做完得到什么、用来干什么？

校准不改 Chronos-2 模型参数，是对输出做后处理，让：

- “名义 80% 区间”在样本外真的覆盖接近 80%
- “预测 0.6 的事件概率”在样本外也接近 60% 的发生频率（可靠性）

实现层面可参考 MAPIE 的 time-series 接口与相关 conformal 文献（[mapie-ts-regressor][], [arxiv-enbpi][], [arxiv-mapie][]）。

#### 5.4.1 区间校准（conformal 扩张；建议先做这个）

以预测对象 $r_{t,N}$（或 $\Delta_{t,N}$）为例，设原始区间为 $[L_t^{raw},U_t^{raw}]$（例如 `q10/q90`）：

1. 切分：`train`（产生预测）/ `calib`（只用于校准）/ `test`（最终评测），全程 walk-forward（禁止未来信息）
2. 在 `calib` 上逐日得到原始区间与真实值 $y_t$，计算非一致性分数：
   $$a_t=\max(L_t^{raw}-y_t,\ y_t-U_t^{raw},\ 0)$$
3. 设目标覆盖率为 $1-\alpha$（例如 0.8），在滚动窗口内取 $a_t$ 的 $(1-\alpha)$ 分位数 $q_a$
4. 得到校准后区间：
   $$L_t^{cal}=L_t^{raw}-q_a,\quad U_t^{cal}=U_t^{raw}+q_a$$
5. 对每个 horizon `N` 单独维护一套 $a_t$ 队列（不同 `N` 的误差分布不同）

产物（建议同时保留 raw 与 cal，便于对照与回溯）：

- `q10_N_raw, q90_N_raw, interval_width_N_raw`
- `q10_N_cal, q90_N_cal, interval_width_N_cal`

#### 5.4.2 概率校准（isotonic / Platt）

以 `p_up_N_a` 为例：

1. 在 `calib` 上构造事件 $E_t=\mathbf{1}[\Delta_{t,N}\ge a]$
2. 收集 `(p_up_raw, E)`，拟合单调映射 $g$，得到 `p_up_cal = g(p_up_raw)`  
   - isotonic：单调非参数（更灵活）  
   - Platt：sigmoid（更平滑、参数更少）
3. 在 `test` 上评估 reliability diagram / Brier，并与 raw 对照
4. 同理对 `p_down_N_a` 单独校准（通常需要两套映射）；必要时按 `(N,a)` 分开维护

产物：

- `p_up_N_a_raw, p_up_N_a_cal`
- `p_down_N_a_raw, p_down_N_a_cal`

#### 5.4.3 校准后能干什么用？（以及是不是“重新生成新号码”）

是的：**校准会生成一套新的“校准后字段”（`*_cal`）**，但它不需要重跑 Chronos-2；raw 字段建议保留。

校准后的主要用途：

- 让 `p0` 这类阈值有可解释语义（例如 `p_up_cal≥0.6` ≈ “历史上六成能达标”，而不是漂移分数）
- 用 `interval_width_N_cal` 做仓位折扣/风险预算（区间变宽就降仓）
- 用 `q10_N_cal`/`q90_N_cal` 作为下行风险约束（类似 VaR 的直觉用法），让“风控触发”更稳定

### 5.5 推荐的最小 PoC 配置（先验证“可用性”，再扩展）

建议先用很小配置跑通闭环：

- `N`（未来 N 个交易日的 horizon）：$N\in\{5,10\}$
- `a`（涨跌幅阈值，用于事件概率）：$a\in\{3\%,5\%\}$  
  说明：`N` 受 `prediction_length` 约束；`a` 只影响后处理，可在同一次预测结果上计算多个阈值
- `quantile_levels=0.01..0.99`
- 协变量：宽基/板块 + 2~3 个形态学数值（先不加复杂宏观/事件）

先把三件事验收：**信号排序能力（IC/ICIR） + 概率/区间可靠性（含校准） + 回测结果**。跑通后再逐步加宏观与 known-future 事件。

### 5.6 最小可验收输出（Definition of Done）

按“不是图好看，而是可验收”的标准：

- 逐日逐股产出：`q10/q50/q90`、`interval_width`、`p_up/p_down`、`score1/2/3`、`static_pass`、`signal_long/short`（建议 raw 与 cal 两套并存）
- 严格 walk-forward：训练/校准/测试时段不混用
- 三类评估报表齐全：
  - 信号层：IC/RankIC + ICIR/RankICIR（按 `N`、按 score 版本）
  - 校准层：覆盖率（名义 vs 实际）+ 区间宽度；概率可靠性（Brier/reliability；raw vs cal 对照）
  - 策略层：年化、回撤、换手、成本敏感性、Portfolio IR

---

## 6. 关键风险点

1. **信息泄漏**：宏观/公告/财务必须 as-of 对齐，否则回测必然虚高。
2. **分布漂移**：金融序列非平稳；需要滚动评估 + 校准层，否则区间/概率会失真。
3. **区间 ≠ 超额收益**：覆盖率好不代表可交易优势；必须把成本与风控一起回测。
4. **horizon 过长信噪比低**：先从短/中周期 N（如 3/5/10/20）分层评估再扩展。
5. **缺失与公司行为**：停牌、除权、事件日缺口会引入系统性偏差，需要规则化处理。

---

## 7. 具体 TODO（按“可验收里程碑”组织）

### M0：口径与配置（一次性定清）

- [ ] 标的池：`csi300` / 自定义池（含可交易性过滤规则）
- [ ] 标签：N 日简单收益 / log-return / 超额收益（与 score 一致）
- [ ] N 集合与阈值集合：如 $N\in\{3,5,10,20\}$、$a\in\{3\%,5\%\}$
- [ ] score 版本：score1/2/3 都保留，先对照评测

### M1：数据与协变量（保证无泄漏）

- [ ] 构建 `context_df`/`future_df`（交易日对齐、缺失策略统一）
- [ ] 宏观/公告/财务做 as-of 对齐（可回溯验证）
- [ ] 形态学特征：先做 2~3 个稳定数值特征（波动/趋势/回撤）
- [ ] regime 标签：先选粒度（全市场/板块/个股），做 `trend×vol` 6 类并固定编码（含 `UNK`）

### M2：推理与落库（Chronos-2 → forecast_df）

- [ ] `prediction_length=max(N_set)`，`quantile_levels=0.01..0.99`
- [ ] 产出 `forecast_df`（含分位数）并缓存（便于重复评测）
- [ ] 从分位数派生：`q10/q50/q90`、`p_up/p_down`、`interval_width`

### M3：构造 score & 信号评测（Qlib：IC/ICIR）

- [ ] 生成 `pred_df(score1/2/3)` + `label_df`
- [ ] 跑 `SigAnaRecord`（按 N、按 score 版本输出报表）
- [ ] 市场状态切片（震荡/趋势）下 IC 稳定性对照

### M4：回测（Qlib：PortAnaRecord）

- [ ] 选择最小策略（TopK/分组，多空或仅多）
- [ ] 成本敏感性分析（加/不加成本对 Portfolio IR 与回撤影响）

### M5：校准（覆盖率 + 概率可靠性）

- [ ] walk-forward 切分：`train` / `calib` / `test`（校准只用 `calib`）
- [ ] 区间校准：用 conformal 扩张（按每个 `N` 维护滚动窗口），产出 `q10_N_cal/q90_N_cal`
- [ ] 概率校准：isotonic/Platt（按 `(N,a)` 对 `p_up/p_down` 分别拟合），产出 `p_*_cal`
- [ ] 评测对照：coverage（名义 vs 实际）/ 区间宽度 / reliability / Brier（raw vs cal）

### M6：迭代与扩展（只做对 ICIR/回测 IR 有增量的）

- [ ] 协变量 ablation（按 ICIR/覆盖率/回测 IR 的边际贡献排序）
- [ ] group 方案对比：单标的组 vs 板块多标的组（控制组大小）

### M7：微调（可选，闭环跑通后再做）

- [ ] zero-shot vs LoRA/全量微调对照（严格 OOS）
- [ ] 若提升不显著：停止微调，把 Chronos 当“不确定性/风险刻画模块”

---

## 参考

[代码实现对照（本仓库）]

- 特征与 regime：`src/chronos/qchronos/features.py`
- 分位数后处理（区间/概率/score）：`src/chronos/qchronos/postprocess.py`
- 静态/动态门控：`src/chronos/qchronos/gating.py`
- 校准（区间 conformal / 概率 isotonic/Platt）：`src/chronos/qchronos/calibration.py`
- Qlib 导出（pred_df/label_df）：`src/chronos/qchronos/qlib_adapter.py`
- Chronos-2 推理包装（可选依赖）：`src/chronos/qchronos/chronos2_infer.py`
- 脚本与自检：`src/chronos/scripts/`

[hf-chronos2-readme]: https://huggingface.co/amazon/chronos-2/raw/main/README.md "amazon/chronos-2 README"
[arxiv-chronos2]: https://arxiv.org/abs/2510.15821 "Chronos-2: From Univariate to Universal Forecasting"
[pypi-chronos-forecasting]: https://pypi.org/project/chronos-forecasting/ "chronos-forecasting (PyPI)"
[autogluon-chronos2-tutorial]: https://auto.gluon.ai/dev/tutorials/timeseries/forecasting-chronos.html "AutoGluon TimeSeries: Chronos-2"
[qlib-recorder]: https://qlib.readthedocs.io/en/latest/component/recorder.html "Qlib Recorder / Records"
[mapie-ts-regressor]: https://mapie.readthedocs.io/en/latest/generated/mapie.regression.MapieTimeSeriesRegressor.html "MAPIE MapieTimeSeriesRegressor"
[arxiv-enbpi]: https://arxiv.org/abs/2010.09107 "Conformal prediction for time series"
[arxiv-mapie]: https://arxiv.org/abs/2207.12274 "MAPIE: distribution-free uncertainty quantification"
