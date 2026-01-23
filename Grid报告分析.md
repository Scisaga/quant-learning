# Grid 报告分析（建议流程）

本文是对 `src/run_grid.py` 产出的 grid run（多市场 / 多 horizon / walk-forward 窗口 / 可选 PIT）的分析指南。

核心思路：**先做分组对比（静态汇总：年化收益 / 信息比率 / 最大回撤）**，再基于参数分组的结果去看 **指标随窗口滚动的动态趋势**，最后再做异常排查与（可选）PIT 细分。

---

## 1. 产物结构（你会用到哪些文件）

默认输出目录：`reports/grid_runs/<timestamp>/`
- `summary.json`：展开后的网格参数与 walk-forward 窗口
- `results.jsonl`：每个 job 一行（status/recorder_id/报告路径/日志路径等）
- `grid_compare.csv` / `grid_compare.xlsx`：**只包含成功（ok）的对比表**（推荐入口）
- `grid_errors.csv`：失败与异常汇总（train_failed/report_failed 等）
- `logs/*.train.log`：训练脚本 stdout+stderr
- `logs/*.report.log`：报告脚本 stdout+stderr

### 1.1 自动化分析工具（与本文同构）

推荐使用 `src/scripts/analyze_grid_run.py` 一键生成分析报告 `grid_run_analysis.md`。该脚本的章节结构与本文一致（数据概览 → 分组对比 → 动态趋势 → 异常排查 → PIT 附录），用于把流程固化，保证每次分析口径一致、可复现。

用法：
- `python src/scripts/analyze_grid_run.py --run-dir reports/grid_runs/<timestamp>`

> 可比性前提：固定交易策略与成本口径，只比较 `market/horizon/window`（以及你需要的话再比较 PIT）。
> - 特别注意：**label horizon 与交易策略最小持有期需要对齐**，否则结果会失真。
> - benchmark 建议“同 universe 对齐基准”（例如 csi300→SH000300，csi1000→SH000852，csiall→SH000985 或自定义全市场基准）。

---

## 2. 数据概览（先做一般性描述）

目标：从量化策略视角，在深入分组对比之前先快速回答：
- **有没有 alpha**（超额 IR/年化是否为正、是否可持续）
- **稳不稳**（跨窗口稳定性、最差窗口是否能接受）
- **能不能交易**（成本拖累、覆盖率/可交易集合是否足够）
- **能不能扩规模**（潜在容量约束：覆盖率偏低/集中度过高/换手导致成本敏感）

建议先从导出表（`grid_compare.csv`/`xlsx`，只含 ok job）提取以下信息：

### 2.1 数据有效性（研究样本是否可信）
- 本次总 job 数、ok 数、失败数（结合 `results.jsonl` / `grid_errors.csv`）
- ok 是否覆盖全窗口：每个 `market/horizon/pit` 的 ok 数是否均衡（避免“只在后半段窗口 ok”的选择偏差）
- 若某组窗口频繁失败：优先判定为数据/流程问题，而不是策略本身

### 2.2 策略可比性（同一把尺）
- `topk/n_drop/hold_thresh` 是否在本次对比中固定
- 成本参数（open/close/min/limit_threshold）是否固定
- benchmark 是否按 market 对齐（同 universe 基准）

### 2.3 策略“可交易性”快检（先把明显不靠谱的剔除）
- 覆盖率/可交易集合规模：`pred_instruments` 是否过低（过低会导致组合不稳、IC/IR 失真、成本敏感）
- 成本敏感性：对比 `without_cost` 与 `with_cost`（成本拖累过大，说明换手/冲击可能是主矛盾）
- 风险底线：查看最差窗口的 MaxDD/IR（策略最难受的时候是否能接受）

### 2.4 “哪个策略最好”（粗粒度导航答案）
在不展开全部窗口细节的前提下，先给一个用于导航的粗结论：
- 分别按 market（或全局）统计各参数组的 **IR_median / AnnRet_median / MaxDD_median**（with cost）
- 同时给出 **IR>0 比例**（稳定性），并标注“最差窗口”用于风险提示

> 注意：这里的“最好”只用于缩小候选集（选 1–3 组继续看趋势），不是最终结论；最终结论必须结合第 4 节的动态趋势与异常窗口。

---

## 3. 分组对比（先看静态汇总）

目标：回答“在同一套交易策略/成本下，哪个参数组合整体更好、更稳”。

### 3.1 先看三大核心指标（with cost 的超额口径）

以导出表中 *with cost* 的超额指标为主：
- **年化收益（Annualized Return）**：收益水平
- **信息比率（Information Ratio, IR）**：收益/波动比，稳定性更强
- **最大回撤（Max Drawdown）**：风险尾部

辅助指标（用于解释而非排名）：
- IC / RankIC：模型信号质量（“预测层”）
- pred_instruments（或 coverage 类字段）：覆盖率/可交易集合规模（“交易层”）

### 3.2 推荐分组维度与统计口径

建议先按下面维度聚合（例如中位数/均值）：
- `market × horizon`（主维度）
- 同一 `(market, horizon)` 内，再按 `window` 看分布/波动（稳定性）
- PIT 维度（`pit`/`pit_fields`）先放次要：除非你明确要回答“PIT 是否有增量”

输出建议（最终要能落到一句话结论）：
- 每个 `(market, horizon)` 的 IR/AnnRet/MaxDD 的 **中位数** + **IR>0 比例**
- 标出最差窗口（供下一步趋势与排查）

---

## 4. 动态趋势（再看随窗口滚动的变化）

目标：回答“这个分组为什么好/坏，是稳定优势还是只在某些年份有效（regime）？”

做法：
1) 从第 2 步选出你关心的几组（例如每个 market 选 1–2 个 horizon）
2) 按窗口时间顺序（通常以 `window.test_start..window.test_end`）展开
3) 对每个窗口记录 AnnRet/IR/MaxDD（必要时加 IC/coverage）
4) 观察：
   - 是否存在持续走强/走弱（趋势）
   - 是否存在特定年份突然翻转（regime change）
   - 大回撤窗口是否集中在某个阶段（风险集中）

常见解读：
- IR 波动很大：多半是覆盖率/可交易集合不稳定、或策略成本拖累在某些阶段更强
- AnnRet 高但 IR 低：收益来自少数窗口/高波动，稳定性不足
- IC 稳定但组合指标差：可能是“交易层/成本/基准/换手”把 alpha 吃掉了

---

## 5. 异常与失败（最后再排查）

### 5.1 train_failed
- 看对应 `logs/<tag>.train.log`
- 常见原因：数据缺失、特征 bin 不齐、依赖缺失、参数不匹配

### 5.2 report_failed
- 看对应 `logs/<tag>.report.log`
- 常见原因：预测信号为空、回测区间无数据、交易日对齐问题、覆盖率过低导致图表阶段报错

### 5.3 “看起来成功但指标异常”
优先检查：
- coverage / pred_instruments 是否过低（过低会导致策略结果不稳定）
- `Mean of empty slice` 之类告警：更常见是“某些日子可交易集合为空/个股休市/信号不覆盖”，不一定要直接判失败，但应把该窗口标记为低质量并谨慎解读

---

## 6. PIT 对比

建议口径：
- 固定 `(market, horizon, window)`，比较 `no_pit` vs `pit_all`（或单字段）
- 同时检查 coverage 是否发生变化（否则可能是“比较对象的 universe 不一致”）

重要提醒：
- PIT 特征如果对 universe 覆盖不齐，会造成窗口间可比样本不一致；这种情况下结论要谨慎，必要时先解决数据覆盖或统一过滤规则。

---

## 7. 结论表达模板（建议）

按 market 分段写，先静态、再趋势、最后异常：
- `(market=..., horizon=...)` 在多数窗口上 IR/AnnRet/MaxDD 表现最好（给出 median + IR>0 比例）
- 动态趋势：在哪些窗口显著变好/变差（列出 2–3 个关键窗口）
- 异常：哪些窗口因覆盖率/报错/数据问题需要剔除或单独说明
