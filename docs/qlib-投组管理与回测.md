## 投组管理 Portfolio Management

Portfolio Strategy 是 Qlib 的组合管理组件，用于基于预测分数（score）生成投资决策，继承基类自定义策略，支持 qrun 自动工作流或独立使用，与 Forecast Model/Backtest 集成评估性能。

- **核心功能**：基于 Forecast Model 的预测分数，采用算法生成投资组合（portfolio）。可集成 Workflow（详见 Workflow: Workflow Management），或独立模块。
- **设计解读**：Qlib 组件松耦合，策略独立于模型/执行器。提供内置策略（如 TopkDropout），支持自定义。
- **优势**：回测（backtest）评估策略性能；分数含义依模型标签（越高越盈利）。
- **适用场景**：组合优化、风险控制；与 SignalRecord 结合记录信号。

### 基类与接口
- **BaseStrategy**：所有策略继承 qlib.strategy.base.BaseStrategy。
  - **关键接口**：generate_trade_decision(trade_date, current_pos, trade_exchange) → TradeDecision。
    - 每个交易 bar 调用（依执行器频率，如每日）。返回空决策（EmptyTradeDecision）可控制交易频率（如周频）。
- **WeightStrategyBase**：BaseStrategy 子类，专注目标仓位（target positions），自动生成订单。
  - **关键接口**：generate_target_weight_position() → dict（{instrument: weight}，权重为总资产百分比，不含现金）。
  - **过程**：目标仓位 → 目标数量 → 订单列表（generate_order_list）。
- **解读**：BaseStrategy 灵活控制决策；WeightStrategyBase 简化仓位管理。参考图示理解 Topk-Drop 算法。

### 实现策略

![Topk-Drop algorithm](img/qlib-1762376264234.png)

- **TopkDropoutStrategy**：继承 BaseStrategy，实现 generate_order_list。
  - **过程**：
    - Topk-Drop 算法：持仓 Topk 只，按分数排名卖出最差 Drop 只，买入最佳未持仓。
    - 参数：topk（持仓数）、n_drop（Drop，每日卖出数）。
    - 换手率：约 2 * Drop / Topk。
  - 生成订单：目标数量 → 订单列表。
- **EnhancedIndexingStrategy**：增强指数策略，结合主动/被动管理，超基准（如 S&P 500）回报，控制跟踪误差。
  - 参考：qlib.contrib.strategy.signal_strategy.EnhancedIndexingStrategy / EnhancedIndexingOptimizer。
- **落地提示**：继承 WeightStrategyBase 只需实现 generate_target_weight_position；自定义算法如优化换手。

### 使用与示例
- **预测分数（Prediction Score）**：pandas DataFrame，index 为 (datetime, instrument)，含 'score' 列。示例：
  ```
  datetime   instrument     score
  2019-01-04 SH600000   -0.505488
  ...        ...         ...
  2019-04-30 SZ300760   -0.126383
  ```
  - 来源：Forecast Model 输出（详见 Forecast Model）。分数规模依模型标签，可能需缩放（e.g., 回归拟合到回报）。
- **运行回测（Running backtest）**：
  - **简单版**（backtest_daily）：
    ```python
    from pprint import pprint
    import qlib
    import pandas as pd
    from qlib.utils.time import Freq
    from qlib.utils import flatten_dict
    from qlib.contrib.evaluate import backtest_daily
    from qlib.contrib.evaluate import risk_analysis
    from qlib.contrib.strategy import TopkDropoutStrategy

    qlib.init(provider_uri="<qlib data dir>")

    CSI300_BENCH = "SH000300"
    STRATEGY_CONFIG = {
        "topk": 50,
        "n_drop": 5,
        "signal": pred_score,  # 预测分数
    }

    strategy_obj = TopkDropoutStrategy(**STRATEGY_CONFIG)
    report_normal, positions_normal = backtest_daily(
        start_time="2017-01-01", end_time="2020-08-01", strategy=strategy_obj
    )
    analysis = {}
    analysis["excess_return_without_cost"] = risk_analysis(report_normal["return"] - report_normal["bench"])
    analysis["excess_return_with_cost"] = risk_analysis(report_normal["return"] - report_normal["bench"] - report_normal["cost"])

    analysis_df = pd.concat(analysis)
    pprint(analysis_df)
    ```
  - **高级版**（控制执行器）：
    ```python
    # ... 同上导入
    FREQ = "day"
    EXECUTOR_CONFIG = {
        "time_per_step": "day",
        "generate_portfolio_metrics": True,
    }
    backtest_config = {
        "start_time": "2017-01-01",
        "end_time": "2020-08-01",
        "account": 100000000,
        "benchmark": CSI300_BENCH,
        "exchange_kwargs": {
            "freq": FREQ,
            "limit_threshold": 0.095,
            "deal_price": "close",
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
        },
    }

    strategy_obj = TopkDropoutStrategy(**STRATEGY_CONFIG)
    executor_obj = executor.SimulatorExecutor(**EXECUTOR_CONFIG)
    portfolio_metric_dict, indicator_dict = backtest(executor=executor_obj, strategy=strategy_obj, **backtest_config)
    analysis_freq = "{0}{1}".format(*Freq.parse(FREQ))
    report_normal, positions_normal = portfolio_metric_dict.get(analysis_freq)

    # 分析（excess_return 等）
    analysis = {}
    analysis["excess_return_without_cost"] = risk_analysis(report_normal["return"] - report_normal["bench"], freq=analysis_freq)
    analysis["excess_return_with_cost"] = risk_analysis(report_normal["return"] - report_normal["bench"] - report_normal["cost"], freq=analysis_freq)
    analysis_df = pd.concat(analysis)
    pprint(analysis_df)
    ```
- **解读**：benchmark 用于超额回报计算（单仪器格式，如 SH000300）；market 是仪器池（如 csi300）。回测输出：report（回报/基准/费用）、positions（仓位）。

### 结果分析
- **示例输出**：
  ```
                                                  risk
  excess_return_without_cost mean               0.000605
                             std                0.005481
                             annualized_return  0.152373
                             information_ratio  1.751319
                             max_drawdown      -0.059055
  excess_return_with_cost    mean               0.000410
                             std                0.005478
                             annualized_return  0.103265
                             information_ratio  1.187411
                             max_drawdown      -0.075024
  ```
- **指标含义**：
  - excess_return_without_cost/with_cost：无/有费用超额回报（CAR）。
    - mean/std：均值/标准差。
    - annualized_return：年化回报。
    - information_ratio：信息比率（IR = 年化回报 / 标准差）。
    - max_drawdown：最大回撤（MDD）。
- **落地提示**：用 risk_analysis 评估；扁平化 dict 便于日志。

### 小结
- 内置 TopkDropout/增强指数，易自定义；回测标准化评估（IR/MDD）。
- 自动化订单生成/风险分析，减少错误；松耦合，便于换模型/策略。
- 与 Forecast Model（分数输入）、Workflow（qrun 集成）、Task Management（批量）闭环。
