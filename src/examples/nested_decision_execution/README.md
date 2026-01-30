# Nested Decision Execution（回测中的嵌套决策执行）

本示例演示 Qlib 在回测中对“嵌套决策执行（nested decision execution）”的支持：在不同频率下使用不同策略进行决策与执行。

典型场景是：

- 低频策略生成组合（portfolio generation，例如周频/日频）；
- 高频策略执行订单（order execution，例如日频/分钟频）。

## 场景 1：周频生成组合 + 日频执行

该流程示例使用：

- 周频：`DropoutTopkStrategy`（基于日频 LightGBM 模型的策略）生成周度组合；
- 日频：`SBBStrategyEMA`（基于 EMA 的规则策略）执行订单。

### 运行方式

回测：

```bash
python src/examples/nested_decision_execution/workflow.py backtest
```

收集/准备数据（若脚本提供该入口）：

```bash
python src/examples/nested_decision_execution/workflow.py collect_data
```

## 场景 2：日频生成组合 + 分钟级执行

该流程示例使用：

- 日频：`DropoutTopkStrategy` 生成组合；
- 分钟频：`SBBStrategyEMA` 执行订单。

### 运行方式

```bash
python src/examples/nested_decision_execution/workflow.py backtest_highfreq
```

## 备注

- 若你使用自定义数据目录，可通过环境变量 `PROVIDER_URI` 指向你的 Qlib 数据路径。
- 回测结果通常会记录到 `mlruns/`（MLflow），具体以脚本内的 recorder 配置为准。
