# 周期性滚动重训（RR, Rolling Retrain）基线

RR（周期性滚动重训）是 concept drift/非平稳市场中最常见的基线：用固定频率“重训 → 预测下一段 → 再重训”，以持续利用最新数据来减缓模型在未来数据上的性能衰减。

本目录提供一个可直接运行的 RR 示例，入口脚本为 `rolling_benchmark.py`，实现基于 Qlib 的 `qlib.contrib.rolling.base.Rolling`（离线滚动任务生成、训练、拼接与统一评估）。

## 目录内容

- `rolling_benchmark.py`：RR 的 CLI 入口（默认自动下载 CN 公共数据；并调用 `Rolling.run()` 执行滚动训练与评估）。
- `workflow_config_linear_Alpha158.yaml`：LinearModel + Alpha158 的 RR 配置。
- `workflow_config_lightgbm_Alpha158.yaml`：LightGBM + Alpha158 的 RR 配置。

## RR 框架具体怎么做（实现细节）

RR 在本示例中的“滚动”由 `qlib.contrib.rolling.base.Rolling` 完成，核心流程如下（可对照 `qlib/contrib/rolling/base.py` 与 `qlib/workflow/task/gen.py`）：

1. 读取 YAML 中的 `task:`（注意：`Rolling` 只读取 `task`，不会使用 YAML 顶部的 `qlib_init` 段；初始化由 `rolling_benchmark.py` 内部的 `auto_init()` 完成）。
2. 生成一组“滚动子任务”（rolling tasks）：
   - 将 `segments.test` 拆成多个连续的窗口，每个窗口长度为 `step` 个交易日（默认 `step=20`）。
   - `train` 使用 expanding window（训练集起点固定，终点随时间向后扩展），`valid/test` 使用 sliding window（窗口整体向后平移）。
   - 为避免标签泄露，`train/valid` 会相对每个窗口的 `test_start` 截断 `trunc_days=horizon+1` 天（本示例默认 `horizon=20`，即截断 21 天；原因是标签形如 `Ref($close, -(horizon+1))/Ref($close, -1)-1`，需要未来 `horizon+1` 天信息才能完整计算）。
3. 依次训练每个窗口对应的模型，并仅记录该窗口的 `pred/label`（每个子任务只挂 `SignalRecord`，其它分析延后到“拼接后”再做）。
4. 使用 `RollingEnsemble` 将所有窗口的 `pred/label` 按时间拼接成完整测试期的序列（伪“在线”输出）。
5. 在拼接后的 `pred.pkl/label.pkl` 上运行配置里原本的评估记录器（如 `SigAnaRecord`、`PortAnaRecord`），得到整段测试期的 IC/回测结果。

运行产生的记录与结果默认落在 `mlruns/`（MLflow）。你也可以通过 `--rolling_exp`（窗口级实验名）与 `--exp_name`（最终拼接评估实验名）显式命名实验，便于多次对比。

## 运行方式

在本目录下运行：

```bash
python rolling_benchmark.py run
```

默认使用 `workflow_config_linear_Alpha158.yaml`（线性模型）。切换为 LightGBM：

```bash
python rolling_benchmark.py --conf_path=workflow_config_lightgbm_Alpha158.yaml run
```

## 常用参数

`rolling_benchmark.py` 通过 `fire` 暴露参数（对应 `Rolling.__init__`），常用的有：

- `--conf_path`：选择 workflow 配置 YAML（决定模型/数据集/回测设置）。
- `--step`：滚动步长（也可理解为“每隔多少交易日重训一次”）。
- `--horizon`：预测/标签周期（会影响标签表达式与截断天数）。
- `--exp_name`：最终“拼接后评估”的实验名（便于在 `mlruns/` 中定位结果）。
- `--rolling_exp`：每个窗口子任务所在实验名（若复用同名实验，可能需要先清理旧的 `mlruns/` 记录以避免 MLflow 的 `.trash` 冲突）。

## 数据与环境变量

- 默认数据：若未设置环境变量 `PROVIDER_URI`，脚本会尝试自动下载并使用 CN 公共数据（`~/.qlib/qlib_data/cn_data`）。
- 自定义数据：可通过设置 `PROVIDER_URI` 指向你的 Qlib 数据目录来复用已有数据，例如（PowerShell）：
  - `setx PROVIDER_URI "C:\\path\\to\\qlib_data\\cn_data"`

