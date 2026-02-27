# 订单执行的强化学习示例（rl_order_execution）

本目录提供一个订单执行场景的强化学习（RL）端到端示例，包含：

- 数据准备与预处理；
- 训练流程（training workflow）；
- 回测流程（backtest workflow）。

## 数据处理

### 1) 下载 5min 数据

建议先进入本目录再执行命令，确保数据落在本目录下的 `data/`：

```bash
cd backend/qlib/examples/rl_order_execution
python -m qlib.cli.data qlib_data --target_dir ./data/bin --region hs300 --interval 5min
```

### 2) 生成 pickle 格式数据

本示例使用 pickle 格式的数据集。运行以下脚本生成（可能需要几分钟）：

```bash
cd backend/qlib/examples/rl_order_execution
python scripts/gen_pickle_data.py -c scripts/pickle_data_config.yml
python scripts/gen_training_orders.py
python scripts/merge_orders.py
```

完成后 `data/` 目录结构应类似：

```text
data
├── bin
├── orders
└── pickle
```

## 训练（Training）

每个训练任务由一个配置文件定义：任务 `TASKNAME` 对应 `exp_configs/train_TASKNAME.yml`。

本示例提供两类任务：

- **PPO**：IJCAL 2020 论文 *An End-to-End Optimal Trade Execution Framework based on Proximal Policy Optimization*。
- **OPDS**：AAAI 2021 论文 *Universal Trading for Order Execution with Oracle Policy Distillation*。

二者主要差异在 reward 定义（细节见各自 config）。

以 OPDS 为例，训练命令如下（包含训练后直接跑一次 backtest）：

```bash
cd backend/qlib/examples/rl_order_execution
python -m qlib.rl.contrib.train_onpolicy --config_path exp_configs/train_opds.yml --run_backtest
```

训练日志、指标与 checkpoint 默认写入 `outputs/opds`（由 `exp_configs/train_opds.yml` 配置）。

## 回测（Backtest）

训练完成后，最新 checkpoint 通常位于 `outputs/opds/checkpoints/latest.pth`。要跑回测：

1. 在训练配置中设置 checkpoint 路径（默认是注释掉的；不设置会导致随机初始化模型，结果无意义）。
2. 运行回测：

```bash
cd backend/qlib/examples/rl_order_execution
python -m qlib.rl.contrib.backtest --config_path exp_configs/backtest_opds.yml
```

回测结果默认写入 `outputs/checkpoints/backtest_result.csv`（以配置为准）。

此外，本示例还提供了 TWAP（Time-weighted average price）作为弱基线：`exp_configs/backtest_twap.yml`。

## 训练测试 vs. 回测结果差异（重要）

训练流程里的 testing 与回测流程使用的 simulator 不同，因此结果可能不一致：

- 训练时常用 `SingleAssetOrderExecutionSimple`（更高效，但不限制成交量/最小成交单位等）。
- 回测时使用更真实的 `SingleAssetOrderExecution`（考虑实际约束，因此可能出现“实际执行量 ≠ 预期执行量”）。

若你希望得到与训练测试完全一致的结果，可仅跑训练 pipeline 的 backtest 阶段：

- 在训练 config 中指定 `weight_file` 指向 checkpoint；
- 执行：`python -m qlib.rl.contrib.train_onpolicy --config_path PATH/TO/CONFIG --run_backtest --no_training`

示例片段：

```yaml
policy:
  class: PPO  # PPO, DQN
  kwargs:
    lr: 0.0001
    weight_file: PATH/TO/CHECKPOINT
  module_path: qlib.rl.order_execution.policy
```

## Benchmarks（TBD）

RL 训练耗时较长，理想评估方式是多次实验取均值。若资源受限，可通过“选取验证集表现最好的若干 checkpoint”来近似多次实验。本示例以 Price Advantage（PA）选择 Top-10 checkpoint 做平均，结果如下：

| Model | PA mean with std. |
|---|---|
| OPDS (with PPO policy) | 0.4785 ± 0.7815 |
| OPDS (with DQN policy) | -0.0114 ± 0.5780 |
| PPO | -1.0935 ± 0.0922 |
| TWAP | ≈ 0.0 ± 0.0 |

> 说明：TWAP 理论上 PA 应为 0。本示例将订单执行拆成“半小时等分 + 每 5 分钟等分”，而日内最后 5 分钟禁止交易，因此会与传统“全日 TWAP”等分略有差异，PA 可能是一个接近 0 的数值。你可以运行 TWAP 回测验证。
