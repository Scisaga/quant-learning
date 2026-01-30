# Examples（示例与教程）

本目录收录 Qlib 的常用示例脚本与 Notebook，覆盖从快速上手、基准模型复现、滚动训练、在线更新、组合优化，到高频/订单簿/RL 等主题。各子目录通常都有更详细的 `README.md`，本文作为总览与导航。

## 运行前准备

- 安装与基础配置：请先阅读 [`docs/qlib-安装配置.md`](../../docs/qlib-安装配置.md)；并可结合 [`docs/qlib-工作流.md`](../../docs/qlib-工作流.md) 理解 workflow 的整体结构。
- 数据准备：多数示例默认使用中国市场数据（`REG_CN`），常见数据目录为 `~/.qlib/qlib_data/cn_data`。
  - `examples/workflow_by_code.py` 会调用 `qlib.tests.data.GetData().qlib_data(...)` 自动下载所需数据（若已存在会跳过）。
  - 其他示例请按各子目录 `README.md` 的说明准备数据（可能包含 1min/5min 高频数据、权重数据、MongoDB 等依赖）。
- 最小硬件建议（用于 `workflow_by_code`）：内存 16GB、可用磁盘 5GB。
- 结果差异说明：不同 OS 的结果会有轻微波动（年化收益差异通常 <2%）。部分表格结果基于 Linux 环境生成。

## 推荐上手顺序

1. `examples/workflow_by_code.py` / `examples/workflow_by_code.ipynb`：用“搭积木”的方式手写完整 workflow（训练→预测→信号分析→回测）。
2. `examples/tutorial/detailed_workflow.ipynb`：更细粒度地拆解 Qlib 组件与研究流程。
3. `examples/benchmarks/`：查看/复现实验基准（Alpha158/Alpha360 等）与模型对比。
4. 进阶主题：动态适应（concept drift）、滚动训练、在线更新、组合优化、高频与 RL 等。

## 目录索引（按主题）

### 快速开始与通用工作流

- [`examples/workflow_by_code.py`](workflow_by_code.py)：
  - 说明：用 Python 代码构建与 `qrun *.yaml` 类似的研究流程；适合快速理解 Qlib 的核心对象（模型、数据集、Recorder、回测）。
  - 运行：`python src/examples/workflow_by_code.py`
- [`examples/workflow_by_code.ipynb`](workflow_by_code.ipynb)：
  - 说明：Notebook 版本，包含 Colab 运行说明与依赖处理逻辑。
  - 运行：用 Jupyter 打开并逐格执行。
- [`examples/tutorial/detailed_workflow.ipynb`](tutorial/detailed_workflow.ipynb)：
  - 说明：更“教程化”的端到端流程拆解，适合想深入各组件细节的用户。
  - 运行：用 Jupyter 打开并逐格执行。

### 基准模型（alpha 挖掘）

- [`examples/benchmarks/`](benchmarks/)（总览：[`examples/benchmarks/README.md`](benchmarks/README.md)）
  - 说明：一组用于“打分/预测→构建组合→评估收益”的基准方法集合，包含线性模型、树模型、深度模型等；并提供 Alpha158/Alpha360 等数据集上的对比表。
  - 典型运行方式：进入具体模型目录，安装其 `requirements.txt`，执行 `qrun workflow_config_*.yaml`。
  - 批量复现：[`examples/run_all_model.py`](run_all_model.py) 可用于批量运行与汇总多次随机种子结果。
    - 注意：脚本内部包含对 Conda 与 `bin/python` 路径的假设（并标注 `# TODO: FIX ME!`），在 Windows 上可能需要自行调整。

### 动态市场适应（concept drift / rolling）

- [`examples/benchmarks_dynamic/`](benchmarks_dynamic/)（总览：[`examples/benchmarks_dynamic/README.md`](benchmarks_dynamic/README.md)）
  - 说明：面向“市场非平稳/数据分布随时间变化”的场景，展示周期性滚动重训与分布生成式适应等方案。
  - [`examples/benchmarks_dynamic/baseline/`](benchmarks_dynamic/baseline/)：周期性滚动重训（RR, Rolling Retrain）框架（见 [`README.md`](benchmarks_dynamic/baseline/README.md)）。
    - RR 是 concept drift 场景下最常用的朴素基线：每隔固定间隔（如 20 个交易日）用最新数据重新训练一次模型，并对下一段时间做预测；最终把各窗口的预测按时间拼接，再统一做 IC/回测评估。
    - 入口脚本：`rolling_benchmark.py`（基于 `qlib.contrib.rolling.base.Rolling`，用 `RollingGen(step=...)` 自动生成滚动任务）。
    - 默认设置：`horizon=20`、`step=20`（20 日收益标签 + 每 20 个交易日重训一次）。
    - 运行：`python src/examples/benchmarks_dynamic/baseline/rolling_benchmark.py run`（或进入目录后 `python rolling_benchmark.py run`）。
    - 常用参数：`--conf_path`（选择模型配置）、`--step`（重训频率）、`--horizon`（标签/预测周期）、`--exp_name`（最终汇总评估实验名）；结果默认记录在 `mlruns/`。
  - [`examples/benchmarks_dynamic/DDG-DA/`](benchmarks_dynamic/DDG-DA/)：DDG-DA（见 [`README.md`](benchmarks_dynamic/DDG-DA/README.md)），并提供 `vis_data.py` 用于可视化相关数据。
  - 注意：DDG-DA 示例的硬件需求显著更高（其 `README.md` 中给出最小建议内存 45GB）。

### 数据相关示例

- [`examples/data_demo/`](data_demo/)（[`examples/data_demo/README.md`](data_demo/README.md)）
  - 说明：演示 Qlib 数据模块的常见用法，例如缓存与内存复用。
  - 入口脚本：`data_cache_demo.py`、`data_mem_resuse_demo.py`
- [`examples/rolling_process_data/`](rolling_process_data/)（[`examples/rolling_process_data/README.md`](rolling_process_data/README.md)）
  - 说明：滚动训练时，如何避免在每个滚动窗口重复生成全量数据；示例使用基于 DataHandler 的 DataLoader + Processor 来在窗口内生成“与窗口相关的加工特征”。
  - 运行：`python src/examples/rolling_process_data/workflow.py rolling_process`
- [`examples/orderbook_data/`](orderbook_data/)（[`examples/orderbook_data/README.md`](orderbook_data/README.md)）
  - 说明：演示“非固定频率数据”（如订单簿/逐笔数据）支持，使用基于 Arctic 的后端；包含导入示例数据与表达式计算示例。
  - 依赖：MongoDB、`arctic` 等（详见其 `README.md`）。

### 高频与订单执行

- [`examples/highfreq/`](highfreq/)（[`examples/highfreq/README.md`](highfreq/README.md)）
  - 说明：包含高频数据集示例，以及“预测高频价格趋势”的基准结果；并演示 Dataset 的 dump/load/reinit。
  - 运行：`python src/examples/highfreq/workflow.py get_data` 或 `python src/examples/highfreq/workflow.py dump_and_load_dataset`
- [`examples/nested_decision_execution/`](nested_decision_execution/)（[`examples/nested_decision_execution/README.md`](nested_decision_execution/README.md)）
  - 说明：回测中的“嵌套决策执行”示例：不同频率使用不同策略（如周频生成组合、日频执行；或日频生成、分钟执行）。
  - 运行：`python src/examples/nested_decision_execution/workflow.py backtest`（以及 `backtest_highfreq`）
- [`examples/rl_order_execution/`](rl_order_execution/)（[`examples/rl_order_execution/README.md`](rl_order_execution/README.md)）
  - 说明：面向订单执行场景的强化学习训练与回测流程（含 PPO/OPDS/TWAP 等），并提供数据预处理脚本与配置。
  - 运行入口：以 `README.md` 中的命令为准（包含数据下载、pickle 数据生成、训练与回测）。

### 超参搜索与模型解释

- [`examples/hyperparameter/LightGBM/`](hyperparameter/LightGBM/)（[`examples/hyperparameter/LightGBM/Readme.md`](hyperparameter/LightGBM/Readme.md)）
  - 说明：基于 Optuna 的 LightGBM 超参搜索示例（Alpha158/Alpha360）。
  - 运行：按 `Readme.md` 分别启动 Optuna Study + Dashboard，并执行 `hyperparameter_158.py` / `hyperparameter_360.py`。
- [`examples/model_interpreter/feature.py`](model_interpreter/feature.py)
  - 说明：训练树模型并输出特征重要性，快速验证因子/特征贡献。
  - 运行：`python src/examples/model_interpreter/feature.py`

### 滚动训练与在线更新（工程化）

- [`examples/model_rolling/task_manager_rolling.py`](model_rolling/task_manager_rolling.py)
  - 说明：基于 TaskManager 的滚动任务生成、训练与结果收集示例；适合了解多任务/滚动实验管理。
  - 运行：`python src/examples/model_rolling/task_manager_rolling.py main`
  - 注意：示例中默认使用 MongoDB 作为任务后端（参数中给出 `task_url` / `task_db_name`）。
- [`examples/online_srv/`](online_srv/)（详解：[`examples/online_srv/README.md`](online_srv/README.md)）
  - 说明：面向“数据不断追加、需要日频例行更新”的场景，展示如何管理 online 模型、增量更新预测、生成/更新滚动任务并产出交易信号；同时提供历史回放（simulate）验证流程。
  - 入口脚本：
    - `update_online_pred.py`：只做“训练一次 → 标记 online → 每天增量更新 pred”
      - `python src/examples/online_srv/update_online_pred.py first_train`
      - `python src/examples/online_srv/update_online_pred.py update_online_pred`
    - `rolling_online_management.py`：`OnlineManager + RollingStrategy` 的在线管理例子（first_run / routine / add_strategy）
    - `online_management_simulate.py`：沿历史日历模拟 online 流程，并用 signals 做示例回测
  - 注意：部分 Trainer（如 `*RM`）依赖 MongoDB 作为任务池后端；请按 `examples/online_srv/README.md` 调整 `task_url` 或切换 Trainer。

### 组合优化（Portfolio）

- [`examples/portfolio/`](portfolio/)（[`examples/portfolio/README.md`](portfolio/README.md)）
  - 说明：演示如何用优化型策略（如 `EnhancedIndexingStrategy`）在“收益-风险/跟踪误差”之间做权衡，替代简单的 TopK 规则策略。
  - 运行：`qrun src/examples/portfolio/config_enhanced_indexing.yaml`（并按其 `README.md` 准备权重与风险数据）。

### 强化学习（通用示例）

- [`examples/rl/simple_example.ipynb`](rl/simple_example.ipynb)
  - 说明：一个极简 RL 示例，演示如何构建 simulator、policy、reward，并运行训练与回测工作流。
  - 运行：用 Jupyter 打开并逐格执行。
