## 任务管理（Task Management）

任务管理是 Qlib Workflow 的“批量编排层”。当你不再只跑一次 `qrun` 或一次训练脚本，而是要做：

- 滚动训练/滚动回测（rolling / walk-forward）
- 多组超参网格/对比实验
- 多个股票池/多个 label/多个特征集合的组合实验
- 分布式 worker 消费同一个任务池

这时就需要把“单次实验的配置”抽象成 **Task**，再把 Task 放入一个 **任务池（Task Pool）**，由一个或多个 **Trainer/Worker** 去并发执行，并用 **Collector** 汇总结果。

![任务管理步骤](img/qlib-1762369488486.png)

> 官方参考：Qlib 文档 [Task Management](https://qlib.readthedocs.io/en/latest/advanced/task_management.html) 与 `microsoft/qlib` 的 rolling 示例脚本。

---

### 1) 核心概念

- **Task（任务定义）**
  - 一个“可执行的实验配置”（通常是 `dict`，结构与 `qrun` 的 YAML 配置接近）。
  - 一般包含：`model`、`dataset`、`record`（记录/评估器）以及自定义字段（例如 rolling 的窗口定义）。

- **TaskGen（任务生成器）**
  - 负责把“模板任务（base task）”扩展成一批任务列表，例如 RollingGen 按时间滚动生成多个训练/验证/测试切片。

- **TaskManager（任务池/任务存储）**
  - 管理任务的生命周期：插入、去重、领取（claim）、状态流转、释放等。
  - 常用后端是 **MongoDB**（适合并发 worker 共享任务池）。

- **Trainer / Worker（任务执行者）**
  - 从任务池领取 `WAITING` 任务，跑训练 + 产出 artifacts（通常写入 MLflow/Recorder）。

- **Recorder（实验记录）**
  - Qlib 用 Recorder 记录一次任务的结果：参数、指标、产物（pred、report、ic 等）。
  - 本仓库使用 MLflow file store：`mlruns/<experiment_id>/<run_id>/...`

- **Collector（结果汇总）**
  - 汇总一批任务的指标与产物，做对比、分组统计、集成（ensemble）等。

---

### 2) 任务状态与并发模型

典型状态机（不同版本命名略有差异）：

- `WAITING`：待执行
- `RUNNING`：已被某个 worker 领取执行
- `PART_DONE`：阶段性完成（例如滚动任务的分段结果已写入）
- `DONE`：任务结束（成功或失败，失败可能会有错误信息/异常栈）

并发执行的关键点：

- **TaskManager 的“领取任务”必须是原子操作**：保证多个 worker 不会抢到同一条任务。
- Worker 与任务池解耦：想扩容就启动更多 worker。
- “去重”依赖 task 的 `filter`（或等价字段）：确保同一配置不会重复插入。

---

### 3) 任务定义长什么样

任务定义通常包含下面三块（与 `qrun` 配置一致/高度相似）：

- `model`：模型类与参数（例如 `LGBModel` + LightGBM 超参）
- `dataset`：数据集与切片（`DatasetH` + `segments={train/valid/test}`）
- `record`：记录哪些产物与评估（`SignalRecord`、`SigAnaRecord`、`PortAnaRecord` 等）

> 本仓库的单次训练脚本 `backend/qlib/train/train_lgb_alpha158_pit.py` 就是一个“任务配置落地成 Python”的例子：
> - `market/instruments`：`--market csi300`
> - `segments`：`--train/--valid/--test`
> - `label`：`--label-expr`
> - 模型参数：`LGBModel` 的 `kwargs`
> - 产物写入：`R.save_objects(trained_model=model)` + `SignalRecord.generate()`

---

### 4) MongoDB 任务池配置

Qlib 的任务管理通常需要一个 MongoDB（单机/集群均可）。常见配置方式：

```python
from qlib.config import C

C["mongo"] = {
  "task_url": "mongodb://localhost:27017/",
  "task_db_name": "qlib_task_db",
}
```

- `task_url`：MongoDB 连接串
- `task_db_name`：任务库名（一个库可以有多个 task pool/collection）

> 没有 MongoDB 也能做“伪任务管理”：自己写 for-loop 批量跑 `qrun` 或训练脚本，但就没有共享任务池与并发 worker 的能力。

---

### 5) 最小可运行流程（推荐落地顺序）

#### A. 单任务先跑通（验证数据/配置）

1) 用脚本跑一次训练：

```bash
python backend/qlib/train/train_lgb_alpha158_pit.py --provider-uri data/qlib_data/cn_data --market csi300 --exp-name tutorial_exp
```

2) 生成 HTML 报告（基于 Recorder/MLflow artifacts）：

```bash
python backend/qlib/backtest/generate_html_report.py --exp-name tutorial_exp --recorder-id <RID>
```

你会在 `mlruns/` 看到一次 run 的完整产物（见 `docs/qlib-数据项.md`）。

#### B. 把单任务模板化（base task）

把上面“单次训练”的关键参数整理成一个 `base_task`：

- 固定：模型类型、特征集合、label 口径
- 可变：train/valid/test 的时间切片、股票池、超参网格

#### C. 生成一批任务（rolling / grid）

- Rolling：根据时间窗生成多个 `(train, valid, test)` 组合
- Grid：根据参数列表生成多个模型超参组合

#### D. 入库（TaskManager.create_task）

把任务列表写入 MongoDB 的某个 pool：

- 任务池名建议包含用途：例如 `rolling_lgb_alpha158_pit`
- `filter` 字段务必包含能唯一标识配置的键（避免重复插入）

#### E. 启动一个或多个 Worker 消费任务

- 单机：开多个进程/多个终端
- 多机：多台机器共用 MongoDB，同一个 pool 即可实现水平扩容

#### F. 收集与对比（Collector）

把多次 run 的指标汇总成表格/图：

- 滚动窗：看 IC/IR 的时间序列稳定性
- 超参对比：看收益、回撤、换手、成本敏感性

---

### 6) 参数与可复现（强烈建议）

要让任务管理真正“可复现 + 可追溯”，建议你在每个任务/每次 run 都能回答这些问题：

- 训练/验证/测试的时间段分别是什么？
- 股票池（market/instruments）是什么？
- 特征集合是什么（是否包含 `pit_` 财务特征）？
- label 表达式是什么？
- 模型超参是什么？
- 回测配置（topk/n_drop/成本/benchmark）是什么？

本仓库的实践建议：

- 在训练脚本里显式打印并/或记录关键参数（`R.log_params(...)` 或写入 artifacts）。
- 产物统一写进 Recorder（MLflow artifacts），报告脚本只读 Recorder，不依赖 notebook 状态。
- 对 PIT 财务：优先用离线生成的 `features/<inst>/pit_*.day.bin`，避免线上读取最新财务数据导致“未来信息”。

---

### 7) 常见坑位

- **Mongo 连接/权限问题**：任务插入失败或 worker 领取失败，优先检查 `task_url`、账号权限、collection 是否可写。
- **任务卡在 RUNNING**：worker 异常退出但状态未回滚，通常需要 `force_release` 或手动重置状态（不同版本 API 不同）。
- **MLflow 参数冲突**：同一个 run/resume 时写入相同 key 不同 value 会报错；报告脚本里避免用 `R.start(resume=True)`，直接 `R.get_recorder(...)` 读取。
- **数据泄露**：rolling 任务要保证 label/特征都严格只使用当时可见的信息；财务数据务必使用 PIT。

---

### 8) 与本仓库内容的关联

- 单次训练/写入 Recorder：`backend/qlib/train/train_lgb_alpha158_pit.py`
- 报告生成（含实验参数汇总）：`backend/qlib/backtest/generate_html_report.py`
- 数据/产物解释：`docs/qlib-数据项.md`
- 工作流（qrun/YAML）：`docs/qlib-工作流.md`
- 回测与投组：`docs/qlib-投组管理与回测.md`

