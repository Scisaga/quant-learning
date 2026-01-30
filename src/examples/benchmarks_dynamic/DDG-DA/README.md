# DDG-DA 示例（可预测概念漂移适应）

本目录是论文 **DDG-DA: Data Distribution Generation for Predictable Concept Drift Adaptation** 的一个可运行示例，基于 Qlib 的 `Meta` 组件实现：

- 论文链接：*DDG-DA: Data Distribution Generation for Predictable Concept Drift Adaptation*（arXiv: https://arxiv.org/abs/2201.04038）
- 代码主体实现：`qlib/contrib/rolling/ddgda.py`（类 `DDGDA`）
- 示例入口脚本：`workflow.py`（类 `DDGDABench`，封装成 CLI）

与 `baseline/` 中的 RR（周期性滚动重训）不同，DDG-DA 会训练一个“元模型（meta model）”去刻画/预测数据分布的演化趋势，然后把该知识迁移到后续的滚动训练任务里，为每个滚动窗口生成更合适的训练样本权重（或等价的重加权策略），从而提升对 concept drift 的适应能力。

## 代码是否“完整”？

就仓库内容而言，本示例是完整的：

- `workflow.py` 直接调用 `qlib.contrib.rolling.ddgda.DDGDA` 完成数据准备、meta 模型训练、滚动任务重加权、滚动训练、拼接与评估。
- 预测模型/回测配置复用 `../baseline/` 下的 YAML（`workflow_config_linear_Alpha158.yaml`、`workflow_config_lightgbm_Alpha158.yaml`）。

需要注意的是：运行它依赖一个可用的 Python 环境（Qlib 依赖 + PyTorch），以及足够的内存/磁盘资源（见下文 Requirements）。

## 目录结构

- `workflow.py`：DDG-DA 运行入口（`python workflow.py run`）。默认用 `../baseline/workflow_config_linear_Alpha158.yaml`。
- `vis_data.py`：可视化分析脚本（对 meta 输入/输出、子模型权重做热力图等；依赖 `matplotlib/seaborn/tqdm`，且需要先跑完 `workflow.py` 生成 `*.pkl`）。
- `requirements.txt`：额外依赖（仅列出 `torch==1.10.0`；其余依赖来自 Qlib 本身或你的环境）。
- `Makefile`：清理中间产物（`*.pkl`、`mlruns/`）。

## DDG-DA 具体做了什么（对应到代码）

`workflow.py` 继承 `DDGDA`，最终调用 `DDGDA.run()`（见 `qlib/contrib/rolling/ddgda.py`）。它大致分为两段：先准备/训练 meta 模型，再进行滚动训练与评估。

### 1) 准备 proxy 数据与 handler

`_dump_data_for_proxy_model()` 会从基础任务（YAML 的 `task:`）构建一个用于 meta 学习的“proxy 数据源”：

- 先从数据集中取出特征与标签，并可选用 GBDT 计算特征重要性，选取 Top-N（默认 `fea_imp_n=30`）。
- 可选对特征做标准化/填充（默认 `meta_data_proc="V01"`）。
- 产出文件：
  - `fea_label_df.pkl`：拼好的特征+标签 DataFrame
  - `handler_proxy.pkl`：将上述数据封装成 `DataHandlerLP`，供后续 meta dataset 复用

### 2) 生成 meta 输入（InternalData）

`_dump_meta_ipt()` 会构建 `InternalData` 并落盘：

- 使用 `sim_task_model`（默认 `gbdt`）来计算不同时段数据的相似度/内部度量（在实现里会跑一个用于数据相似度的训练流程）。
- 产出文件：`internal_data_s{step}.pkl`（例如 `internal_data_s20.pkl`）。
- 同时会在 `mlruns/` 里创建实验（默认名形如 `data_sim_s20`）保存相关中间结果。

### 3) 训练 meta model（MetaModelDS）

`_train_meta_model()` 会：

- 基于上一步的 `InternalData` 构建 `MetaDatasetDS`，再训练 `MetaModelDS`。
- 默认把 meta 模型记录到实验 `DDG-DA`（固定名字），并保存对象 `model`。
- meta 任务的起止日期可以用 `--meta_1st_train_end` 等参数调整（见“常用参数”）。

### 4) 用 meta model 给滚动任务做推理，生成带重加权的 tasks

`get_task_list()` 会：

- 读取实验 `DDG-DA` 中保存的 meta 模型。
- 先按 RR 的方式生成“基础滚动任务”（来自 `Rolling.get_task_list()`，即 `RollingGen(step=...)` 生成的窗口任务）。
- 构造 `MetaDatasetDS(task_mode=MetaTask.PROC_MODE_TRANSFER)`，调用 `meta_model.inference(...)` 把 meta 学到的知识迁移到这些基础滚动任务上，产出“新任务列表”（通常会在 task 中附加 `reweighter` 等信息）。
- 产出文件：`tasks_s{step}.pkl`（例如 `tasks_s20.pkl`）。

### 5) 执行滚动训练、拼接预测并统一评估

最后 `super().run()`（见 `qlib/contrib/rolling/base.py`）会：

- 用“新任务列表”逐窗口训练并记录 `pred/label`。
- 用 `RollingEnsemble` 拼接各窗口预测。
- 在拼接后的序列上做统一的 IC/回测评估（记录器来自 YAML 里的 `task.record`）。

## 数据集说明

- 论文原始数据是私有数据；本示例使用 Qlib 的公开数据集做复现/演示。
- 若未设置环境变量 `PROVIDER_URI`，`workflow.py` 会尝试自动下载并使用 CN 公共数据（默认目录 `~/.qlib/qlib_data/cn_data`）。
- 若你想复用已有数据目录，可设置 `PROVIDER_URI` 指向你的数据路径。

补充：`examples/benchmarks_dynamic/README.md` 中给出了 crowd-sourced 数据的下载方式；在一些场景下（例如需要 `VWAP` 等字段）会显著影响结果与可复现性。

## 运行方式

建议在本目录下执行：

```bash
python workflow.py run
```

默认配置是 **Linear + Alpha158**（配置文件位于 `../baseline/`）。切换为 LightGBM：

```bash
python workflow.py --conf_path=../baseline/workflow_config_lightgbm_Alpha158.yaml run
```

为了更方便和 `vis_data.py` 配合（其默认读取实验名 `DDG-DA` / `rolling_ds` / `rolling_models`），建议显式指定实验名：

```bash
python workflow.py --rolling_exp=rolling_models --exp_name=rolling_ds run
```

## 常用参数（CLI）

`workflow.py` 使用 `fire` 暴露参数，既包含 DDG-DA 自己的参数，也包含 Rolling 基类参数；常用的有：

- 模型/任务配置：`--conf_path`（指向 `../baseline/workflow_config_*.yaml`）
- RR/滚动相关：`--step`（滚动步长/重训频率，默认 20）、`--horizon`（标签周期，默认 20）
- 实验命名：`--rolling_exp`（窗口级实验名）、`--exp_name`（拼接评估实验名）
- DDG-DA（meta）相关：
  - `--sim_task_model`：`gbdt` 或 `linear`（用于数据相似度/内部度量的模型）
  - `--fea_imp_n`：特征重要性选 Top-N（默认 30；设为 `None` 表示不筛选）
  - `--meta_data_proc`：meta 数据处理方式（默认 `V01`；设为 `None` 可关闭）
  - `--meta_1st_train_end`：meta 第一个任务训练集截止日期
  - `--segments`、`--hist_step_n`、`--alpha`、`--loss_skip_thresh`：meta dataset / meta model 的超参

## 输出与产物

运行后你会在本目录看到（文件名与 `step` 相关）：

- `fea_label_df.pkl`、`handler_proxy.pkl`
- `internal_data_s20.pkl`（或 `internal_data_s{step}.pkl`）
- `tasks_s20.pkl`（或 `tasks_s{step}.pkl`）

实验与评估结果默认写入 `mlruns/`：

- `DDG-DA`：meta 模型实验（保存 `model`）
- `data_sim_s{step}`：生成 `InternalData` 时的相似度/内部度量相关实验
- `rolling_exp`：各滚动窗口的子任务训练记录（由 `--rolling_exp` 控制）
- `exp_name`：拼接后的统一评估记录（由 `--exp_name` 控制）

## 可视化（vis_data.py）

`vis_data.py` 是一个偏 Notebook 风格的分析脚本：

- 需要先运行 `workflow.py` 生成 `internal_data_s20.pkl` 与 `tasks_s20.pkl`。
- 它默认会读取实验 `DDG-DA` 的 meta 模型，并尝试读取 `rolling_ds` / `rolling_models` 实验来画子模型权重热力图。
- 若你使用了不同的实验名，请在 `vis_data.py` 中改为你自己的实验名，或按上文示例固定为 `rolling_ds` / `rolling_models`。

## 清理旧结果

重复运行时若遇到 MLflow 实验名冲突或想释放磁盘，可清理本目录中间文件与 `mlruns/`：

- Linux/macOS：`make clean`
- PowerShell（Windows）：`Remove-Item -Recurse -Force .\\mlruns, .\\*.pkl -ErrorAction SilentlyContinue`

## Requirements（硬件与依赖）

- 依赖：需要安装 Qlib 及其依赖，并安装 PyTorch（本示例 `requirements.txt` 里固定为 `torch==1.10.0`）。
- 硬件（运行 `workflow.py` 的最低建议）：
  - 内存：45GB
  - 磁盘：4GB

仅使用 CPU + RAM 也可以运行（不强制 GPU），但耗时与内存压力会更大。

