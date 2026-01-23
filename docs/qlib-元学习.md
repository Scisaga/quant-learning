## 元学习 Meta Controller = Meta-Task & Meta-Dataset & Meta-Model

金融市场非平稳，训练期与未来测试期分布常发生漂移；单一模型易“过期”。Meta Controller 的目标是在一系列预测任务之间学习可迁移的“规律”，并用这些规律去指导后续的预测模型/任务（guidance），从而提升鲁棒性与泛化。

* **问题（为什么需要）**：金融等流数据场景里分布会随时间漂移（concept drift）。如果训练只跟着“最近数据”走，往往**滞后一步**：漂移发生后你才去适应。
* **直觉（它在做什么）**：很多漂移并非完全随机，可能有季节性/周期性/趋势。元学习希望从“很多个历史任务”里学到规律，让训练过程能**更主动**地对齐未来分布。

### 先把它想成一件事：让“训练过程”可学习

在 Qlib 里，你可以把元学习理解为：**再训练一个模型（MetaModel），它不直接预测股票，而是输出“怎么训练基础模型更好”的指导信息。**

* 基础预测模型（Forecast Model）学的是 `X -> y`：给每只股票打分/预测收益（详见 `docs/qlib-预测模型.md`）。
* 元模型（MetaModel）学的是 `task/history -> guidance`：告诉你哪些数据更该信、哪些窗口更相似、怎么加权/采样/设超参……

关键点：**Meta Controller 不是替代预测模型，而是控制/改写预测模型的训练方式。**

### 它在 Qlib 流程里的位置（心智模型）

把 Qlib 的“任务（task）”想成一个可执行的训练定义（模型 + 数据集 + 记录器）。Rolling / Online / Task 管理等都会不断生成/调度这些任务。Meta Controller 则插在它们“生成任务”和“训练任务”之间：

1) 先有一批基础任务（例如 rolling 切出来的很多时间窗口任务）  
2) MetaTaskDataset 把这些任务加工成 MetaTask（附带 meta-information）  
3) MetaModel 学到规律后，在推理时输出 guidance，并把它写回任务（或在训练时动态干预）  
4) Trainer 仍然训练基础预测模型，但训练数据/过程已经被指导过

### meta-information 到底是什么（不玄学）

meta-information 的核心要求只有一个：**它应该能“描述任务/时间窗口的状态”，并且在不同任务之间可比较**。常见形式包括：

* **任务窗口统计摘要**：特征分布/标签分布/缺失率/波动率等；
* **历史表现信号**：某个简单模型在不同数据片段上的表现（IC/RankIC/损失），形成“时间 × 数据片段”的矩阵；
* **市场状态特征**：把 market regime（牛/熊/高波动/低流动性）编码成可学习的向量。

在 DDG-DA（Qlib 实现）里，meta-information 就属于“历史表现信号”这一类：用一批 proxy 模型在滚动切分的数据片段上评估得到的表现矩阵，作为元模型输入的一部分（见后文实际流程里的 `InternalData` / `data_ic_df`）。

### 三个核心对象（对应 Qlib 抽象）

* **MetaTask**：元学习的最小单元 = `基础任务(task)` + `元信息(meta_info)` +（可选）为元模型准备的训练输入。  
  Qlib 的设计里，MetaTask 有不同处理模式（`PROC_MODE_FULL / TEST / TRANSFER`）：
  * `FULL`：训练元模型时通常需要 `X, y, X_test, y_test` 等（由具体 MetaTask 子类在初始化阶段准备好，并通过 `get_meta_input()` 暴露给元模型）。
  * `TEST`：只做评估/验证时，可能不需要准备完整输入。
  * `TRANSFER`：把元模型迁移到别的数据集/任务时，通常只依赖 meta-information（不再要求完整的 `X/y/...`）。
* **MetaTaskDataset**：负责**生成元信息并组织 MetaTask 列表**。它的职责是把“一堆基础任务”变成“可供元模型学习的一堆 MetaTask”，并通过 `prepare_tasks(["train","test"])` 提供不同分段的 meta tasks。
* **MetaModel**：执行“指导（guidance）”。分两类：

  1. **MetaTaskModel**：**改写/生成基础任务定义**（`inference(meta_dataset)` 产出被改写的任务定义，再用这些定义去训练新的基础预测模型）。
  2. **MetaGuideModel**：**插入基础模型训练过程**，在训练阶段提供指引（超参/采样/损失加权等）。均暴露 `fit(..)` / `inference(..)`。

工程落地上，一个非常常见、也最“好对照代码”的 guidance 形式是：**给任务注入 `reweighter`（样本权重器）**。训练器在执行任务时，会把 `task["reweighter"]` 传给 `model.fit(dataset, reweighter=...)`（对应 Qlib 的 trainer 实现逻辑，也与 `docs/qlib-预测模型.md` 的 fit 签名相匹配）。

### 容易混淆的三个概念（建议先分清）

* **Rolling（滚动训练）**：回答“任务怎么切分、每个窗口怎么重训/回测”。它会生成很多基础任务，但**不一定**用元学习。  
* **Online Serving（在线策略）**：回答“什么时候训练、哪些模型上线、每天怎么更新预测/信号”。它关心的是**模型生命周期管理**，不等同于元学习（详见 `docs/qlib-线上策略管理.md`）。  
* **Meta Controller（元学习）**：回答“在分布漂移里，基础模型的训练该如何被更聪明地改写/控制”。它关心的是**训练策略本身可学习**。

---

### DDG-DA（Data Distribution Generation for Predictable Concept Drift Adaptation）

0. AAAI 2022《DDG-DA: Data Distribution Generation for Predictable Concept Drift Adaptation》。
1. **学未来分布趋势**：训练一个“分布预测器”（元模型的一部分）去预测下一时段的目标分布趋势。
2. **生成/加权训练集**：基于预测的未来分布，对历史样本**加权重采样**（或生成等价的样本权重），得到更贴近未来的“合成训练集”。
3. **训练基础模型**：用合成训练集训练真正做预测的基础模型（如 Linear / LightGBM 等），从而在未来测试期更稳健。
4. **有效场景**：存在可预测漂移（季节性/周期性/再现的 market regime 等）。
5. **不太有效场景**：纯突发、不可建模的跳变（黑天鹅、极端政策/财报意外）。

Qlib 把 DDG-DA 实现为 **Meta Model** 的一个示例，抽象流程是：
① 构造 meta-information → ② 训练 MetaModel → ③ 推理得到 guidance（如权重）→ ④ 把 guidance 应用到基础预测模型训练。

* 本仓库示例入口：`src/benchmarks_dynamic/DDG-DA/workflow.py`。

---

## 实际执行流程（以本仓库 DDG-DA 为例）

下面只写“能从代码里直接确认”的执行链路：本仓库的入口脚本 + Qlib（`pyqlib==0.9.7`）对应源码的调用顺序与产物；如果你的本地 Qlib 版本不同，细节可能会有差异。

### 0）准备环境（必要条件）

* 依赖：
  * 本仓库根目录 `requirements.txt` 里包含 `pyqlib==0.9.7`；
  * DDG-DA 示例目录 `src/benchmarks_dynamic/DDG-DA/requirements.txt` 额外要求 `torch==1.10.0`（上游示例给的最小版本约束）。
* 数据：
  * 默认会下载 Qlib 公开数据到 `~/.qlib/qlib_data/cn_data`（入口脚本里调用 `qlib.tests.data.GetData().qlib_data(exists_skip=True)`）。
  * 或者你自行准备好数据，并通过环境变量 `PROVIDER_URI` 指向数据目录（入口脚本会把它透传给 `qlib.auto_init(provider_uri=...)`）。
* 资源：`src/benchmarks_dynamic/DDG-DA/README.md` 提到该示例的最小硬件需求大约是内存 45G、磁盘 4G（CPU 也可跑）。

### 1）选择配置（决定数据/模型/时间段）

DDG-DA 示例默认使用滚动训练基线目录下的配置文件：

* Linear：`src/benchmarks_dynamic/baseline/workflow_config_linear_Alpha158.yaml`
* LightGBM：`src/benchmarks_dynamic/baseline/workflow_config_lightgbm_Alpha158.yaml`

这些 YAML 里最关键的是：

* `qlib_init.provider_uri`：数据所在目录（默认 `~/.qlib/qlib_data/cn_data`）
* `task.model`：基础预测模型（Linear / LightGBM）
* `task.dataset.kwargs.segments`：train/valid/test 的时间划分

### 2）运行入口脚本（本仓库代码能确认的第一步）

入口是一个 Fire CLI，会实例化 `DDGDABench(DDGDA)` 然后调用你传入的方法名（常用 `run`）：

* 在示例目录运行：

```bash
cd src/benchmarks_dynamic/DDG-DA
python workflow.py run
```

* 在仓库根目录运行：

```bash
python src/benchmarks_dynamic/DDG-DA/workflow.py run
```

* 切换基础预测模型（LightGBM）：

```bash
cd src/benchmarks_dynamic/DDG-DA
python workflow.py --conf_path=../baseline/workflow_config_lightgbm_Alpha158.yaml run
```

入口脚本 `src/benchmarks_dynamic/DDG-DA/workflow.py` 的行为可以直接确认：

1. 若未设置 `PROVIDER_URI`：尝试下载 Qlib 数据（存在则跳过）。
2. 调用 `qlib.auto_init(...)` 初始化 Qlib。
3. 交给 `fire.Fire(DDGDABench)` 解析参数并执行（例如执行 `run`）。

### 3）DDG-DA 的“元学习 + 滚动”主流程（Qlib v0.9.7 源码能确认）

> 说明：本仓库没有把 Qlib 源码 vendoring 进来；DDG-DA 的核心实现来自 Qlib。以下步骤来自 Qlib v0.9.7 的 `qlib/contrib/rolling/ddgda.py` 与 `qlib/contrib/rolling/base.py`（可用同版本 tag 在 GitHub 上对照）。

#### 3.1 `DDGDA.run()`：先准备元模型，再做滚动训练

Qlib 里 `DDGDA.run()` 明确按下面顺序执行（并在 `working_dir` 下落盘中间产物）：

1) **准备代理数据（给“元模型”用的 proxy 任务）**：`_dump_data_for_proxy_model()`

* 从 `basic_task()` 拿到基础任务配置（来自 YAML），并根据 `sim_task_model`（默认 `gbdt`）调整任务（例如用 GBDT 做特征重要性/相似度；或用 linear 并补上预处理）。
* 从数据集准备出 `feature/label`，可选按特征重要性取 Top-N（`fea_imp_n`）。
* 落盘产物（默认在 `src/benchmarks_dynamic/DDG-DA/`）：
  * `fea_label_df.pkl`：拼好的特征/标签数据
  * `handler_proxy.pkl`：一个 `DataHandlerLP`（StaticDataLoader）形式的 handler，后续构造 meta tasks 会用到

2) **准备 meta-information（内部数据相似度/表现矩阵）**：`_dump_meta_ipt()`

* 构造一个“用于计算分布表征/相似度”的任务 `sim_task`，并用 `InternalData(sim_task, step, exp_name=...)` 做 `setup()`。
* `InternalData.setup()` 会把训练期切成 rolling 小段，在这些段上训练一批 proxy 模型，并抽取它们在不同日期上的表现，最终得到 `data_ic_df`（用来当作元信息输入）。
* 落盘产物：
  * `internal_data_s{step}.pkl`：序列化后的 `InternalData`（含 `data_ic_df` 等）

3) **训练元模型（MetaModel）**：`_train_meta_model()`

* 构造 `proxy_forecast_model_task`（只含 dataset，handler 指向 `handler_proxy.pkl`，并把时间段拆成 train/test）。
* 用 `MetaDatasetDS(...)` 把 `proxy_forecast_model_task` 生成一串 `MetaTaskDS`：
  * 任务生成：`RollingGen(step=self.step, trunc_days=1+self.horizon)`（避免信息泄露）
  * 每个 meta task 的 `meta_info`：来自 `_prepare_meta_ipt()`，它会在 `InternalData.data_ic_df` 里取最近 `hist_step_n * step` 天的历史“表现矩阵”，并对 overlap 区域做 mask
  * 如果是训练阶段（`PROC_MODE_FULL`），`MetaTaskDS` 还会准备 `X/y/X_test/y_test/time_belong/test_idx` 作为元模型训练输入
* 用 `MetaModelDS.fit(meta_dataset)` 训练元模型，并通过 MLflow（Qlib 的 `R`）保存。

#### 3.2 `DDGDA.get_task_list()`：元模型推理，把“指导”注入基础滚动任务

当后续进入滚动训练阶段，`Rolling._train_rolling_tasks()` 会调用 `get_task_list()`；在 DDG-DA 里它被重写为：

1. 从 MLflow 里加载 `MetaModelDS`（实验名默认是 `DDG-DA`）。
2. 调用 `Rolling.get_task_list()` 生成基础滚动任务列表（来自 YAML 的任务模板 + `RollingGen(step=self.step, trunc_days=self.horizon+1)`）。
3. 把“基础滚动任务列表”包装成 `MetaDatasetDS(..., task_mode=MetaTask.PROC_MODE_TRANSFER, segments=0.0)`（此时只需要 meta_info，不再需要准备 `X/y/...`）。
4. 调用 `meta_model.inference(mds)` 得到 **新任务列表**：对每个 task 做浅拷贝，并注入一个 `reweighter`（`TimeReweighter`），用于给样本/时间片加权。
5. 落盘产物：
  * `tasks_s{step}.pkl`：推理后的任务列表（带 `reweighter`）

#### 3.3 `Rolling.run()`：执行滚动训练并汇总评估

最后 `DDGDA.run()` 调用 `super().run()`（也就是 `Rolling.run()`），它做三件事：

1. `TrainerR(experiment_name=rolling_exp)` 训练每个 rolling task（每个 task 对应一个时间窗口）。
2. `RollingEnsemble()` 把所有 rolling 的预测拼接/集成。
3. 按 YAML 里配置的 record（例如 `PortAnaRecord` 等）做评估，并把结果保存到 MLflow 的 `mlruns/` 目录。

### 4）如果你“在本仓库里找不到元学习代码”的原因

本仓库的 DDG-DA 示例主要是入口脚本（`workflow.py`）+ 配置（YAML）；元学习的核心实现（`DDGDA` / `MetaDatasetDS` / `MetaModelDS` / `MetaTask` 等）来自 Qlib（`pyqlib` 包）。因此：

* 你在本仓库里 grep 不到 `DDGDA` 的类定义是正常的（它不在本仓库）。
* 若需要逐行跟踪元学习细节，请对照 Qlib 的同版本源码（例如 `v0.9.7` tag）再看上述函数的实现。
