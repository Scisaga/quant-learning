## 元学习 Meta Controller = Meta-Task & Meta-Dataset & Meta-Model

金融市场非平稳，训练期与未来测试期分布常发生漂移；单一模型易“过期”。Meta Controller 的目标是在一系列预测任务之间学习可迁移的“规律”，并用这些规律去指导后续的预测模型/任务（guidance），从而提升鲁棒性与泛化。
* **问题**：金融等流数据场景里分布会随时间漂移（concept drift），如果训练只跟着“最近数据”走，往往**滞后一步**，新漂移来了你才去适应。
* **思想**：在很多场景里，漂移**并非完全随机**，存在季节性/周期性/趋势，因此**可预测未来分布趋势**，先把训练集“重采样/生成”为**更像未来**的分布，再训练基础预测模型，让它**提前对齐未来**。

* **MetaTask**：元学习框架的最小单元，保存可供 MetaModel 使用的数据；同一组 MetaTask 可共享 DataHandler。训练/测试/迁移阶段对输入数据的处理不同（`PROC_MODE_FULL / TEST / TRANSFER`）——训练任务通常需要 `X, y, X_test, y_test`，而迁移到其他数据集时只需 `meta_info`。常用 `prepare_task_data()` 得到可直接喂给 MetaModel 的数据。
* **MetaTaskDataset**：负责**生成元信息并组织 MetaTask 列表**，对 MetaModel 提供训练/推理所需的“元数据”。支持**跨数据集迁移**（在 A 上学到的模式可用于 B），常用 `prepare_tasks(["train","test"])` 取回各分段的任务列表。
* **MetaModel**：执行“指导”。分两类：

  1. **MetaTaskModel**：**改写/生成基础任务定义**（prepare_tasks 产出被改写的任务定义，再用这些定义去训练新的基础预测模型）。
  2. **MetaGuideModel**：**插入基础模型训练过程**，在训练阶段提供指引（超参/采样/损失加权等）。均暴露 `fit(..)` / `inference(..)`。

### DDG-DA（Data Distribution Generation for Predictable Concept Drift Adaptation）

0. AAAI 2022《DDG-DA: Data Distribution Generation for Predictable Concept Drift Adaptation》。在股票趋势、电力负荷、太阳辐照等真实任务上，对多种主流模型都有显著提升。
1. **学未来分布**：用一个“分布预测器”（元模型的一部分，参数常记为 θ）去**预测下一时段的目标分布**；实现上用历史数据学到的“分布表征”与一个**可微的分布距离**来训练该预测器。
2. **生成训练集合成分布**：基于预测的未来分布，对历史样本**加权重采样**（或生成等价的数据权重），得到一个更贴近未来分布的“合成训练集”。
3. **再训练基础模型**：用合成训练集去训练真正做预测的基础模型（如 LGBM/MLP 等），从而在未来测试期表现更好。
  - 论文给出了**可微分布距离**并证明它与 KL 等传统距离的等价性，从而能端到端学习“如何重采样到更像未来”的训练数据。
4. **有效**：存在**可预测的漂移**（季节性/周期性/再现的 market regime 等），例如你常见的月度/季度节奏、行业轮动带来的分布缓慢迁移。
5. **不太有效**：**纯粹突发且难以建模的跳变**（如黑天鹅、极端政策/财报意外），这类漂移**不可预测**，DDG-DA 也难以提前合成出“未来像什么”。论文里也强调“多数漂移呈非随机趋势；若完全随机则难处理”。

Qlib 把 DDG-DA 实现为 **Meta Model** 的一个示例，完整流程是：
① 计算 **meta-information** 并封成 **MetaTask**（组成 **MetaDataset**）→ ② 在训练段上训练 DDG-DA → ③ 推理得到“指导信息”（即未来分布引导/样本权重等）→ ④ 将指导应用到基础预测模型训练以提升表现。

* 示例代码路径：`examples/benchmarks_dynamic/DDG-DA/workflow.py`。