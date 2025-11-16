# DoubleAdapt

## 一、基本信息与提出背景

* 全称：**DoubleAdapt: A Meta-learning Approach to Incremental Learning for Stock Trend Forecasting**
* 作者：Lifan Zhao, Shuming Kong, Yanyan Shen（上海交通大学 DMT 组）
* 发表：**KDD 2023**（Long paper）([arXiv][1])
* 官方代码：

  * 论文主仓库 API：**SJTU-DMTai/DoubleAdapt**，为通用 PyTorch 版本，不依赖 qlib([GitHub][2])
  * 与 qlib 的集成：通过 PR 将脚本放到 `examples/benchmarks_dynamic` 下，作为增量学习基线之一（PR 1560）([GitHub][3])

研究目标：在**股票趋势预测的在线场景**中，处理**随时间变化的数据分布（concept drift / distribution shift）**，在保证收益的同时控制计算开销，实现**增量学习（Incremental Learning, IL）**框架下的 **SOTA 预测表现**。([arXiv][1])

---

## 二、问题设定：带分布漂移的增量股票趋势预测

### 1. 股票趋势定义

论文将股票趋势定义为未来一天的收益率，例如：

[
y_t = \frac{P_{t+1} - P_t}{P_t}
]

其中 (P_t) 通常为收盘价（也可用开盘价或 VWAP）。特征 (x_t) 为若干天内的 O/H/L/C/V 等指标（在实验中主要使用 Alpha360 的 6 维 K 线特征序列）。([arXiv][1])

### 2. 增量学习任务序列

* 将时间轴按固定间隔 (S)（例如每 20 个交易日）切成一系列 **IL 任务**：(\mathcal{T}_1, \mathcal{T}_2, \dots)。
* 对于第 (k) 个任务：

  * 增量训练集 (D^{\text{train}}_k)：最近一段时间的数据；
  * 测试集 (D^{\text{test}}_k)：随后的若干天数据；
* 在线流程：

  1. 用当前参数在 (D^{\text{test}}_k) 上预测；
  2. 等到这些日期的真实收益率可见，再计算损失；
  3. 用 (D^{\text{train}}_k) 更新模型；
  4. 进入下一任务 (\mathcal{T}_{k+1})。([arXiv][1])

目标：在整个测试区间内，综合最大化 **IC / ICIR / Rank IC / Rank ICIR 以及超额年化收益与 IR**。([arXiv][4])

---

## 三、核心思想：双重适配 + 元学习的增量框架

DoubleAdapt 的核心有三点：([arXiv][5])

1. **把一串增量任务当作元学习任务序列**

   * 每个 IL 任务是一个“任务（task）”；
   * 跨任务学习“如何快速适配”（meta-learning），而不是每次从零开始微调。

2. **双重适配（Double Adaptation）**

   * **数据适配器（Data Adapter, DA）**：

     * 对特征和标签做可学习变换，使每个任务在变换后更加“局部平稳”，弱化短期剧烈波动和噪声；
   * **模型适配器（Model Adapter, MA）**：

     * 为当前任务生成一个更合适的模型初始参数，使少量增量数据上的训练更高效、更稳定。

3. **双层优化（Bi-level Optimization）**

   * 下层：给定当前 DA/MA，像普通增量学习那样更新基础预测模型；
   * 上层：用任务测试集上的表现更新 DA/MA 参数，即经典 MAML 风格的元学习，只是这里的 query set 是**当前任务的测试时间段**。([arXiv][1])

---

## 四、模型结构与训练流程

可以将 DoubleAdapt 看成三块拼起来的系统：

> **数据适配器（DA） → 预测模型（GRU/LSTM/CNN/Transformer 等） → 模型适配器（MA）**

### 4.1 数据适配器 Data Adapter

目标：在进入预测模型前，对 **特征** 和 **标签** 做轻量级可学习变换，缓解短期分布漂移。([arXiv][1])

#### (1) 特征适配（Feature Adaptation）

* 输入：某只股票在一个 lookback 窗口内的特征矩阵 (X\in \mathbb{R}^{L \times d})（长度 L，维度 d）。
* 结构：**多头（multi-head）仿射变换 + 门控**：

  * 每个 head (h) 有一组线性变换 (W_h, b_h)；
  * 为每个样本计算其与各个“原型向量”的相似度，通过 softmax 得到各个 head 的权重；
  * 输出是所有 head 变换结果的加权和。([arXiv][1])
* 共享策略：论文在附录里强调，为避免 RNN 梯度消失，**所有时间步共享同一个适配层参数**，只是在不同样本上有不同的 head 权重。([arXiv][1])

直观理解：

* 不同 head 对应“不同的市场状态/风格”；
* DA 为每个样本选择“更合适的状态变换”，让输入特征看起来更接近训练期见过的模式。

#### (2) 标签适配（Label Adaptation）

* 输入：原始标签（未来收益或 rank）；
* 步骤：

  1. 线性映射到低维嵌入；
  2. 通过与若干原型向量的相似度做 head 选择（类似特征适配）；
  3. 在被选中的 head 上进行仿射或简单非线性变换，得到“适配后的标签”；
  4. 再通过可逆映射还原到原标签空间，或在该空间上优化。([arXiv][1])
* 同时加入一个**正则项**，约束“适配后标签”不要离原始标签太远，否则模型学习会变得困难。([arXiv][1])

角色：

* 对付的是**短期、局部的“异常”波动**（例如情绪过度反应），希望在训练时“稍微拉回正常”，提高泛化能力。([Scribd][6])

### 4.2 模型适配器 Model Adapter

目标：为每个增量任务提供一个**更适合当前分布的初始化**，让基础预测模型在少量增量数据上更快收敛。([arXiv][5])

形式上：

1. 对于第 (k) 个任务，有基础模型参数 (\theta_{k-1})；
2. 模型适配器 (M_\psi) 利用当前任务的信息（通常是梯度方向、历史参数等）生成适配后的初始化 (\tilde{\theta}*k = M*\psi(\theta_{k-1}, \text{task info}))；
3. 再从 (\tilde{\theta}_k) 出发，用增量训练集进行若干步梯度下降，得到最终参数 (\theta_k)。

实现上基于 **MAML/在线 meta-learning 的一阶近似**：

* 利用 `facebookresearch/higher` 库对训练过程做可微“展开”，在上层计算近似的元梯度，而不显式求高阶梯度，降低计算量。([GitHub][2])

### 4.3 双层优化流程

#### (1) 下层：普通增量更新（Inner Loop）

对每个任务 (\mathcal{T}_k)：

1. 用 DA/MA 变换 (D^{\text{train}}_k) 与初始化参数，得到适配后的训练数据和模型；
2. 在训练集上做几步梯度下降，得到任务专属的 (\theta_k)；
3. 在 (D^{\text{test}}_k) 上预测并计算损失（MSE 或 IC 相关的 proxy loss）。([arXiv][1])

#### (2) 上层：元学习更新 DA / MA（Outer Loop）

* 将各任务测试损失加总为元损失；
* 通过一阶近似方法，反向传播到 DA/MA 的参数：

  * 数据适配器参数：影响输入/标签的变换；
  * 模型适配器参数：影响各任务初始参数选择；
* 更新 DA/MA，使**经过“先适配再训练”的模型，在未来任务上表现更好**。([arXiv][1])

在推理部署阶段，可继续在线微调 DA/MA（学习率较小），也可以固定 DA、只微调 MA 等多种策略；附录中给出了学习率敏感性分析，表明**持续更新 MA 对性能提升尤为重要**。([arXiv][1])

---

## 五、与 qlib 的关系与工程实现

### 1. qlib 中的集成位置

* DoubleAdapt 的作者原本在 **microsoft/qlib** 中实现了增量学习与 DoubleAdapt 脚本，并通过 PR 引入 `examples/benchmarks_dynamic`（如 PR #1560），包含：

  * 基础增量学习（IL）；
  * DDG-DA；
  * DoubleAdapt 等方法。([GitHub][3])
* 随后为了降低耦合，团队在 **SJTU-DMTai/DoubleAdapt** 仓库中重构了代码，并在 README 中说明，将 `main.py` 放入 `examples/benchmarks_dynamic/incremenetal/` 目录，可以直接基于 qlib 数据运行实验。([GitHub][2])

### 2. 工程侧关键点（来自官方 README）

官方 README 给了几条实用建议：([GitHub][2])

1. **IL + RR 联合使用**

   * 论文中 IL 和 Rolling Retraining（RR）是对比方法，但在实践里两者是正交的；
   * 建议：每月 **从头用全部历史数据 RR 训练一遍 DoubleAdapt**，然后在月内每 2–3 个交易日用 DoubleAdapt 做一次增量适配。

2. **重设计数据适配器**

   * 实验主要在 **Alpha360（6 维因子）** 上进行，DA 采用的是 (6\times 6) 的全连接仿射变换；
   * 对于常见的几百因子（如 Alpha158），这类 FC 层会严重过参数化、表现变差；
   * 建议：

     * 按因子分组，在组内学习小矩阵变换；
     * 或先将因子嵌入到低维，再在 embedding 空间做轻量变换（如 element-wise normalizing flow 等）。

3. **学习率与步长调参**

   * 下层学习率 `lr` & 上层 `lr_da`、`lr_ma` 需要独立 grid search；
   * 在线阶段可用 `--online_lr "{'lr': 0.0005, 'lr_da': 0.0001, 'lr_ma': 0.0005}"` 这种形式单独设置。

4. **step 与 horizon 的关系**

   * `horizon` 决定标签是 (\text{Ref}(\text{close}, -horizon-1)/\text{Ref}(\text{close}, -1) - 1)；最近 `horizon` 天的真实收益是未知的，无法参与元学习；
   * 因此推荐 `step` 至少比 `horizon` 大 3–4，例如 `--step 5 --horizon 1`；
   * 当前实现不支持 `step <= horizon` 的在线训练。([GitHub][2])

5. **资源开销**

   * `step=20` 且 CSI500 时，约需 8GB RAM、10GB GPU；减少 `step` 可以降低显存并提升性能，但 RR/ DDG-DA 会变得很慢。([GitHub][2])

---

## 六、实证表现：相对 IL、RR 与 DDG-DA 的优势

### 1. 论文原实验（KDD 2023）

数据与设置：([arXiv][5])

* 市场：

  * CSI300（大盘蓝筹）
  * CSI500（中小盘）
* 特征：Alpha360（6 维价量因子 + 60 日历史窗），后续也测了 Alpha158。
* 对比方法：

  * **RR 系列**：Rolling Retraining，含 DDG-DA（生成不同历史分布样本做重加权）、RR-FT 等；
  * **IL 系列**：Naive IL（只对最新增量数据做 OGD）、C-MAML、MetaCoG 等元学习增量方法。

关键结论（以 CSI300 + GRU 为例）：

* **整体 SOTA**：摘要和结论中明确指出，DoubleAdapt 在 IC/ICIR/Rank IC/Rank ICIR 与投资回测（超额年化收益和 IR）上**均达到或超过现有方法**，并且相比 RR 系列具有更高效率。([arXiv][1])
* **消融实验（Table 2）**：

  * 只用数据适配（IL+DA）已明显优于 DDG-DA，在 CSI300 上 **IC 提升约 5.6%**；
  * 只用模型适配（IL+MA）优于 C-MAML，IC 提升约 3.3%；
  * 同时使用 DA+MA 的 **完整 DoubleAdapt** 在所有评价指标上表现最佳。([arXiv][1])

从这些结果可以看出：

* 与传统 RR 相比，DoubleAdapt 在保持或提升收益的同时，大幅降低了反复全量训练的时间开销；
* 与只做模型适配的 C-MAML / MetaCoG 等相比，多了“数据适配”这一维度，在存在剧烈短期波动时更稳健。

### 2. 后续工作中的再评估（以 MetaDA 为例）

2024 年的 MetaDA（Incremental Learning of Stock Trends via Meta-Learning with Dynamic Adaptation）在同样的 CSI300 / CSI500 + Alpha360 设置下，对包括 DoubleAdapt 在内的多种方法进行了统一复现：([arXiv][7])

* 在该论文 Table 1 中（以 GRU 为例）：

  * DoubleAdapt 在 CSI300 / CSI500 上的 **IC 约 0.057–0.058**，
  * 对应的投资指标（超额年化收益的 IR，eARIR）在 **2.5–3.1 左右**，
  * 明显优于 IL、DDG-DA、C-MAML、MetaCoG 等基线；
  * 但整体略低于它们提出的 MetaDA（后者在 eARIR 上有小幅提升）。

这说明：

* 在后续工作中，DoubleAdapt 仍然被视作**强基线**，且在统一复现实验中依旧保持领先于大量经典方法。
* 更新的方法（如 MetaDA）是在 DoubleAdapt 的思路基础上进一步扩展（例如引入“可预见的概念漂移”建模）而获得进一步提升。([arXiv][7])

---

## 七、优势与局限性分析

### 1. 优势

1. **针对增量量化场景量身定制**

   * 直接面向“定期再训练 + 流式数据”的股票趋势预测任务设计，而不是从通用 CL/Meta-learning 概念硬套。([arXiv][1])

2. **数据适配 + 模型适配的双重视角**

   * 多数方法要么只改模型（初始化、正则等），要么只改采样方式（DDG-DA 的重加权）；
   * DoubleAdapt 在**输入空间和参数空间两端**同时做适配，兼顾了**分布漂移**和**快速收敛**两类问题。([arXiv][1])

3. **元学习带来的“任务间迁移”**

   * 用 IL 任务序列来做 meta-train / meta-test，把“如何适配下一批增量数据”本身当作学习目标；
   * 在真实市场中，虽然分布在变，但“变的方式”本身具有一定共性，这正是元学习捕捉的对象。([arXiv][1])

4. **计算效率相对可控**

   * 相对于频繁 rolling retrain 全量模型，DoubleAdapt 只在增量数据上训练，同时重用 meta-learned 的适配器；
   * 利用一阶近似避免高阶梯度计算，使得在 CSI300 / CSI500 级别的数据规模上可以在合理时间内跑完多次实验。([arXiv][1])

5. **工程实践经验相对充足**

   * 官方仓库提供了详细的 `step/horizon`、在线学习率、内存与显存占用等实测经验，以及如何在 qlib 中挂载运行的说明，对落地较友好。([GitHub][2])

### 2. 局限性与风险点

1. **原始数据适配器设计与高维因子不匹配**

   * DA 最初是为 Alpha360（仅 6 个因子）设计；
   * 在 Alpha158 这类上百维因子上，全连接 (d\times d) 变换会严重过参数化；
   * 官方 README 明确指出，在 Alpha158 上原始 DA 会“**表现次优**”，并建议重新设计（分组、小型变换、embedding 级别变换等）。([GitHub][2])

   这意味着：

   * 在实际高维因子体系中，直接照搬论文的 DA，效果有一定不确定性；
   * 更合理的做法是把 DoubleAdapt 看成一个“框架”和“训练范式”，而针对特定因子体系重写 DA。

2. **对“可预见概念漂移”的处理有限**

   * DoubleAdapt 强调使用**最近数据**进行适配，假设“未来趋势主要由最近模式决定”；([arXiv][1])
   * 但后续的 MetaDA 等工作指出：DoubleAdapt **只用最新数据进行适配**，在某些“可预测的长周期漂移”场景下仍有不足，需要利用选取的历史数据来建模“任务间结构”。([arXiv][7])

3. **元学习超参敏感，调参成本较高**

   * DoubleAdapt 同时涉及：

     * 下层学习率 `lr`；
     * 上层 DA / MA 学习率 `lr_da`、`lr_ma`；
     * 任务间隔 `step` 与预测期 `horizon`；
     * head 数量、softmax 温度、正则权重等；
   * 附录中的实验表明：

     * softmax 温度太小（近似 one-hot）会显著拉低表现；
     * 正则过大则抑制标签适配带来的收益；
     * 冻结 meta-learner 在线不更新时，性能明显下降。([arXiv][1])

   换言之，这套框架在工程上**并非“开箱即用”**，需要耐心的实验调参。

4. **资源与实现复杂度**

   * 作者给出的参考：`step=20`、CSI500 时约需 8GB RAM、10GB GPU，且若希望更频繁更新（`step=5`），RR 和 DDG-DA 的时间成本很高，实验三天起步。([GitHub][2])
   * 实现依赖 `higher` 做可微训练展开，对工程代码质量和调试能力有一定要求。

5. **项目层面：与 qlib 的整合状态**

   * DoubleAdapt 的主实现目前在 SJTU-DMTai/DoubleAdapt 仓库中维护，与 qlib 主项目存在一定“松耦合”；
   * qlib 仓库中的增量学习脚本部分以 PR 形式存在，合并与更新节奏明显落后于学术与社区侧的演进，这在工程侧需要自行判断使用哪一版代码。([GitHub][2])

---

## 八、实践落地时可参考的使用策略

结合论文与官方 README 的建议，一个较务实的使用范式可以概括为：([GitHub][2])

1. **先以 GRU + Alpha360/Alpha158 做小规模验证**

   * 使用 CSI300 / CSI500 crowd 数据做复现实验；
   * 在 `step` 从 5–20、`horizon` 从 1–5 的组合上进行网格搜索；
   * 重点看 IC/ICIR 与回测 eAR/eARIR 的稳定性。

2. **结合 Rolling Retraining**

   * 每月或每季度全量 retrain DoubleAdapt 一次，把 DA/MA 与基础模型都训练到新数据上；
   * 月内按 2–3 个交易日为间隔，运行 DoubleAdapt 增量更新；
   * 在资源允许的情况下，将上述频率再适当提升。

3. **针对自有因子体系重写 Data Adapter**

   * 对几十~几百因子的场景，优先考虑：

     * 分组仿射：同一风格/行业/技术因子共用一个小变换矩阵；
     * 或 embedding 级变换：先投影到较低维 embedding，再做 element-wise / channel-wise 适配；
   * 用 qlib /自建框架评估不同 DA 带来的 IC / eAR 变化，逐步修剪冗余结构。

4. **将 DoubleAdapt 视作“元学习引擎”，而非固定黑盒模型**

   * 模型适配器（MA）这一部分具有相当通用性，可以与 GRU / LSTM / Transformer / GNN 等多种结构组合；
   * 在一些更复杂的图结构或多模态场景中，可以保留 DoubleAdapt 的“训练范式”（任务级双层优化），而替换掉基础模型与数据适配器。

---

### 小结

在当前公开文献与实际复现结果中，DoubleAdapt 可以视为**面向股票增量趋势预测的标志性元学习方法**之一：

* 在 Alpha360 + CSI300/CSI500 的标准设置下，相比 RR 与多种 meta-learning 基线普遍获得更高的 IC/ICIR 与收益指标；([arXiv][1])
* 思路清晰：把分布漂移拆为**输入/标签空间的变形**与**参数空间的适配**两条线，共同通过元学习跨任务优化；
* 同时，也清楚暴露了高维因子场景下数据适配器设计、超参敏感、资源开销等实际问题，后续工作（如 MetaDA）正是在这些不足基础上进一步改进。

从方法论角度来看，更合理的定位是：

> DoubleAdapt 提供了一套“在非平稳金融时间序列上应用元学习 + 双重适配”的**可行范式**，适合作为后续方法和自定义系统的基础框架，而不是一套无需改动即可直接搬进任何因子体系的终极方案。

[1]: https://arxiv.org/html/2306.09862v3 "DoubleAdapt: A Meta-learning Approach to Incremental Learning for Stock Trend Forecasting"
[2]: https://github.com/SJTU-DMTai/DoubleAdapt "GitHub - SJTU-DMTai/DoubleAdapt: The official API of DoubleAdapt (KDD'23), an incremental learning framework for online stock trend forecasting, WITHOUT dependencies on the qlib package."
[3]: https://github.com/microsoft/qlib/pull/1560/files "Release codes of incremental learning and DoubleAdapt ..."
[4]: https://arxiv.org/html/2306.09862v3 "DoubleAdapt: A Meta-learning Approach to Incremental ..."
[5]: https://arxiv.org/pdf/2306.09862 "DoubleAdapt: A Meta-learning Approach to Incremental ..."
[6]: https://www.scribd.com/document/918837608/DoubleAdapt-a-Meta-learning-Approach-to-Incremental-Learning-for-Stock-Trend-Forecasting "DoubleAdapt A Meta-Learning Approach To Incremental ..."
[7]: https://arxiv.org/html/2401.03865v3 "Incremental Learning of Stock Trends via Meta-Learning with Dynamic Adaptation"
