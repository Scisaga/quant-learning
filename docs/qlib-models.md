## 一、传统机器学习 / 表格模型

https://github.com/microsoft/qlib/tree/main/examples/benchmarks

### 1. Linear（线性模型）

* **大致年代**：线性回归可以追溯到 19–20 世纪早期，这里就是普通线性打分模型。
* **核心思想**：
  对特征做线性加权求和，假设“未来收益”与因子是线性关系，等价于经典 Barra 风格多因子回归。
* **在 Qlib 中的表现（CSI300 / Alpha158）**：

  * IC ≈ 0.04、ICIR ≈ 0.30，年化收益 ≈ 6.9%，IR ≈ 0.92，表现其实不弱，说明 **高质量因子 + 线性模型** 已能挖出有价值信号。
* **评价**：

  * 优点：简单、稳健、易解释，是很好的 baseline。
  * 缺点：只能建模线性关系，对非线性、交互项无能为力；在 Alpha360 这种“原始价量 + 少工程”的场景中就明显不够用了。

### 2. MLP（多层感知机）

* **提出年代**：多层感知机在 1980s 被系统化推广（早期感知机 ~1958）。
* **核心思想**：
  多层全连接网络 + 非线性激活，能够表示复杂的非线性映射，是最朴素的“深度因子”模型。
* **Qlib Alpha158 表现**：

  * IC ≈ 0.0376，ICIR ≈ 0.28；年化收益 ≈ 8.95%，IR ≈ 1.14，**收益率和 IR 都很亮眼**，接近甚至超过部分树模型。
* **评价**：

  * 在因子工程已经很强（Alpha158）的场景里，MLP 是一个性价比很高的非线性基线。
  * 但对时序结构仅靠“窗口展平”，在 Alpha360 这种时序结构更重要的数据上通常不如时序专用模型。

### 3. XGBoost（Tianqi Chen, et al.）

* **年份**：2016 年 KDD 论文提出。
* **核心思想**：
  高效的梯度提升树（GBDT）实现，支持稀疏特征、正则化与分布式训练，是 Kaggle 时代的“万能模型”。
* **Qlib Alpha158 表现**：

  * IC ≈ 0.0498、ICIR ≈ 0.38；年化收益 ≈ 7.8%，IR ≈ 0.91。各项指标稳健，仅略低于 LightGBM / DoubleEnsemble。
* **评价**：

  * 对**人工设计因子**非常友好；对少量、高质量特征时表现好、调参成熟。
  * 在原始价量（Alpha360）这类强时序、强局部结构数据上，XGBoost 难以利用时序依赖，表现相对一般。

### 4. LightGBM（Guolin Ke, et al.）

* **年份**：2017 年提出。
* **核心思想**：
  GBDT 的高效实现，采用 GOSS（梯度采样）、叶子优先生长和高效直方统计，在大规模数据上有更好速度与效果。
* **Qlib Alpha158 表现**：

  * IC ≈ 0.0448、ICIR ≈ 0.37；年化收益 ≈ 9.0%，IR ≈ 1.02。
* **评价**：

  * Alpha158 上是**非常强的基线**，收益和稳定性都很优秀。
  * 对于 Alpha360，依然缺少显式时序建模，与深度时序模型相比会处于劣势。

### 5. CatBoost（Liudmila Prokhorenkova, et al.）

* **年份**：2018 年提出。
* **核心思想**：
  对类别特征特别友好的 GBDT，使用“有序目标编码”等技术减轻 target leakage 和过拟合。
* **Qlib Alpha158 表现**：

  * IC ≈ 0.0481；年化收益 ≈ 7.65%，IR ≈ 0.80，整体略低于 XGBoost/LightGBM。
* **评价**：

  * 在有大量类别特征的信用评分等任务中优势更明显；在 Alpha158 这种以数值因子为主的数据上，优势不突出。
  * 总体是一个稳定、略次于 LightGBM/XGBoost 的树模型 baseline。

### 6. TabNet（Sercan O. Arik, et al.）

* **年份**：2019 年（arXiv 预印本，后续 ICLR/AAAI 发表）。
* **核心思想**：
  对表格数据设计的深度网络，使用**顺序注意力**在每一层选择子集特征，并通过稀疏正则提升可解释性。
* **Qlib 表现**：

  * Alpha158：IC ≈ 0.0204，ICIR 较低，年化收益和 IR 都明显落后树模型。
  * Alpha360：官方表格显示 IC 仍然偏低，收益一般。
* **评价**：

  * 理论上很适合同一时刻“宽表格”的任务；但在金融高噪声、强时序的数据上，效果未必优于传统 GBDT。
  * 在 Qlib 的设定下，TabNet 工作量大但收益有限，更适合作为“表格 SOTA 对比项”，而不是实用首选。

### 7. DoubleEnsemble（Chuheng Zhang, et al.）

* **年份**：2020 年（arXiv 2010.01265）。
* **核心思想**：
  专门为金融数据设计的**双重集成框架**：
  1）基于学习轨迹的样本重加权，突出“关键样本”；
  2）基于洗牌的特征选择，筛掉弱因子，以提高稳定性和抗噪声能力。底层基模型在 Qlib 中为 LightGBM。
* **Qlib Alpha158 表现**：

  * IC ≈ 0.0521（全表最高）、ICIR ≈ 0.42；年化收益 ≈ 11.6%，IR ≈ 1.34，**全面碾压其它表格模型**。
* **评价**：

  * 是 Alpha158 场景下最强的公开基线之一，非常契合“高噪声 + 多因子”的量化环境。
  * 算法更复杂、训练开销也更大，更适合离线因子挖掘和高投入研究场景。

---

## 二、经典序列模型（RNN/CNN/Transformer/TFT）

### 8. LSTM（Sepp Hochreiter, et al.）

* **年份**：1997 年提出。
* **核心思想**：
  通过输入门、遗忘门、输出门缓解长序列的梯度消失问题，是经典的“记忆型”循环单元。
* **Qlib Alpha158 表现**：

  * 使用 20 维精选因子，IC ≈ 0.0318，年化收益 ≈ 3.8%，整体略好于 GRU，弱于 SFM/ALSTM 等针对金融优化的模型。
* **评价**：

  * 结构稳定，但在价格/因子这类中短期序列上未必有明显优势。
  * 更适合作为“深度时序 baseline”，而不是最优方案。

### 9. GRU（Kyunghyun Cho, et al.）

* **年份**：2014 年提出。
* **核心思想**：
  删除了 LSTM 的输出门，只保留重置门和更新门，参数更少、计算更快，在很多任务中效果接近或优于 LSTM。
* **Qlib Alpha158 表现**：

  * 同样在 20 因子子集上，IC ≈ 0.0315，收益略低于 LSTM。
* **评价**：

  * 对金融 Alpha 任务来说，单纯 GRU/LSTM 都不是最强解，但可以配合更高级策略（如 TCTS、AdaRNN）增强。

### 10. TCN（Temporal Convolutional Network, Shaojie Bai, et al.）

* **年份**：2018 年（“An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling”）。
* **核心思想**：
  使用**因果卷积 + 扩张卷积**构建一维卷积网络，实现长依赖建模，同时训练更并行稳定。
* **Qlib Alpha158 表现**：

  * IC ≈ 0.028，年化收益 ≈ 2.6%，弱于 SFM/ALSTM/TRA 等专为金融设计的时序模型。
* **评价**：

  * 架构优雅、训练高效，是一个好的通用时序基线。
  * 在高度噪声的金融数据中，如果没有额外结构（图、注意力、任务调度），性能很容易被更金融化的模型超越。

### 11. Transformer（Ashish Vaswani, et al.）

* **年份**：2017 年（“Attention Is All You Need”）。
* **核心思想**：
  纯注意力架构，自注意力捕获远距离依赖，已成为 NLP 和时序建模的基础架构。
* **Qlib 表现**：

  * Alpha158：IC ≈ 0.026、ICIR ≈ 0.21，收益 ~2.7%，显著弱于树模型和金融专用模型。
  * Alpha360：IC≈0.011，年化收益为负，IR < 0。
* **评价**：

  * 原始 Transformer 在金融场景中**并非“开箱即用的 SOTA”**；对噪声高、信号弱的数据，需要更多结构化约束。
  * 在 Qlib 中更适合作为“通用深度模型对照组”，而不是实战首选。

### 12. TFT（Temporal Fusion Transformer, Bryan Lim, et al.）

* **年份**：2019 年（IJF 论文）。
* **核心思想**：
  针对多步时间序列预测设计的 Transformer 变体，引入门控残差、可解释注意力、静态/动态变量融合。
* **Qlib Alpha158 表现**：

  * IC ≈ 0.036，但 Rank IC 很低、收益波动大，IR 中等，整体不如许多更简单的模型。
* **评价**：

  * 在“宏观多变量 + 多步预测”任务中很有竞争力；但在 Qlib 的 Alpha 任务里，其额外结构未能充分转化为收益优势。

### 13. SFM（State Frequency Memory, Liheng Zhang, et al.）

* **年份**：2017 年（KDD 论文，专门用在股价预测上）。
* **核心思想**：
  把时间序列分解到多个“频段”记忆（state-frequency），对应不同时间尺度的交易模式。
* **Qlib Alpha158 表现**：

  * IC ≈ 0.038，ICIR ≈ 0.30；收益≈4.6%，整体优于普通 RNN/TCN。
* **评价**：

  * 对金融数据“多周期共存”的假设非常贴合，属于**早期专为股价建模设计的网络**，性能也印证了这一点。

### 14. ALSTM（Attention-based LSTM, Yao Qin, et al.）

* **年份**：2017 年（DA-RNN / 注意力 LSTM 用于时序预测）。
* **核心思想**：
  在 LSTM 上加入注意力（对历史时间步或对输入特征），提升对关键片段的关注度。
* **Qlib Alpha158 表现**：

  * IC ≈ 0.036，年化收益 ≈ 4.7%，IR ≈ 0.70 左右，比纯 LSTM/GRU 略好。
* **评价**：

  * 简单、实用，是“RNN+注意力”范式的代表。
  * 不过在 Qlib 里已经被 SFM、TRA 等更复杂模型超越。

---

## 三、图注意力 / 局部结构模型

### 15. GATs（Graph Attention Networks, Petar Veličković, et al.）

* **年份**：2017/2018 年（ICLR 2018）。
* **核心思想**：
  在图上对邻居做注意力加权，学习节点表达。Qlib 中通常用行业、相似性等构建股票图。
* **Qlib Alpha158 表现**：

  * IC ≈ 0.035，Rank IC ≈ 0.046，收益 ≈ 5%，IR ≈ 0.73，整体优于基础 RNN/TCN，但略逊于 TRA / DoubleEnsemble。
* **评价**：

  * 能利用“股票之间的关系”（行业、相关性），对多标的任务很自然。
  * 真正发挥优势时，往往需要更精心设计的图结构和正则。

### 16. Localformer（Juyong Jiang, et al.）

* **年份**：Localformer 本身更偏工程命名，公开论文信息较少，大致在 2020s 之后在推荐/时序场景被广泛引用。
* **核心思想**：
  主要思想是**局部注意力**：与其对全序列做全局 Self-Attention，不如聚焦局部片段（local window），提高效率并减少噪声干扰。
* **Qlib Alpha158 表现**：

  * IC ≈ 0.0356，ICIR ≈ 0.28；收益 ≈ 4.4%，略优于 SFM/ALSTM，属于“中上游”深度模型。
* **评价**：

  * 适合在长序列、局部模式明显的金融数据上使用，比完全全局的 Transformer 更稳。
  * 在 Qlib 中，表现不错，但仍被 LightGBM/DoubleEnsemble 等传统强化模型压制。

---

## 四、金融专用 / 高级序列模型

### 17. TRA（Temporal Routing Adaptor, Hengxu Lin, et al.）

* **年份**：2021 年，KDD 2021 论文“Learning Multiple Stock Trading Patterns with Temporal Routing Adaptor”。
* **核心思想**：
  针对股票序列的**多模式路由**：通过一个“路由器”把时间步/样本分配给不同的子专家，从而学习多种交易模式。
* **Qlib 表现**：

  * Alpha158：

    * 在“20 特征”版本和“全特征”版本中，TRA 的 IC、年化收益都比多数常规深度模型更高，年化收益可达 ~7% 以上，IR ~1。
  * Alpha360：

    * 在官方表格中，TRA 属于 Alpha360 上表现较好的模型之一，IC 和收益明显优于 Transformer/TabNet 等通用模型。
* **评价**：

  * 是**“专为金融时序模式挖掘”**的代表作之一，在两种数据集上都比大多数通用结构更强。
  * 模型复杂度相比简单 RNN/GBDT 较高，工程落地需要更多算力和调参。

### 18. TCTS（Temporally Correlated Task Scheduling, Xueqing Wu, et al.）

* **年份**：2021 年（ICML 论文）。
* **核心思想**：
  提出**任务调度器**：在训练序列模型时动态选择不同辅助任务的顺序，使与主任务“时间相关”的辅助任务优先学习，从而改善泛化。Qlib 中基于 GRU 作为基模型。
* **Qlib Alpha360 表现**：

  * 官方基准显示，在 Alpha360 上，TCTS 的 IC/ICIR 明显高于纯 GRU/Transformer，收益也更可观。
* **评价**：

  * 思路偏“训练策略 + 时序 meta-learning”，而不是纯结构创新。
  * 对“多任务 + 序列”金融场景具有实践参考价值。

### 19. AdaRNN（Adaptive RNN, Yuntao Du, et al.）

* **年份**：2021 年（CIKM，arXiv:2108.04443）。
* **核心思想**：
  针对**Temporal Covariate Shift（时间协变量漂移）**提出：

  * 先用“Temporal Distribution Characterization”将时间划分为若干分布不同的时期；
  * 再用“Temporal Distribution Matching”在这些时期之间对 RNN 表征做分布对齐。
* **Qlib Alpha360 表现**：

  * 在 Alpha360 这类分布变化明显的原始价量数据上，AdaRNN 的 IC/IR 和收益都明显高于基础 GRU。
* **评价**：

  * 思想非常贴近金融市场“regime shift / regime change”，在 OOD 和分布漂移场景中具有理论与实战价值。
  * 模型和训练流程相对复杂，更适合研究型/高投入策略。

### 20. ADD（Qlib 中的 ADD 模型）

* **年份**：Qlib 中的 ADD 作为代码模型出现在 2020s 左右，与多任务/多模式金融序列相关；公开文献中对其确切对应论文不算统一。
* **核心思想（根据 Qlib 文档与相关综述推断）**：

  * 一般是通过多子模型 + 特定训练策略，强化在关键子样本 / 子任务上的表现，以改善极低信噪比场景。
* **Qlib Alpha360 表现**：

  * 在 Alpha360 表格中表现处于中上水平，优于基础 GRU/LSTM，但略低于 AdaRNN / IGMTF / HIST 等最新模型。
* **评价**：

  * 可以视为 Qlib 团队在“多模型 + 任务调度/重加权”方向上的一次探索，为后续更强模型提供了经验。

### 21. IGMTF（Instance-wise Graph-based Multi-task Framework, Wentao Xu, et al.）

* **年份**：2021 年左右（IGMTF 相关工作发表于 2021 年，针对多任务时序预测）。
* **核心思想**：

  * 把每个实例（如一只股票在某一日期）看成图上的节点，利用图结构刻画实例间关系；
  * 结合多任务学习框架，针对不同预测任务共享/专用表征。
* **Qlib Alpha360 表现**：

  * 在 Alpha360 基准中，IGMTF 的 IC 和年化收益接近表格顶部，与 HIST、Sandwich 等同属“最强一档”。
* **评价**：

  * 强调“实例间关系 + 多任务”的组合，比单一序列模型更适合跨股票、多标签的量化任务。
  * 模型结构复杂、实现成本较高，更偏研究型 SOTA。

### 22. HIST（Hybrid Stock Trend model, Wentao Xu, et al.）

* **年份**：2021 年（“HIST: A Hybrid Model for Stock Trend Forecasting” 类型工作）。
* **核心思想**：

  * 顾名思义是“混合模型”：将多种结构（如 CNN/RNN/注意力/图结构）组合，用以同时捕捉时间依赖、跨股票关系和高阶模式。
* **Qlib Alpha360 表现**：

  * Alpha360 的 Qlib 表格中，HIST 的 IC 和年化收益处于全模型最前列之一，IR 较高，最大回撤也可接受。
* **评价**：

  * 在 Qlib 现有公开基准中，可以把 HIST 视为**Alpha360 场景里的旗舰模型**之一。
  * 代价是工程复杂度极高，对数据、算力和实现细节要求也更严格。

### 23. KRNN（Qlib 专用模型）

* **年份**：KRNN 为 Qlib 仓库中的自定义模型，大致在 2020s 加入；公开论文相对少见，多为工程实现。
* **核心思想（根据命名和 Qlib 文档推断）**：

  * 一般被理解为在 RNN 上融合 kernel / kNN 之类的结构，使模型更好地利用样本间相似性。
* **Qlib Alpha360 表现**：

  * 在 Alpha360 基准表中，KRNN 的表现优于基础 RNN/Transformer，但略逊于 IGMTF/HIST/Sandwich 等最新架构。
* **评价**：

  * 适合用作“中间层次”的高阶模型：结构比简单 RNN 强，但又不像 HIST/IGMTF 那么重。

### 24. Sandwich（Qlib 专用模型）

* **年份**：Sandwich 同样是 Qlib 内部命名的高阶模型，出现在近期量化深度学习工作中。
* **核心思想（从相关综述推断）**：

  * 通过“上下夹层”的多模型结构（如 GBDT + 深度网络 + 更多正则）来兼顾不同类型特征与模式，因此得名 “Sandwich”。
* **Qlib Alpha360 表现**：

  * 在 Alpha360 上，Sandwich 的 IC/年化收益与 IGMTF/HIST 接近，属于公开基准中最强的一批。
* **评价**：

  * 体现了“多模型混合 + 金融先验”思想，对生产环境的可解释性与稳定性有潜在优势，但实现难度也最大。

---

## 五、整体对比与选择建议（结合 Qlib benchmarks）

结合 Qlib 官方 CSI300 基准（Alpha158 / Alpha360），可以得出几个结构化结论：

1. **在高质量因子场景（Alpha158）**

   * 传统 **GBDT 系列 + DoubleEnsemble** 仍是最强

     * DoubleEnsemble > LightGBM ≈ XGBoost > MLP ≈ TRA > Linear > 其它深度模型。
   * 深度时序/图模型（SFM、ALSTM、GATs、Localformer、TRA）能提供增量收益，但整体**并没有颠覆 GBDT**。
   * 说明：当因子工程足够好时，“简单模型 + 好因子”仍然极具竞争力。

2. **在原始价量场景（Alpha360）**

   * 通用模型（Transformer、TabNet、纯 RNN/TCN/MLP）表现普遍一般甚至亏损。
   * 专为金融设计的**AdaRNN、TCTS、TRA、IGMTF、HIST、Sandwich、KRNN 等**明显更强，IC 与年化收益显著领先。
   * 这反映了：

     * 需要显式建模 **分布漂移（AdaRNN）**、
     * 利用 **图结构与多任务（IGMTF/HIST）**、
     * 或在训练过程中进行 **任务调度与样本/特征重加权（TCTS/ADD/DoubleEnsemble）**，
       才能真正挖掘出原始价量中的微弱 alpha。

3. **模型选择的实践启示**

   * 若已有成熟因子库（类似 Alpha158）：

     * 首选 LightGBM / XGBoost / DoubleEnsemble；
     * 深度模型主要作为补充或做“非线性修正层”。
   * 若要直接从原始价量挖信号（类似 Alpha360）：

     * 不宜只依赖通用 Transformer/TCN/MLP；
     * 更有价值的是采用 **“金融专用 + 多模型混合 + 任务调度/图结构”** 的路线（AdaRNN、TRA、IGMTF、HIST、Sandwich 等），并结合自身数据特点做裁剪。
