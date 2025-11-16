# 动态基准测试

https://github.com/microsoft/qlib/tree/main/examples/benchmarks_dynamic

用来研究 随时间滚动训练/预测时的“概念漂移”（concept drift）问题

## 朴素滚动训练基线策略

`baseline/` 目录提供的是一组 **“不带复杂元学习”的传统动态训练策略**，核心脚本是 `rolling_benchmark.py`，用于做 **rolling backtest / rolling training**：
在每个时间滚动窗口上重新训练或更新模型，然后在未来一段时间内预测并回测。

相关 issue 中明确提到：

* `rolling_benchmark.py` 中设置：

  * 滚动步长 `step = 20` 天。
  * 预测长度 `horizon = 20` 天。
  * 在每一轮滚动里，对接下来 20 天进行逐日再平衡（每天调仓）。

这说明 baseline 是典型的 **“固定窗口/滚动窗口 + 固定再平衡频率”** 的动态训练框架。

### 策略类型

1. **Static / One-shot 模型**

   * 只在第一次训练期内训练一次模型，之后不再更新。
   * 作为“完全不适应概念漂移”的下界基线。

2. **Rolling Retraining（滚动重训）**

   * 每滚动一个窗口，使用 **最近一段时间的数据** 重新训练模型（窗口宽度固定）。
   * 在 Qlib 社区与博文中通常称为 **Rolling Retrain（RR）**，被当成 DDG-DA 的主要对照基线之一。

3. **Expanding Window（扩展窗口） / Incremental Refit 的变体**

   * 每一步滚动时，将训练集扩展到包含 **从起始至当前时刻的所有历史数据**。
   * 适合认为“老数据也始终有价值”的情景。

4. **Exponential Forgetting / Decay（指数遗忘）**

   * 仍使用全部历史数据，但对样本施加 **时间衰减权重**，越新的数据权重越大。
   * 在动态基准的介绍与 DDG-DA 教程中，被明确列作对照策略（即“仅靠时间衰减而不预测未来分布”的方法）。

这些 baseline 通常是通过 **Qlib 的 Rolling 训练接口** + **统一的基础预测模型（如 Linear / LightGBM）** 来实现。核心关注点不是“模型结构”，而是“训练数据如何随时间更新”。

> * 优点：简单、直观、工程实现成本低，几乎适用于任何监督模型。
> * 缺点：
>
>   * 只根据“过去已经发生的数据”做适应，对 **未来分布的可预测模式** 没有显式建模；
>   * 在概念漂移较强、又存在周期或趋势的场景中，往往会滞后，无法主动“提前适应”未来市场状态。

## DDG-DA：AAAI 2022 概念漂移元模型

`DDG-DA`（Data Distribution Generation for Predictable Concept Drift Adaptation）是 2022 年在 AAAI 发表的工作，旨在解决 **“可预测的概念漂移”** 问题：

* 传统做法：

  * 先检测漂移是否发生，再把模型适配到最近的数据分布（如 rolling retrain / forgetting）。
* DDG-DA 做的事：

  * 认为在很多真实场景（如金融市场）中，**数据分布的变化本身也存在一定规律**；
  * 因此尝试 **“预测未来的数据分布”**，而不是只适配“当前最新分布”。

### 具体实现方式

1. **构造 Meta Task & Meta Dataset**

   * 从历史滚动过程（多个时间片、多个市场状态）中抽取“任务级信息”（如某一时间窗口的特征分布、标签分布等），形成一组 Meta Tasks；
   * 所有 Meta Tasks 组成 Meta Dataset。

2. **训练 DDG-DA 元模型**

   * 元模型的任务：给定“一个时间片的历史信息”，输出“如何对历史样本加权重采样”，从而生成一个 **预测的未来数据分布**。

3. **在未来滚动时做推断（Inference）**

   * 对每个新的时间片，DDG-DA 根据当前的元信息预测未来的数据分布，并利用 **权重重采样（weighted resampling）** 生成新的训练集。

4. **用“预测分布生成的训练集”训练基础预测模型**

   * 比如 Linear / LightGBM 等，然后再做常规回测。
   * 这一步和 baseline 的训练流程兼容，只是训练数据经过了 DDG-DA 的“再加权”。

### 代码实现


* `workflow.py`：主脚本，封装上述四步，调用 Qlib 的 `MetaGuideModel` 接口。
* `requirements.txt`：依赖（PyTorch、Qlib 等）。
* 若干 `workflow_config_*.yaml`：

  * 例如 `workflow_config_lightgbm_Alpha158.yaml`，用于指定：

    * 基础数据集（Alpha158/Alpha360，cn_data 或 crowd_data）。
    * 基础预测模型（Linear / LightGBM / 其他）。
    * 训练/验证/测试时间区间、滚动参数等。

运行形式通常为：

```bash
python workflow.py --conf_path=../workflow_config_lightgbm_Alpha158.yaml run
```

> * DDG-DA 属于 **“模型无关（model-agnostic）” 的元方法**：
>
>   * 不改变基础预测模型结构，可以套在 Linear、LightGBM 等任意模型外层；
>   * 通过预测未来分布 + 再权重采样，使基础模型在训练时“看到更像未来的数据”。
> * 相比 baseline 的“被动适应”，DDG-DA 尝试 **主动预测漂移方向**，理论和实验上在可预测漂移场景中更有优势。

### 评价

DDG-DA[LightGBM] + Alpha158
- IC: 0.0878 / ICIR: 0.6185 / Rank IC: 0.0975 / Rank ICIR: 0.6524
- Annualized Return: 0.1261 / Information Ratio: 2.0096 / Max Drawdown: 	-0.0744

1. **信号层面（IC / ICIR / Rank IC / Rank ICIR）**

   * baseline 策略（Static / Rolling / Exp Forget 等）在 Alpha158/Alpha360 上通常能达到 **中等水准的 IC（约 0.06–0.09 量级）**；
   * DDG-DA 在相同基础模型（尤其是 Linear）条件下，**IC 和 Rank IC 有持续抬升**，ICIR 也相应提高，说明：

     * 不仅相关性更高，而且在不同时间窗口上更稳定（波动更小）。

2. **收益层面（Annualized Return / Information Ratio）**

   * 使用 DDG-DA 产生的信号执行同样的回测流程，**年化收益和信息比率普遍高于 baseline**；
   * DDG-DA 相对于 Rolling Retrain / Exponential Forget 的 **超额收益更为显著**。

3. **计算代价与工程权衡**

   * 官方 release notes 里提到，DDG-DA 在更细的滚动步长（如 `step=5`）下，**时间和空间开销不小**，这也是后来引入 DoubleAdapt 等更高效增量学习方法的一个背景。
   * 在实际使用中，若硬件资源有限，baseline（简单 Rolling）依然是非常实用的起点；在此基础上再考虑是否引入 DDG-DA 等更复杂方法。