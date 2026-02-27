# TCTS（时序相关任务调度）

TCTS（Temporally Correlated Task Scheduling for Sequence Learning）通过“可学习的任务调度器（scheduler）”在训练过程中动态选择一组 **时间相关的辅助任务**，来帮助主任务更好收敛并提升泛化效果。

## 背景

在序列学习中，一个主任务往往伴随多个时间相关任务：例如股票收益预测可以同时预测 `t+1`、`t+2`、`t+3` 等不同 horizon 的收益。如何在训练中“用哪些任务、何时用”会显著影响最终效果。

## 方法概述

TCTS 的关键是引入一个可学习的 task scheduler：

- scheduler 会根据模型当前状态与当前训练数据（例如 minibatch）选择一个合适的辅助任务；
- scheduler 与主模型通过 **双层优化（bi-level optimization）** 联合训练：scheduler 以验证集表现为目标，主模型以训练 loss 为目标；
- 过程示意图见下方。

<p align="center">
<img src="workflow.png"/>
</p>

在第 <img src="https://latex.codecogs.com/png.latex?s" title="s" /> 步，给定训练数据 <img src="https://latex.codecogs.com/png.latex?x_s,y_s" title="x_s,y_s" />，scheduler <img src="https://latex.codecogs.com/png.latex?\varphi" title="\varphi" /> 选择任务 <img src="https://latex.codecogs.com/png.latex?T_{i_s}" title="T_{i_s}" />（绿色实线）用于更新主模型 <img src="https://latex.codecogs.com/png.latex?f" title="f" />（蓝色实线）。每经过 <img src="https://latex.codecogs.com/png.latex?S" title="S" /> 步，会在验证集评估主模型并更新 scheduler（绿色虚线）。

## 关于实验设置（两套版本）

由于 **数据版本** 与 **Qlib 版本** 的差异，论文中的原始实验设置与当前 Qlib 基准存在不同。因此常见会有两套代码/设置：

1. 可复现论文结果的代码：https://github.com/lwwang1995/tcts
2. 当前 Qlib 基线实现：https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/pytorch_tcts.py

### Setting 1（论文复现版本）

- 数据集：CSI300 的 300 只股票，2008/01/01 - 2020/08/01；按时间切分 train/valid/test。
- 主任务 <img src="https://latex.codecogs.com/png.latex?T_k" title="T_k" />：预测股票 <img src="https://latex.codecogs.com/png.latex?i" title="i" /> 的收益：
  <div align=center>
  <img src="https://latex.codecogs.com/png.image?\dpi{110}&space;r_{i}^{t,k}&space;=&space;\frac{price_i^{t&plus;k}}{price_i^{t&plus;k-1}}-1" title="r_{i}^{t,k} = \frac{price_i^{t+k}}{price_i^{t+k-1}}-1" />
  </div>
- 时间相关任务集合 <img src="https://latex.codecogs.com/png.latex?\mathcal{T}_k" title="\mathcal{T}_k" />：例如 <img src="https://latex.codecogs.com/png.latex?\mathcal{T}_3" title="\mathcal{T}_3" />、<img src="https://latex.codecogs.com/png.latex?\mathcal{T}_5" title="\mathcal{T}_5" />、<img src="https://latex.codecogs.com/png.latex?\mathcal{T}_{10}" title="\mathcal{T}_{10}" /> 分别用于不同的主任务 horizon。

### Setting 2（Qlib 基线版本）

- 数据集：同样基于 CSI300 的 300 只股票，但 train/valid/test 切分不同。
- 主任务定义：
  <div align=center>
  <img src="https://latex.codecogs.com/png.image?\dpi{110}&space;r_{i}^{t,k}&space;=&space;\frac{price_i^{t&plus;1&plus;k}}{price_i^{t&plus;1}}-1" title="r_{i}^{t,k} = \frac{price_i^{t+1+k}}{price_i^{t+1}}-1" />
  </div>
- Qlib 基线中通常使用 <img src="https://latex.codecogs.com/png.latex?\mathcal{T}_3" title="\mathcal{T}_3" /> 来辅助 <img src="https://latex.codecogs.com/png.latex?T_1" title="T_1" />。

## 运行方式（本仓库基线）

本目录提供 `workflow_config_tcts_Alpha360.yaml`。在仓库根目录执行：

```bash
cd backend/qlib/examples/benchmarks/TCTS
pip install -r requirements.txt
qrun workflow_config_tcts_Alpha360.yaml
```

## 结果与参考

- Setting 1 的实验结果见论文：http://proceedings.mlr.press/v139/wu21e/wu21e.pdf
- Setting 2（Qlib baseline）的对比表见：`backend/qlib/examples/benchmarks/README.md`
