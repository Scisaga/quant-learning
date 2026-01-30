# 组合优化策略（Portfolio Optimization）

## 背景

在 `src/examples/benchmarks/` 中，我们提供了多种 **alpha** 模型用于预测股票收益，并使用一个简单的规则策略 `TopkDropoutStrategy` 来评估投资表现。

但 TopK 类策略过于简单，难以显式控制组合风险（例如相关性、波动率、跟踪误差等）。

因此，本示例展示如何使用基于优化的策略 `EnhancedIndexingStrategy`：在尽可能提高组合收益的同时，控制相对基准的跟踪误差（tracking error）。

## 准备工作

示例默认使用中国市场数据（CSI300）。

### 1) 准备 CSI300 权重数据

```bash
wget https://github.com/SunsetWolf/qlib_dataset/releases/download/v0/csi300_weight.zip
unzip -d ~/.qlib/qlib_data/cn_data csi300_weight.zip
rm -f csi300_weight.zip
```

> 说明：公开免费的基准权重数据资源较少，因此这里使用了社区提供的数据包；你也可以替换为自己的权重数据源。

### 2) 准备风险模型数据

```bash
python src/examples/portfolio/prepare_riskdata.py
```

本示例使用 `qlib.model.riskmodel` 中的 **统计风险模型**。更高质量的生产级风控通常建议使用：

- 基本面风险模型（例如 MSCI BARRA）
- 深度风险模型（例如 https://arxiv.org/abs/2107.05201）

## 端到端运行

完成准备后，可直接运行：

```bash
qrun src/examples/portfolio/config_enhanced_indexing.yaml
```

该配置与 `src/examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml` 的主要差异集中在 strategy 部分（用优化策略替换简单 TopK 策略）。
