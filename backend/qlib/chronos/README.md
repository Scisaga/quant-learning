# Chronos-2 落地代码（qchronos）

本目录实现了 `docs/chronos-2.md` 中“信号层 / 策略层 / 校准层”的关键工程部件，目标是：

- 把 Chronos-2 的分位数预测结果**后处理**为：区间、阈值概率、score
- 把 score 接入 Qlib 风格评测：IC/ICIR（信号层）与回测（策略层）
- 提供可人工验证的**中间字段**（raw/cal 同时保留），并用充足中文注释解释每一步做了什么

> 注意：本仓库当前 `requirements.txt` 未包含 `chronos-forecasting`/`mapie`/`autogluon.timeseries`。
> 本目录代码对这些依赖做了“可选导入”，你可以按需安装。

## 目录结构

- `backend/qlib/chronos/qchronos/`：核心库（避免与第三方 `chronos` 包名冲突）
  - `features.py`：past-only 特征 + known-future 日历特征 + regime 标签
  - `postprocess.py`：分位数 → 区间/概率 → score（核心后处理）
  - `gating.py`：静态/动态门控（策略规则）
  - `calibration.py`：区间（conformal）与概率（isotonic/Platt）校准
  - `qlib_adapter.py`：导出 Qlib 需要的 `pred_df` / `label_df`
  - `chronos2_infer.py`：Chronos-2 推理包装（可选依赖：`chronos-forecasting`）
- `backend/qlib/chronos/scripts/`：可运行脚本（建议从这里开始）

## 快速上手（建议）

1) 先用脚本对 **后处理/校准**做“纯数学自测”（不需要 Chronos-2 模型）：

```bash
python backend/qlib/chronos/scripts/self_check_postprocess.py
```

2) 若你已安装 Chronos-2 推理包：

```bash
pip install "chronos-forecasting>=2.0"
```

再尝试跑一个最小 PoC（需要你准备 `context_df` / `future_df`，见脚本注释）：

```bash
python backend/qlib/chronos/scripts/run_chronos2_poc.py
```

3) 若你使用 Qlib 数据源，可在脚本里按注释接入 `qlib.init(...)`，导出 `pred_df/label_df` 后复用现有 Qlib workflow。

