# 动态市场适应（benchmarks_dynamic）

金融市场环境具有明显的非平稳性（non-stationary）：不同时间段的数据分布会发生变化（concept drift），导致“用过去训练的模型”在未来测试期的表现逐步衰减。因此，如何让预测模型/策略适应市场动态变化非常重要。

本目录汇总了 concept drift 场景下的基准结果与可运行示例：

- `baseline/`：周期性滚动重训（RR, Rolling Retrain）基线
- `DDG-DA/`：DDG-DA（可预测 concept drift 的分布生成式适应）

## 数据准备（推荐：crowd-sourced 数据）

部分方案（尤其是 DDG-DA）对字段完整性较敏感。推荐使用 crowd-sourced 版本的 Qlib 数据（包含 `VWAP` 等字段）：  
https://github.com/chenditc/investment_data/releases

示例下载命令：

```bash
wget https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz
mkdir -p ~/.qlib/qlib_data/cn_data
tar -zxvf qlib_bin.tar.gz -C ~/.qlib/qlib_data/cn_data --strip-components=2
rm -f qlib_bin.tar.gz
```

## Alpha158 数据集结果（示例）

下表展示了不同方案在不同预测模型上的表现（示例表格）：

| Model Name       | Dataset | IC | ICIR | Rank IC | Rank ICIR | Annualized Return | Information Ratio | Max Drawdown |
|------------------|---------|------|------|---------|-----------|-------------------|-------------------|--------------|
| RR[Linear]       |Alpha158 |0.0945|0.5989|0.1069   |0.6495     |0.0857             |1.3682             |-0.0986       |
| DDG-DA[Linear]   |Alpha158 |0.0983|0.6157|0.1108   |0.6646     |0.0764             |1.1904             |-0.0769       |
| RR[LightGBM]     |Alpha158 |0.0816|0.5887|0.0912   |0.6263     |0.0771             |1.3196             |-0.0909       |
| DDG-DA[LightGBM] |Alpha158 |0.0878|0.6185|0.0975   |0.6524     |0.1261             |2.0096             |-0.0744       |

实验设置说明：

- `Alpha158` 的 label horizon 设置为 20。
- rolling 重训间隔（step）设置为 20 个交易日。
- 测试滚动区间为 2017/01 - 2020/08。
- 结果基于 crowd-sourced 版本数据生成。
  - Yahoo 版本 Qlib 数据不包含 `VWAP`，因此相关因子会缺失并被填 0，导致矩阵秩亏（rank-deficient），从而使 DDG-DA 的底层优化问题可能无法求解。

## 快速运行

RR（Rolling Retrain）基线（在仓库根目录执行）：

```bash
python backend/qlib/examples/benchmarks_dynamic/baseline/rolling_benchmark.py run
```

DDG-DA 示例（硬件需求更高；建议先阅读其 README）：

```bash
python backend/qlib/examples/benchmarks_dynamic/DDG-DA/workflow.py run
```
