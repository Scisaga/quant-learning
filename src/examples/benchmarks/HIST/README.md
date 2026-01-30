# HIST（概念共享信息挖掘的图模型）

- 论文：HIST: A Graph-based Framework for Stock Trend Forecasting via Mining Concept-Oriented Shared Information（https://arxiv.org/abs/2110.13716）
- 参考代码：https://github.com/Wentao-Xu/HIST

本目录提供 HIST 在 Qlib `Alpha360` 数据集上的基准 workflow 配置。

## 文件说明

- `requirements.txt`：本模型额外依赖。
- `workflow_config_hist_Alpha360.yaml`：`qrun` 配置文件。
- `qlib_csi300_stock_index.npy`：示例所需的辅助数据（若配置中引用了它，请不要删除）。

## 运行方式

在仓库根目录执行：

```bash
cd src/examples/benchmarks/HIST
pip install -r requirements.txt
qrun workflow_config_hist_Alpha360.yaml
```
