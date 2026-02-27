# IGMTF（实例级图多变量时序预测）

- 论文：IGMTF: An Instance-wise Graph-based Framework for Multivariate Time Series Forecasting（https://arxiv.org/abs/2109.06489）
- 参考代码：https://github.com/Wentao-Xu/IGMTF

本目录提供 IGMTF 在 Qlib `Alpha360` 数据集上的基准 workflow 配置。

## 文件说明

- `requirements.txt`：本模型额外依赖。
- `workflow_config_igmtf_Alpha360.yaml`：`qrun` 配置文件。

## 运行方式

在仓库根目录执行：

```bash
cd backend/qlib/examples/benchmarks/IGMTF
pip install -r requirements.txt
qrun workflow_config_igmtf_Alpha360.yaml
```
