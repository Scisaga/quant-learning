# TFT（Temporal Fusion Transformers）基准

## 参考来源

- 论文：Lim, Bryan, et al. *Temporal fusion transformers for interpretable multi-horizon time series forecasting.* arXiv:1912.09363（2019）
- 参考代码：https://github.com/google-research/google-research/tree/master/tft

本目录提供 TFT 在 Qlib `Alpha158` 数据集上的基准 workflow 配置。

## 运行方式

在仓库根目录执行：

```bash
cd src/examples/benchmarks/TFT
pip install -r requirements.txt
qrun workflow_config_tft_Alpha158.yaml
```

## 注意事项（非常重要）

1. 该脚本/依赖组合通常只支持 `Python 3.6 - 3.7`（建议使用独立 Conda 环境）。
2. 模型需要在 GPU 上运行；若没有 GPU，可能会直接报错或耗时极长。
3. 若你的 CUDA 版本不是 10.0，可参考以下命令安装对应运行时（按你的环境调整）：
   - `conda install anaconda cudatoolkit=10.0`
   - `conda install cudnn`
4. 若要接入新数据集，需要在 `data_formatters/` 中注册 formatter；细节请参考上游 TFT 项目说明。
