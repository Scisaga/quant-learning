# TRA（Temporal Routing Adaptor）+ Optimal Transport

Temporal Routing Adaptor（TRA）用于在股票市场数据中捕捉多种交易模式，并通过“路由 + 最优传输”机制提升对异质模式的建模能力。更多细节请参考论文：

- 论文：Learning Multiple Stock Trading Patterns with Temporal Routing Adaptor and Optimal Transport（KDD 2021）  
  http://arxiv.org/abs/2106.12950

如果你的研究使用了该工作，建议引用：

```bibtex
@inproceedings{HengxuKDD2021,
 author = {Hengxu Lin and Dong Zhou and Weiqing Liu and Jiang Bian},
 title = {Learning Multiple Stock Trading Patterns with Temporal Routing Adaptor and Optimal Transport},
 booktitle = {Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery \\& Data Mining},
 series = {KDD '21},
 year = {2021},
 publisher = {ACM},
}

@article{yang2020qlib,
  title={Qlib: An AI-oriented Quantitative Investment Platform},
  author={Yang, Xiao and Liu, Weiqing and Zhou, Dong and Bian, Jiang and Liu, Tie-Yan},
  journal={arXiv preprint arXiv:2009.11189},
  year={2020}
}
```

## 推荐用法（与 Qlib workflow 集成）

`TRA` 已迁移到 `qlib.contrib.model.pytorch_tra`，以更好支持 `qlib.workflow` 与 `Alpha158/Alpha360` 等基准数据集。建议按官方 workflow 文档方式使用：  
https://qlib.readthedocs.io/en/latest/component/workflow.html

本目录提供可直接 `qrun` 的示例配置：

- `workflow_config_tra_Alpha360.yaml`：在 `Alpha360` 上运行 TRA
- `workflow_config_tra_Alpha158.yaml`：在 `Alpha158` 上运行 TRA（带特征子采样）
- `workflow_config_tra_Alpha158_full.yaml`：在 `Alpha158` 上运行 TRA（不做特征子采样）

在仓库根目录执行（以 Alpha158 为例）：

```bash
cd src/examples/benchmarks/TRA
pip install -r requirements.txt
qrun workflow_config_tra_Alpha158.yaml
```

TRA 的基准对比结果可参考 `src/examples/benchmarks/README.md`（表格中包含 TRA 行）。

## 论文复现用法（旧脚本，可能不再维护）

本节用于复现论文中的旧版运行方式（与当前 Qlib workflow 的组织形式不同）。相关脚本与配置位于：

- `run.sh`：论文复现脚本入口
- `configs/`：旧版配置
- `example.py`：可通过命令行传参运行旧版配置

两种运行方式：

1. 使用 `qrun` 直接运行旧版配置（示例）：

```bash
qrun configs/config_alstm.yaml
```

2. 使用代码运行并自定义参数（示例）：

```bash
python example.py --config_file configs/config_alstm.yaml
```

> 说明：论文复现流程中，TRA 会基于一个预训练的 backbone，因此通常需要先运行 `*_init.yaml` 再运行 TRA 的脚本。

### 论文报告的指标（qlib==0.7.1）

运行后结果文件通常在 `./output`：

- `info.json`：配置与指标
- `log.csv`：训练日志
- `model.bin`：模型参数
- `pred.pkl`：预测结果

论文中报告的结果如下（供对照）：

| Methods | MSE | MAE | IC | ICIR | AR | AV | SR | MDD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Linear | 0.163 | 0.327 | 0.020 | 0.132 | -3.2% | 16.8% | -0.191 | 32.1% |
| LightGBM | 0.160(0.000) | 0.323(0.000) | 0.041 | 0.292 | 7.8% | 15.5% | 0.503 | 25.7% |
| MLP | 0.160(0.002) | 0.323(0.003) | 0.037 | 0.273 | 3.7% | 15.3% | 0.264 | 26.2% |
| SFM | 0.159(0.001) | 0.321(0.001) | 0.047 | 0.381 | 7.1% | 14.3% | 0.497 | 22.9% |
| ALSTM | 0.158(0.001) | 0.320(0.001) | 0.053 | 0.419 | 12.3% | 13.7% | 0.897 | 20.2% |
| Trans. | 0.158(0.001) | 0.322(0.001) | 0.051 | 0.400 | 14.5% | 14.2% | 1.028 | 22.5% |
| ALSTM+TS | 0.160(0.002) | 0.321(0.002) | 0.039 | 0.291 | 6.7% | 14.6% | 0.480 | 22.3% |
| Trans.+TS | 0.160(0.004) | 0.324(0.005) | 0.037 | 0.278 | 10.4% | 14.7% | 0.722 | 23.7% |
| ALSTM+TRA(Ours) | 0.157(0.000) | 0.318(0.000) | 0.059 | 0.460 | 12.4% | 14.0% | 0.885 | 20.4% |
| Trans.+TRA(Ours) | 0.157(0.000) | 0.320(0.000) | 0.056 | 0.442 | 16.1% | 14.2% | 1.133 | 23.1% |

更详细的实验报告示例可参考 `Reports.ipynb`。

## 常见问题

- 若遇到 loss 为 `NaN`，请重点检查 sinkhorn 算法中的 `epsilon` 参数；`epsilon` 的量级需要与输入尺度匹配。
- 其他问题建议提交 Issue（或在本仓库的讨论区反馈）。
