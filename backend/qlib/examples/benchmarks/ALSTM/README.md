# ALSTM（带注意力的 LSTM）

- 论文：A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction（IJCAI 2017）  
  https://www.ijcai.org/Proceedings/2017/0366.pdf

ALSTM 在普通 LSTM 的基础上增加了时间注意力聚合层，用于对历史序列的不同时间步赋予不同权重。

> 说明：本仓库中的实现是 **简化版 ALSTM**（可视为“带注意力的 LSTM”），用于基准对比与复现流程。

## 文件说明

- `requirements.txt`：本模型额外依赖。
- `workflow_config_alstm_Alpha158.yaml`：`Alpha158` 数据集配置。
- `workflow_config_alstm_Alpha360.yaml`：`Alpha360` 数据集配置。

## 运行方式

在仓库根目录执行（以 Alpha158 为例）：

```bash
cd backend/qlib/examples/benchmarks/ALSTM
pip install -r requirements.txt
qrun workflow_config_alstm_Alpha158.yaml
```

切换数据集只需替换 YAML：

```bash
qrun workflow_config_alstm_Alpha360.yaml
```

## 输出

- 默认写入 `mlruns/`（MLflow）。
- 指标含义与对比表见 `backend/qlib/examples/benchmarks/README.md`。
