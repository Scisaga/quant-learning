# Sandwich

- 参考实现（FOST）：https://github.com/microsoft/FOST/blob/main/fostool/model/sandwich.py

本目录提供 Sandwich 在 Qlib `Alpha360` 数据集上的基准 workflow 配置。

## 文件说明

- `requirements.txt`：本模型额外依赖（通常包含 PyTorch 等）。
- `workflow_config_sandwich_Alpha360.yaml`：`qrun` 配置文件。

## 运行方式

在仓库根目录执行：

```bash
cd backend/qlib/examples/benchmarks/Sandwich
pip install -r requirements.txt
qrun workflow_config_sandwich_Alpha360.yaml
```

## 备注（关于依赖/显卡）

- FOST 原始实现中会使用 `torch_geometric`；本仓库的基准实现 **未强制依赖** 它。
- 若你希望使用 GPU，请确保 CUDA 与 PyTorch 版本匹配；本配置在 `CUDA==10.2` + `torch==1.12.1` 的环境下做过测试。
