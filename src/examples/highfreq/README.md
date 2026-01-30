# highfreq（高频数据示例）

本目录包含两类示例：

1. **高频 Dataset 示例**：展示如何在 Qlib 中使用高频数据集（面向 RL 高频交易等场景）。
2. **高频价格趋势预测基准**：给出一个在高频数据上做趋势/方向预测的基准结果示例（可持续扩展更多模型）。

## 目录内容

- `workflow.py`：示例入口（CLI 风格）。
- `workflow_config_High_Freq_Tree_Alpha158.yaml`：高频树模型（GBDT 风格）基准 workflow 配置（`qrun`）。
- `highfreq_handler.py` / `highfreq_processor.py` / `highfreq_ops.py`：高频数据处理相关实现与算子。

## 高频 Dataset 示例

### 1) 获取高频数据

示例脚本提供了获取数据的入口（具体依赖于你的网络/数据源配置）：

```bash
python src/examples/highfreq/workflow.py get_data
```

### 2) dump / load / reinit（序列化与重初始化）

示例中高频 Dataset 基于 `qlib.data.dataset.DatasetH` 实现，并继承自 `qlib.utils.serial.Serializable`，因此可以用 `pickle` 进行序列化/反序列化。

Qlib 还支持在从磁盘加载 Dataset 后进行 **重初始化（reinit）**：例如重设 `instruments`、`start_time`、`end_time`、`segments` 等状态，并据此重新生成数据。

运行示例：

```bash
python src/examples/highfreq/workflow.py dump_and_load_dataset
```

## 高频趋势预测基准（Tree / Alpha158）

本目录提供了一个可直接运行的 `qrun` 配置：`workflow_config_High_Freq_Tree_Alpha158.yaml`。

注意：该配置默认使用 1min 数据目录 `~/.qlib/qlib_data/cn_data_1min`（见 YAML 的 `qlib_init.provider_uri`）。请先确保你已准备好 1min 数据。

运行方式（在仓库根目录执行）：

```bash
qrun src/examples/highfreq/workflow_config_High_Freq_Tree_Alpha158.yaml
```

## 基准结果（示例）

以下为“高频价格趋势预测”示例的基准结果（会持续更新）：

| Model Name | Dataset | IC | ICIR | Rank IC | Rank ICIR | Long precision | Short Precision | Long-Short Average Return | Long-Short Average Sharpe |
|---|---|---|---|---|---|---|---|---|---|
| LightGBM | Alpha158 | 0.0349±0.00 | 0.3805±0.00 | 0.0435±0.00 | 0.4724±0.00 | 0.5111±0.00 | 0.5428±0.00 | 0.000074±0.00 | 0.2677±0.00 |
