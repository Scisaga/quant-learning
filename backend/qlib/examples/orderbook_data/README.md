# orderbook_data（订单簿/逐笔等非固定频率数据示例）

本示例演示 Qlib 如何支持 **非固定频率（non-fixed-frequency）** 数据。

例如：

- 日频 OHLCV 属于固定频率数据（每天一条或固定采样间隔）。
- 订单/委托/成交等数据是事件驱动的，可能在任意时间点出现，属于非固定频率数据。

为支持这类数据，Qlib 提供了基于 Arctic 的后端。本目录给出一个示例：如何导入数据、以及如何查询并计算一些高频特征。

## 安装依赖

1. 安装 MongoDB（示例脚本默认连接 `localhost` 的默认端口，且 **不带鉴权**）。请参考 MongoDB 官方安装文档：  
   https://docs.mongodb.com/manual/installation/

2. 安装 Python 依赖：

```bash
pip install pytest coverage gdown
pip install arctic  # 注意：某些环境下 pip 可能无法自动解析到正确依赖，请自行确认依赖版本兼容
```

## 导入示例数据

1. （可选）准备 Qlib 的 1min 数据  
   如果你的后续示例需要参考 Qlib 的 1min 数据，可先按仓库的数据准备说明获取 1min 数据。

2. 下载订单簿示例数据并解压：

```bash
cd backend/qlib/examples/orderbook_data
gdown https://drive.google.com/uc?id=15FuUqWn2rkCi8uhJYGEQWKakcEqLJNDG  # 可能需要代理
python ../../scripts/get_data.py _unzip --file_path highfreq_orderbook_example_data.zip --target_dir .
```

3. 导入数据到 MongoDB：

```bash
python create_dataset.py initialize_library
python create_dataset.py import_data
```

## 查询与计算示例

数据导入完成后，可以运行 `example.py` 来计算一些高频特征（使用 pytest 运行便于按用例选择）：

```bash
pytest -s --disable-warnings example.py
pytest -s --disable-warnings example.py::TestClass::test_exp_10
```

## 已知限制

- 不同频率之间的表达式计算（expression computing between different frequencies）目前尚不支持。
