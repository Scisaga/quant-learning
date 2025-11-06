## 安装

python 3.12

```shell
pip uninstall -y gym
pip install -U gymnasium
pip install -U "pyqlib[all]"

# 验证
python -c "import qlib; print(qlib.__version__)"

# 下载A股日线数据
python src/get_data.py qlib_data --target_dir data/qlib_data/cn_data --region cn

# 下载A股分钟线数据
python src/get_data.py qlib_data --target_dir data/qlib_data/cn_data_1min --region cn --interval 1min

# 下载美股数据
python src/get_data.py qlib_data --target_dir data/qlib_data/us_data --region us

# 安装训练库，关闭warning
pip install "pyqlib==0.9.7" "lightgbm==4.4.0" "xgboost==2.1.4" "catboost==1.2.8"
pip install "torch>=2.2,<3" --index-url https://download.pytorch.org/whl/cpu
```

## [初始化](https://qlib.readthedocs.io/en/latest/start/initialization.html)

```py
import qlib
from qlib.constant import REG_CN

provider_uri = "data/qlib_data/cn_data"  # 数据目录
qlib.init(provider_uri=provider_uri, region=REG_CN)
```

参数详解：
- **provider_uri**（str）：数据 URI（如 get_data.py 存储路径）。必需，与 region 对齐。
- **region**（str，默认 REG_CN）：市场模式。
  - REG_CN ('cn')：中股（trade_unit=100, limit_threshold=0.099）。
  - REG_US ('us')：美股（trade_unit=1, 无 limit）。
  - 影响：交易单位/涨跌停/费用。仅快捷配置，可手动覆盖 trade_unit/limit_threshold。
- **redis_host**（str，默认 "127.0.0.1"）：Redis 主机，用于锁/缓存机制。
- **redis_port**（int，默认 6379）：Redis 端口。连接失败 → 无缓存（详见 Cache）。
- **exp_manager**（dict）：实验管理器配置，用于 Recorder（实验跟踪）。示例：
  ```python
  exp_manager = {
      "class": "MLflowExpManager",
      "module_path": "qlib.workflow.expm",
      "kwargs": {
          "uri": "mlruns",  # 跟踪 URI
          "default_exp_name": "Experiment"
      }
  }
  qlib.init(..., exp_manager=exp_manager)
  ```
  - 支持 MLflow 等；详见 Recorder: Experiment Management。
- **mongo**（dict）：MongoDB 配置，用于 Task Management（高性能/集群处理）。需先安装 MongoDB。示例：
  ```python
  mongo = {
      "task_url": "mongodb://localhost:27017/",  # URL，支持凭证如 "mongodb://user:pwd@host:port"
      "task_db_name": "rolling_db"  # 数据库名
  }
  qlib.init(..., mongo=mongo)
  ```
- **logging_level**：日志级别。
- **kernels**：表达式引擎计算特征的进程数。调试异常时设为 1。

## [工作流管理](https://qlib.readthedocs.io/en/latest/component/workflow.html)

- 核心功能：Qlib 组件松耦合设计，用户可自定义量化流程。qrun 接口提供更友好方式：基于 YAML 配置自动运行整个工作流（execution），包括：
  - **数据（Data）**：加载、处理、切片。
  - **模型（Model）**：训练、推理、保存/加载。
  - **评估（Evaluation）**：信号分析、回测。
- 追踪机制：每个 execution 自动记录训练/推理/评估信息及产物（artifacts），详见 Recorder: Experiment Management。
- YAML 标准化流程，`qrun` 一键执行 + 自动记录；松耦合，便于换模型/数据集/策略。
  - 相对手写：省 boilerplate 代码，提升复现性/效率；集成 Recorder 追踪实验。
  - 适用场景：批量实验、策略回测；与 Task Management 结合支持分布式。
  - 生态扩展：与 Recorder/Task Management/Online Serving 闭环。

### 配置 YAML

LightGBM + Alpha158 + TopkDropout 策略
```yaml
# Qlib Workflow 配置示例：定义整个量化研究流程，包括数据、模型、评估。
# 使用 qrun configuration.yaml 运行。
# 注意：YAML 中的 & 定义锚点，便于复用参数。

# Qlib 初始化部分：设置全局参数，用于 Qlib 初始化。
qlib_init:
    # provider_uri: Qlib 数据的 URI，例如 get_data.py 存储的数据路径。
    provider_uri: "~/.qlib/qlib_data/cn_data"
    # region: 市场区域，'cn' 表示中国模式，'us' 表示美国模式，必须与数据对齐。
    region: cn

# market: 定义市场锚点，例如 csi300（沪深300），用于后续复用。
market: &market csi300
# benchmark: 定义基准锚点，例如 SH000300（上证指数），用于后续复用。
benchmark: &benchmark SH000300

# data_handler_config: 数据处理器配置锚点，定义数据时间范围和仪器列表。
data_handler_config: &data_handler_config
    # start_time: 数据整体起始时间。
    start_time: 2008-01-01
    # end_time: 数据整体结束时间。
    end_time: 2020-08-01
    # fit_start_time: 拟合（训练）起始时间。
    fit_start_time: 2008-01-01
    # fit_end_time: 拟合（训练）结束时间。
    fit_end_time: 2014-12-31
    # instruments: 仪器列表，使用 market 锚点。
    instruments: *market

# port_analysis_config: 组合分析配置锚点，包括策略和回测参数。
port_analysis_config: &port_analysis_config
    # strategy: 策略配置。
    strategy:
        # class: 策略类名。
        class: TopkDropoutStrategy
        # module_path: 策略在 Qlib 中的模块路径。
        module_path: qlib.contrib.strategy.strategy
        # kwargs: 策略参数。
        kwargs:
            # topk: 选择前 K 只股票。
            topk: 50
            # n_drop: 每期丢弃的股票数。
            n_drop: 5
            # signal: 信号来源，<PRED> 表示模型预测。
            signal: <PRED>
    # backtest: 回测配置。
    backtest:
        # start_time: 回测起始时间。
        start_time: 2017-01-01
        # end_time: 回测结束时间。
        end_time: 2020-08-01
        # account: 初始账户金额。
        account: 100000000
        # benchmark: 基准指数，使用 benchmark 锚点。
        benchmark: *benchmark
        # exchange_kwargs: 交易所参数。
        exchange_kwargs:
            # limit_threshold: 涨跌停阈值。
            limit_threshold: 0.095
            # deal_price: 成交价格字段，例如 close（收盘价）。
            deal_price: close
            # open_cost: 开仓佣金率。
            open_cost: 0.0005
            # close_cost: 平仓佣金率。
            close_cost: 0.0015
            # min_cost: 最低佣金。
            min_cost: 5

# task: 任务定义，包括模型、数据集和记录器。
task:
    # model: 模型配置。
    model:
        # class: 模型类名，例如 LGBModel（LightGBM）。
        class: LGBModel
        # module_path: 模型在 Qlib 中的模块路径。
        module_path: qlib.contrib.model.gbdt
        # kwargs: 模型参数。
        kwargs:
            # loss: 损失函数，例如 mse（均方误差）。
            loss: mse
            # colsample_bytree: 列采样率。
            colsample_bytree: 0.8879
            # learning_rate: 学习率。
            learning_rate: 0.0421
            # subsample: 行采样率。
            subsample: 0.8789
            # lambda_l1: L1 正则化。
            lambda_l1: 205.6999
            # lambda_l2: L2 正则化。
            lambda_l2: 580.9768
            # max_depth: 最大树深。
            max_depth: 8
            # num_leaves: 叶子节点数。
            num_leaves: 210
            # num_threads: 线程数。
            num_threads: 20
    # dataset: 数据集配置。
    dataset:
        # class: 数据集类名，例如 DatasetH。
        class: DatasetH
        # module_path: 数据集在 Qlib 中的模块路径。
        module_path: qlib.data.dataset
        # kwargs: 数据集参数。
        kwargs:
            # handler: 数据处理器配置。
            handler:
                # class: 数据处理器类名，例如 Alpha158（158 个 alpha 因子）。
                class: Alpha158
                # module_path: 数据处理器在 Qlib 中的模块路径。
                module_path: qlib.contrib.data.handler
                # kwargs: 使用 data_handler_config 锚点。
                kwargs: *data_handler_config
            # segments: 数据切片段。
            segments:
                # train: 训练段。
                train: [2008-01-01, 2014-12-31]
                # valid: 验证段。
                valid: [2015-01-01, 2016-12-31]
                # test: 测试段。
                test: [2017-01-01, 2020-08-01]
    # record: 记录器列表，用于跟踪训练结果和评估。
    record:
        # 第一个记录器：信号记录。
        - class: SignalRecord
          # module_path: 记录器在 Qlib 中的模块路径。
          module_path: qlib.workflow.record_temp
          # kwargs: 参数（此处为空）。
          kwargs: {}
        # 第二个记录器：组合分析记录。
        - class: PortAnaRecord
          # module_path: 记录器在 Qlib 中的模块路径。
          module_path: qlib.workflow.record_temp
          # kwargs: 参数。
          kwargs:
              # config: 使用 port_analysis_config 锚点。
              config: *port_analysis_config
```

最小运行策略：
* 准备数据：运行 `get_data.py` 下载/存储数据到 provider_uri。
* 编写 YAML：复制完整示例，调整参数（如 market、model kwargs）。
* 执行：`qrun your_config.yaml`（输出 Recorder 记录）。
* 调试/扩展：用 pdb 调试；集成 Task Management 批量运行。
* 参考：examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml。

## [Data Layer 数据层](https://qlib.readthedocs.io/en/latest/component/data.html)

Data Layer 是 Qlib 的数据管理核心，提供用户友好 API 和高性能基础设施。典型流程：数据准备 → API 检索 → Loader/Handler 处理 → Dataset 建模 → 缓存优化。

- **核心功能**：管理/检索金融数据，支持公式化 alpha（e.g., 通过表达式引擎构建 "Ref($close, 60) / $close"）。组件包括：准备、API、Loader、Handler、Dataset、Cache。
- **流程解读**：数据从原始源加载（OHLCV），经表达式引擎生成基本特征（如 60 日回报），Handler 处理复杂操作（如归一化），Dataset 准备模型输入。缓存加速重复访问。
- **优势**：为量化设计，高性能文件存储（.bin 格式，详见 Qlib 论文 File storage design）。支持实时处理。
- **适用场景**：回测、因子研究；集成 Workflow/Task Management。

### 数据准备（Data Preparation）
- **Qlib 格式数据**：.bin 文件，专为金融科学计算设计。Qlib 提供预置数据集（Alpha360/Alpha158，美/中市场）。
- **下载命令**：
  - 日频（中）：`python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn`
  - 1min（中）：`python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/qlib_cn_1min --region cn --interval 1min`
  - 日频（美）：`python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/us_data --region us`
- **高频示例**：参考链接运行。
- **价格调整**：Qlib 将首日价格归一化为 1；用 $factor 恢复原价（e.g., $close / $factor）。
- **自动更新**（日频）：手动更新一次后，用 crontab 定时（e.g., `python collector.py update_data_to_bin --qlib_data_1d_dir <dir>`）。手动：指定 trading_date/end_date。
- **CSV 转 .bin**：用 `scripts/dump_bin.py` 转换。CSV 需含股票名/日期列，支持 –symbol_field_name/–date_field_name。
  - 示例：`python scripts/dump_bin.py dump_all --csv_path ~/.qlib/csv_data/my_data --qlib_dir ~/.qlib/qlib_data/my_data --include_fields open,close,high,low,volume,factor`
  - 必需列：open/close/high/low/volume/factor（调整因子）。
- **健康检查**：用 `scripts/check_data_health.py` 检查缺失/异常跳变。示例：调整阈值如 missing_data_num/large_step_threshold。
- **股票池**：导入预定义（如 CSI300）：`python collector.py --index_name CSI300 --qlib_dir <dir> --method parse_instruments`。
- **多市场模式**：中（trade_unit=100, limit=0.099）；美（trade_unit=1, no limit）。初始化：`qlib.init(provider_uri=<dir>, region=REG_CN/REG_US)`。
- **落地提示**：自定义数据源 PR 欢迎（参考 yahoo collector）。停牌数据设为 NaN。

**目前CN数据只更新到2020/09/25**

### 数据 API（Data API）
- **检索**：用 qlib.data API（如 D.features/D.calendar）获取数据。参考 Data Retrieval。
- **Feature**：从提供者加载（如 $high/$close）。支持自定义 Operator（参考 tests/test_register_ops.py）。
- **ExpressionOps**：用表达式引擎构建特征（如 Mean($close, 5)）。操作符详见链接。
- **Filter**：NameDFilter（正则过滤仪器）；ExpressionDFilter（表达式过滤，如 '$close/$open>5' 或跨截面 '$rank($close)<10'）。
  - 配置示例：在 handler_config 加 filter_pipe。
- **解读**：API 简化数据访问，支持动态过滤。参考 Data API。

### 数据加载器（Data Loader）
- **QlibDataLoader**：从 Qlib 源加载原始数据。
- **StaticDataLoader**：从文件/提供数据加载。
- **接口**：load(instruments, start_time, end_time) → DataFrame。多索引可选（datetime/instrument）。
- **关键点**：支持 instruments 过滤（市场名/配置）。参考 Data Loader API。

### 数据处理器（Data Handler）
- **DataHandlerLP**：学可处理器，支持处理器链（infer/learn/shared）。过程类型：PTYPE_I（独立）/PTYPE_A（追加）。
- **处理器**：DropnaProcessor/TanhProcess 等（详见链接）。自定义继承 Processor。
- **示例**：Alpha158 配置（start_time 等）；独立运行：初始化 h = Alpha158(**config)，h.fetch(col_set="label/feature")。
- **接口**：fit/process_data/fetch。from_df 快速创建。
- **落地提示**：用 qrun 自动运行（改配置）；自定义用 ConfigSectionProcessor。
- **注意**：标签用 Ref($close, -2)/Ref($close, -1) - 1（T+1 到 T+2 变动，中股 T+1 买/T+2 卖）。参考 Data Handler API。

### 数据集（Dataset）
- **DatasetH**：带 Handler 的数据集，准备模型输入（segments: train/valid/test）。
- **接口**：prepare(segments, col_set, data_key=DK_I/DK_L)。支持模型特定处理。
- **自定义**：继承 DatasetH 实现特殊处理；否则直接用。
- **解读**：灵活处理（如 GBDT 容忍 NaN，NN 不行）。参考 Dataset API。

### 缓存（Cache）
- **全局内存**：MemCache (H['c']/H['i']/H['f']) 缓存日历/仪器/特征。限大小（length/sizeof）。
- **ExpressionCache**：缓存表达式（如 Mean($close, 5)）。自定义：覆盖 _uri/_expression/update。
- **DatasetCache**：缓存数据集（仪器/字段/时间/频）。自定义：覆盖 _uri/_dataset/update。
- **磁盘实现**：DiskExpressionCache/DiskDatasetCache。update 到最新日历。
- **关键点**：加速重复查询；Redis 锁防读写冲突（服务器侧）。参考 cache_to_origin_data/normalize_uri_args。

### 数据和缓存文件结构（Data and Cache File Structure）
- **结构**：
  ```
  - data/
      - calendars/  # day.txt 等
      - instruments/  # all.txt, csi500.txt 等
      - features/  # 原始特征，如 sh600000/open.day.bin
      - calculated features/  # 表达式缓存，hash(instrument, expr, freq)
      - cache/  # 数据集缓存，hash(stockpool, fields, freq) + .meta/.index
  ```
- **解读**：原始数据更新触发缓存更新。详见 Qlib 论文。

### 最小运行指引
1. **初始化**：`qlib.init(provider_uri='~/.qlib/qlib_data/cn_data', region=REG_CN)`
2. **数据准备**：下载/转换 CSV → .bin。
3. **加载/处理**：
   ```python
   from qlib.contrib.data.handler import Alpha158
   h = Alpha158(start_time="2008-01-01", end_time="2020-08-01", fit_start_time="2008-01-01", fit_end_time="2014-12-31", instruments="csi300")
   print(h.fetch(col_set="feature"))  # 特征
   ```
4. **数据集**：`ds = DatasetH(h, segments={"train": ("2008-01-01", "2014-12-31"), "test": ("2017-01-01", "2020-08-01")})；ds.prepare("train")`
5. **缓存**：默认启用；手动 update。


### 小结
- **优势**：端到端数据流（准备→处理→缓存），表达式引擎简化 alpha 构建；学可处理器支持实时（如归一化参数学习）。
- **相对手动**：自动化更新/转换，减少错误；高性能 .bin + 缓存加速回测。
- **生态扩展**：集成 PIT（时点一致性）；自定义 Operator/Processor/Cache。

## Forecast Model 模型训练与预测

Forecast Model 是 Qlib 的预测组件，用于生成股票分数（score），继承 Model 基类实现 fit/predict，支持 qrun 自动工作流或独立使用，与 Workflow/Recorder 集成。

- **核心功能**：预测股票分数（score），表示模型对仪器的评级（越高越盈利）。可集成 qrun 自动工作流（详见 Workflow: Workflow Management），或作为独立模块。
- **设计解读**：Qlib 组件松耦合，模型独立于数据/工作流。默认 score 含义依用户标签设置（e.g., 回报预测）。
- **优势**：模型库包括 LightGBM/MLP/LSTM 等基线；易扩展自定义模型。
- **适用场景**：量化预测、回测；与 Data Handler/Dataset 结合处理输入。

### 基类与接口
- **Model 基类**：所有模型继承 qlib.model.base.Model。
- **关键接口**：
  - fit(dataset: Dataset, reweighter: Reweighter)：训练模型。从 dataset 取 x_train/y_train/w_train（特征/标签/权重）。属性名勿以 '_' 开头（便于 dump）。
    - 示例提取数据：
      ```python
      df_train, df_valid = dataset.prepare(["train", "valid"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
      x_train, y_train = df_train["feature"], df_train["label"]
      # 权重处理：若无 'weight'，设为 1
      ```
  - predict(dataset: Dataset, segment: str|slice = 'test') → object：预测，返回 pandas.Series 等。
- **ModelFT**：支持 finetune 方法（细调模型）。
- 接口标准化训练/预测；reweighter 支持样本加权。参考 [Model API](https://qlib.readthedocs.io/en/latest/reference/api.html#module-qlib.model.base)。

### LightGBM 示例
先 qlib.init（详见 Initialization）
```python
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.data.handler import Alpha158
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord

market = "csi300"
benchmark = "SH000300"

data_handler_config = {
    "start_time": "2008-01-01",
    "end_time": "2020-08-01",
    "fit_start_time": "2008-01-01",
    "fit_end_time": "2014-12-31",
    "instruments": market,
}

task = {
    "model": {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {
            "loss": "mse",
            "colsample_bytree": 0.8879,
            "learning_rate": 0.0421,
            "subsample": 0.8789,
            "lambda_l1": 205.6999,
            "lambda_l2": 580.9768,
            "max_depth": 8,
            "num_leaves": 210,
            "num_threads": 20,
        },
    },
    "dataset": {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": data_handler_config,
            },
            "segments": {
                "train": ("2008-01-01", "2014-12-31"),
                "valid": ("2015-01-01", "2016-12-31"),
                "test": ("2017-01-01", "2020-08-01"),
            },
        },
    },
}

# 模型/数据集初始化
model = init_instance_by_config(task["model"])
dataset = init_instance_by_config(task["dataset"])

# 启动实验
with R.start(experiment_name="workflow"):
    R.log_params(**flatten_dict(task))
    model.fit(dataset)  # 训练

    # 预测
    recorder = R.get_recorder()
    sr = SignalRecord(model, dataset, recorder)
    sr.generate()  # 生成 pred_score
```
- **解读**：用 Alpha158 Handler 准备数据；SignalRecord 记录信号。完整 Jupyter：examples/train_backtest_analyze.ipynb。

## 四、自定义模型（Custom Model）
- **步骤**：继承 Model，实现 fit/predict。集成到 Qlib：参考 Custom Model Integration。
- **落地提示**：确保 predict 返回 Series（仪器分数）；测试时用小数据集验证。

### 小结
- 基类简化开发；集成 qrun/Recorder 自动训练/记录；模型库（GBDT/NN）作基线，便于对比。
- 标准化接口，提升复现；finetune 支持迭代优化。
- 与 Dataset（数据准备）、Workflow（端到端）、Task Management（批量）闭环；自定义易 PR。

## 投组管理 Portfolio Management

Portfolio Strategy 是 Qlib 的组合管理组件，用于基于预测分数（score）生成投资决策，继承基类自定义策略，支持 qrun 自动工作流或独立使用，与 Forecast Model/Backtest 集成评估性能。

- **核心功能**：基于 Forecast Model 的预测分数，采用算法生成投资组合（portfolio）。可集成 Workflow（详见 Workflow: Workflow Management），或独立模块。
- **设计解读**：Qlib 组件松耦合，策略独立于模型/执行器。提供内置策略（如 TopkDropout），支持自定义。
- **优势**：回测（backtest）评估策略性能；分数含义依模型标签（越高越盈利）。
- **适用场景**：组合优化、风险控制；与 SignalRecord 结合记录信号。

### 基类与接口
- **BaseStrategy**：所有策略继承 qlib.strategy.base.BaseStrategy。
  - **关键接口**：generate_trade_decision(trade_date, current_pos, trade_exchange) → TradeDecision。
    - 每个交易 bar 调用（依执行器频率，如每日）。返回空决策（EmptyTradeDecision）可控制交易频率（如周频）。
- **WeightStrategyBase**：BaseStrategy 子类，专注目标仓位（target positions），自动生成订单。
  - **关键接口**：generate_target_weight_position() → dict（{instrument: weight}，权重为总资产百分比，不含现金）。
  - **过程**：目标仓位 → 目标数量 → 订单列表（generate_order_list）。
- **解读**：BaseStrategy 灵活控制决策；WeightStrategyBase 简化仓位管理。参考图示理解 Topk-Drop 算法。

### 实现策略

![Topk-Drop algorithm](img/qlib-1762376264234.png)

- **TopkDropoutStrategy**：继承 BaseStrategy，实现 generate_order_list。
  - **过程**：
    - Topk-Drop 算法：持仓 Topk 只，按分数排名卖出最差 Drop 只，买入最佳未持仓。
    - 参数：topk（持仓数）、n_drop（Drop，每日卖出数）。
    - 换手率：约 2 * Drop / Topk。
  - 生成订单：目标数量 → 订单列表。
- **EnhancedIndexingStrategy**：增强指数策略，结合主动/被动管理，超基准（如 S&P 500）回报，控制跟踪误差。
  - 参考：qlib.contrib.strategy.signal_strategy.EnhancedIndexingStrategy / EnhancedIndexingOptimizer。
- **落地提示**：继承 WeightStrategyBase 只需实现 generate_target_weight_position；自定义算法如优化换手。

### 使用与示例
- **预测分数（Prediction Score）**：pandas DataFrame，index 为 (datetime, instrument)，含 'score' 列。示例：
  ```
  datetime   instrument     score
  2019-01-04 SH600000   -0.505488
  ...        ...         ...
  2019-04-30 SZ300760   -0.126383
  ```
  - 来源：Forecast Model 输出（详见 Forecast Model）。分数规模依模型标签，可能需缩放（e.g., 回归拟合到回报）。
- **运行回测（Running backtest）**：
  - **简单版**（backtest_daily）：
    ```python
    from pprint import pprint
    import qlib
    import pandas as pd
    from qlib.utils.time import Freq
    from qlib.utils import flatten_dict
    from qlib.contrib.evaluate import backtest_daily
    from qlib.contrib.evaluate import risk_analysis
    from qlib.contrib.strategy import TopkDropoutStrategy

    qlib.init(provider_uri="<qlib data dir>")

    CSI300_BENCH = "SH000300"
    STRATEGY_CONFIG = {
        "topk": 50,
        "n_drop": 5,
        "signal": pred_score,  # 预测分数
    }

    strategy_obj = TopkDropoutStrategy(**STRATEGY_CONFIG)
    report_normal, positions_normal = backtest_daily(
        start_time="2017-01-01", end_time="2020-08-01", strategy=strategy_obj
    )
    analysis = {}
    analysis["excess_return_without_cost"] = risk_analysis(report_normal["return"] - report_normal["bench"])
    analysis["excess_return_with_cost"] = risk_analysis(report_normal["return"] - report_normal["bench"] - report_normal["cost"])

    analysis_df = pd.concat(analysis)
    pprint(analysis_df)
    ```
  - **高级版**（控制执行器）：
    ```python
    # ... 同上导入
    FREQ = "day"
    EXECUTOR_CONFIG = {
        "time_per_step": "day",
        "generate_portfolio_metrics": True,
    }
    backtest_config = {
        "start_time": "2017-01-01",
        "end_time": "2020-08-01",
        "account": 100000000,
        "benchmark": CSI300_BENCH,
        "exchange_kwargs": {
            "freq": FREQ,
            "limit_threshold": 0.095,
            "deal_price": "close",
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
        },
    }

    strategy_obj = TopkDropoutStrategy(**STRATEGY_CONFIG)
    executor_obj = executor.SimulatorExecutor(**EXECUTOR_CONFIG)
    portfolio_metric_dict, indicator_dict = backtest(executor=executor_obj, strategy=strategy_obj, **backtest_config)
    analysis_freq = "{0}{1}".format(*Freq.parse(FREQ))
    report_normal, positions_normal = portfolio_metric_dict.get(analysis_freq)

    # 分析（excess_return 等）
    analysis = {}
    analysis["excess_return_without_cost"] = risk_analysis(report_normal["return"] - report_normal["bench"], freq=analysis_freq)
    analysis["excess_return_with_cost"] = risk_analysis(report_normal["return"] - report_normal["bench"] - report_normal["cost"], freq=analysis_freq)
    analysis_df = pd.concat(analysis)
    pprint(analysis_df)
    ```
- **解读**：benchmark 用于超额回报计算（单仪器格式，如 SH000300）；market 是仪器池（如 csi300）。回测输出：report（回报/基准/费用）、positions（仓位）。

### 结果分析
- **示例输出**：
  ```
                                                  risk
  excess_return_without_cost mean               0.000605
                             std                0.005481
                             annualized_return  0.152373
                             information_ratio  1.751319
                             max_drawdown      -0.059055
  excess_return_with_cost    mean               0.000410
                             std                0.005478
                             annualized_return  0.103265
                             information_ratio  1.187411
                             max_drawdown      -0.075024
  ```
- **指标含义**：
  - excess_return_without_cost/with_cost：无/有费用超额回报（CAR）。
    - mean/std：均值/标准差。
    - annualized_return：年化回报。
    - information_ratio：信息比率（IR = 年化回报 / 标准差）。
    - max_drawdown：最大回撤（MDD）。
- **落地提示**：用 risk_analysis 评估；扁平化 dict 便于日志。

### 小结
- 内置 TopkDropout/增强指数，易自定义；回测标准化评估（IR/MDD）。
- 自动化订单生成/风险分析，减少错误；松耦合，便于换模型/策略。
- 与 Forecast Model（分数输入）、Workflow（qrun 集成）、Task Management（批量）闭环。

## 高频交易的嵌套决策执行框架设计

日度交易（如投资组合管理）和日内交易（如订单执行）是量化投资的热门话题，通常被单独研究。
- **问题**：
  - 要获得日度和日内交易的联合性能，必须让它们相互交互并进行联合回测。
  - 为了支持多级联合回测策略，需要相应的框架。
  - 公开的高频交易框架未考虑多级联合交易，导致回测不准确。
- **策略优化问题**：
  - 不同级别的策略优化不是独立的，相互影响。例如，最佳投资组合管理策略可能随着订单执行性能的改变而变化（例如，当订单执行策略改善时，高换手率的组合可能变得更好）。
  - 要实现整体良好性能，需要考虑不同级别策略的交互。

![Design of the Framework](img/qlib-1762377282108.png)

框架设计每个级别包括`Trading Agent`（交易代理）和`Execution Env`（执行环境）。

- **Trading Agent 组件**：
  - `Information Extractor`：数据处理模块。
  - `Forecast Model`：预测模块，输出预测信号。
  - `Decision Generator`：决策生成器，根据预测信号生成决策。
- **工作流程**：
  - 交易算法通过`Decision Generator`生成决策。
  - 决策传递给`Execution Env`，返回执行结果。
- **自定义性**：
  - 用户可自定义交易算法的频率（例如，日内、日频、周频）、决策内容和执行环境。
  - 执行环境可嵌套更细粒度的交易算法和环境（即子工作流，例如，日频订单可日内拆分成更细决策）。
- **优势**：
  - 灵活性便于用户探索不同级别交易策略的组合效果。
  - 打破不同级别交易算法之间的优化壁垒。
- **优化支持**：通过 QlibRL 支持嵌套决策执行框架的优化。

## 元学习 Meta Controller = Meta-Task & Meta-Dataset & Meta-Model

金融市场非平稳，训练期与未来测试期分布常发生漂移；单一模型易“过期”。Meta Controller 的目标是在一系列预测任务之间学习可迁移的“规律”，并用这些规律去指导后续的预测模型/任务（guidance），从而提升鲁棒性与泛化。
* **问题**：金融等流数据场景里分布会随时间漂移（concept drift），如果训练只跟着“最近数据”走，往往**滞后一步**，新漂移来了你才去适应。
* **思想**：在很多场景里，漂移**并非完全随机**，存在季节性/周期性/趋势，因此**可预测未来分布趋势**，先把训练集“重采样/生成”为**更像未来**的分布，再训练基础预测模型，让它**提前对齐未来**。

* **MetaTask**：元学习框架的最小单元，保存可供 MetaModel 使用的数据；同一组 MetaTask 可共享 DataHandler。训练/测试/迁移阶段对输入数据的处理不同（`PROC_MODE_FULL / TEST / TRANSFER`）——训练任务通常需要 `X, y, X_test, y_test`，而迁移到其他数据集时只需 `meta_info`。常用 `prepare_task_data()` 得到可直接喂给 MetaModel 的数据。
* **MetaTaskDataset**：负责**生成元信息并组织 MetaTask 列表**，对 MetaModel 提供训练/推理所需的“元数据”。支持**跨数据集迁移**（在 A 上学到的模式可用于 B），常用 `prepare_tasks(["train","test"])` 取回各分段的任务列表。
* **MetaModel**：执行“指导”。分两类：

  1. **MetaTaskModel**：**改写/生成基础任务定义**（prepare_tasks 产出被改写的任务定义，再用这些定义去训练新的基础预测模型）。
  2. **MetaGuideModel**：**插入基础模型训练过程**，在训练阶段提供指引（超参/采样/损失加权等）。均暴露 `fit(..)` / `inference(..)`。

### DDG-DA（Data Distribution Generation for Predictable Concept Drift Adaptation）

0. AAAI 2022《DDG-DA: Data Distribution Generation for Predictable Concept Drift Adaptation》。在股票趋势、电力负荷、太阳辐照等真实任务上，对多种主流模型都有显著提升。
1. **学未来分布**：用一个“分布预测器”（元模型的一部分，参数常记为 θ）去**预测下一时段的目标分布**；实现上用历史数据学到的“分布表征”与一个**可微的分布距离**来训练该预测器。
2. **生成训练集合成分布**：基于预测的未来分布，对历史样本**加权重采样**（或生成等价的数据权重），得到一个更贴近未来分布的“合成训练集”。
3. **再训练基础模型**：用合成训练集去训练真正做预测的基础模型（如 LGBM/MLP 等），从而在未来测试期表现更好。
  - 论文给出了**可微分布距离**并证明它与 KL 等传统距离的等价性，从而能端到端学习“如何重采样到更像未来”的训练数据。
4. **有效**：存在**可预测的漂移**（季节性/周期性/再现的 market regime 等），例如你常见的月度/季度节奏、行业轮动带来的分布缓慢迁移。
5. **不太有效**：**纯粹突发且难以建模的跳变**（如黑天鹅、极端政策/财报意外），这类漂移**不可预测**，DDG-DA 也难以提前合成出“未来像什么”。论文里也强调“多数漂移呈非随机趋势；若完全随机则难处理”。

Qlib 把 DDG-DA 实现为 **Meta Model** 的一个示例，完整流程是：
① 计算 **meta-information** 并封成 **MetaTask**（组成 **MetaDataset**）→ ② 在训练段上训练 DDG-DA → ③ 推理得到“指导信息”（即未来分布引导/样本权重等）→ ④ 将指导应用到基础预测模型训练以提升表现。

* 示例代码路径：`examples/benchmarks_dynamic/DDG-DA/workflow.py`。

## Recorder 记录

量化实验的「实验本」—— 自动记录参数、指标、预测、模型，随时回看、对比。

| 概念 | 比喻 | 作用 |
|------|------|------|
| **Experiment** | 一本实验记录本 | 包含多次运行（Recorder） |
| **Recorder** | 本子里的一页 | 记录一次训练/回测的所有信息 |
| **R** | 你的「笔」 | 全局工具，直接写东西到当前页 |

### 结构图

```
ExperimentManager
└── Experiment: "回测实验A"
    ├── Recorder: "LSTM v1"
    ├── Recorder: "Transformer v2"
    └── Recorder: "当前运行"
```

### 核心操作

```python
from qlib.workflow import R

with R.start(experiment_name="alpha_test"):   # 打开实验本
    model.fit(dataset)
    pred = model.predict(dataset)
    
    R.log_metrics(IC=0.06, RankIC=0.08)       # 写指标
    R.save_objects(pred=pred)                 # 存预测
    R.log_params(lr=0.001, batch=128)         # 存参数
# 自动结束，保存到 MLflow
```

## 常用 API

| 命令 | 作用 |
|------|------|
| `R.start(exp_name)` | 开始实验（推荐用 `with`） |
| `R.log_metrics(IC=0.05)` | 记录指标 |
| `R.log_params(lr=0.01)` | 记录超参 |
| `R.save_objects(pred=pred)` | 保存预测/模型 |
| `R.get_recorder().load_object("pred")` | 加载保存的对象 |

### 可视化

```bash
mlflow ui
```
打开浏览器 → 看到所有实验、指标对比、参数、文件下载。

## [评估结果分析](https://qlib.readthedocs.io/en/latest/component/report.html)


| 模块 | 功能 | 适用对象 |
|------|------|----------|
| `analysis_position` | **仓位级分析**：评估策略整体收益、风险、换手 | 策略开发者、基金经理 |
| `analysis_model` | **模型级分析**：评估预测分数的质量与稳定性 | 模型研究员、量化科学家 |

> **核心思想**：所有累计收益指标（如年化收益、最大回撤）均采用 **线性求和**，避免指数化扭曲长期表现。

---

支持的图形报告

```python
import qlib.contrib.report as qcr
print(qcr.GRAPH_NAME_LIST)
```

```python
[
    'analysis_position.report_graph',
    'analysis_position.score_ic_graph',
    'analysis_position.cumulative_return_graph',
    'analysis_position.risk_analysis_graph',
    'analysis_position.rank_label_graph',
    'analysis_model.model_performance_graph'
]
```

### `analysis_position` 仓位分析

#### `report_graph` —— **策略表现总览仪表盘**

图形构成

| 曲线 | 名称 | 含义 |
|------|------|------|
| `cum return wo cost` | 无成本累计收益 | 策略在无交易成本下的理论累计收益 |
| `cum return w cost` | 有成本累计收益 | 真实可实现的累计收益 |
| `cum bench` | 基准累计收益 | 市场或指数表现 |
| `cum ex return wo cost` | 无成本超额收益 | 策略 - 基准（无成本） |
| `cum ex return w cost` | 有成本超额收益 | 策略 - 基准（有成本） |
| `turnover` | 换手率 | 每日仓位调整比例 |
| `return wo cost mdd` | 无成本最大回撤 | 累计收益最大跌幅 |
| `cum ex return wo cost mdd` | 无成本超额最大回撤 | 超额收益最大跌幅 |

阴影区域
- **上部阴影**：对应 `cum return wo cost` 的最大回撤区间
- **下部阴影**：对应 `cum ex return wo cost` 的最大回撤区间

评估要点
| 问题 | 如何判断 |
|------|----------|
| 策略是否赚钱？ | `cum return w cost` 持续上升 |
| 是否跑赢市场？ | `cum ex return w cost` > 0 且上升 |
| 交易成本影响大吗？ | `wo cost` 与 `w cost` 差距 |
| 回撤是否可控？ | 阴影区域宽度与深度 |

#### `score_ic_graph` —— **预测信号质量时序图**

指标定义

| 指标 | 公式 | 含义 |
|------|------|------|
| **IC** | `corr(pred, label)` | 预测值与真实收益的 **线性相关性** |
| **Rank IC** | `spearman_corr(rank(pred), rank(label))` | 预测排名与收益排名的 **排序相关性** |

> **标签示例**：`Ref($close, -2)/Ref($close, -1) - 1`（未来1日收益率）

图形解读
- **X轴**：交易日
- **Y轴**：IC / Rank IC 值（每日计算）
- **理想状态**：IC > 0.03 且稳定，Rank IC > 0.05 更佳

评估要点

| 问题 | 如何判断 |
|------|----------|
| 模型是否有预测力？ | IC 均值 > 0.03 |
| 预测是否稳定？ | IC 波动小、不衰减 |
| 排名能力如何？ | Rank IC 更鲁棒，优先关注 |


#### `risk_analysis_graph` —— **风险指标全面评估**

整体风险柱状图（4大核心指标）

| 指标 | 公式 | 解释 |
|------|------|------|
| **Annualized Return** | `(1 + total_return)^{252/n} - 1` | 年化收益率 |
| **Max Drawdown** | `min((trough - peak)/peak)` | 最大回撤 |
| **Information Ratio (IR)** | `Annualized Excess Return / Annualized Excess Std` | 单位风险超额收益 |
| **Std (Volatility)** | `std(excess_return) * sqrt(252)` | 年化超额波动率 |

> **IR > 1.0**：优秀策略  
> **IR > 2.0**：顶级策略

月度分析图

| 图表 | 内容 | 用途 |
|------|------|------|
| **月度年化收益** | 每月超额年化收益率 | 发现季节性规律 |
| **月度最大回撤** | 每月最大跌幅 | 识别风险集中期 |
| **月度IR** | 每月信息比率 | 评估稳定性 |
| **月度波动率** | 每月超额收益标准差 | 评估波动来源 |


### `analysis_model` 模型预测能力

#### `model_performance_graph` —— **模型分层验证**

##### **分层累计收益图**

**分组逻辑**：
- 按 **真实收益（label）升序排名**
- 分为 5 组（20% 分位）
- 计算每组的累计收益

| 曲线 | 含义 |
|------|------|
| **Group1** | 收益最低 20% 股票 |
| **Group5** | 收益最高 20% 股票 |
| **long-short** | long-short 收益 = Group 5 的累计收益 - Group 1 的累计收益 |
| **long-average** | Group1 - 市场平均 |

> **理想状态**：`long-short` 持续上升 → 模型能有效区分好坏股

##### **long-short 分布图**

- 每日 `long-short` 收益的 **箱线图**
- 评估 **收益稳定性** 和 **极端值**

| 情况 | `long-short` | `long-average` | 你的策略收益 | 结论 |
|------|--------------|----------------|---------------|------|
| A | +30% | +15% | +18% | 优秀！接近上限 |
| B | +30% | +15% | +5%  | 差！模型无效 |
| C | +5%  | +2%  | +3%  | 一般，市场无 alpha |
| D | +50% | +25% | +10% | 潜力大，但模型太弱 |

##### **IC 分析图**

| 图表 | 内容 |
|------|------|
| **IC 曲线** | 每日 IC |
| **Monthly IC** | 每月平均 IC |
| **IC 分布图** | 所有交易日的 IC 分布 |
| **Q-Q 图** | IC 是否服从正态分布 |

> Q-Q 图贴近对角线 → IC 分布稳定

##### **自相关图（Auto Correlation）**

- 计算：`corr(pred_t, pred_{t-lag})`
- **用途**：估计 **换手率**
  - 自相关衰减慢 → 信号持续性强 → 换手低
  - 衰减快 → 需频繁调整 → 换手高


### 核心指标完整解释表

| 指标 | 公式 | 业务含义 | 优秀标准 |
|------|------|----------|---------|
| **IC** | `corr(pred, label)` | 预测准确性 | > 0.03 |
| **Rank IC** | `spearman_corr(rank(pred), rank(label))` | 排序能力 | > 0.05 |
| **年化超额收益** | `(1 + excess_return)^{252/n} - 1` | 跑赢市场能力 | > 10% |
| **最大回撤** | `min((low - high)/high)` | 抗风险能力 | < 15% |
| **IR** | `年化超额收益 / 年化超额波动率` | 风险调整后收益 | > 1.0 |
| **换手率** | `sum(|w_t - w_{t-1}|)` | 交易成本 | < 200% |
| **long-short 收益** | `return_top - return_bottom` | 选股能力 | 持续正向 |

### 使用流程

```python
# 1. 获取分析对象
from qlib.workflow import R
from qlib.contrib.evaluate import PortAnaRecord

recorder = R.get_recorder()
analysis = PortAnaRecord(recorder).generate()

# 2. 生成所有图形
import qlib.contrib.report as qcr

qcr.analysis_position.report_graph(analysis)
qcr.analysis_position.score_ic_graph(analysis)
qcr.analysis_position.risk_analysis_graph(analysis)
qcr.analysis_model.model_performance_graph(analysis)
```

### 评估策略的“四看”法则

| 看什么 | 用哪张图 | 关键指标 |
|--------|----------|----------|
| **总收益** | `report_graph` | 超额收益曲线 |
| **预测力** | `score_ic_graph` | IC / Rank IC |
| **风险性价比** | `risk_analysis_graph` | IR |
| **模型能力** | `model_performance_graph` | long-short + IC 分布 |

> **最终目标**：  
> **IC > 0.05** + **IR > 1.0** + **long-short 持续上升** + **回撤可控**  
> → 即可认为策略具备实盘潜力。

## Online Serving 实时市场预测与交易

**“在线策略”是一套动态管理模型生命周期的规则：决定**什么时候训练新模型、什么时候上线模型、什么时候生成交易信号**，以适应不断变化的市场。**

| 问题 | 传统回测的缺陷 | 在线策略的解决方案 |
|------|----------------|------------------|
| **市场在变** | 回测用历史数据训练一个模型，实盘时失效 | **定期重新训练**，动态上线新模型 |
| **模型会过时** | 固定模型无法适应新风格 | **滚动训练 + 动态切换** |
| **信号需实时** | 回测信号是“事后诸葛亮” | **每日更新预测 → 生成下一交易日信号** |
| **多模型管理混乱** | 手动切换模型容易出错 | **自动化管理上线/下线** |

> **核心思想**：**模型不是“一劳永逸”的，而是“活的”——要随着市场不断迭代。**

### 核心概念总览（4 个关键角色）

| 角色 | 比喻 | 职责 |
|------|------|------|
| **Online Manager** | 指挥中心 | 统筹所有策略，控制每日流程 |
| **Online Strategy** | 作战方案 | 决定“什么时候训练、用哪个模型、怎么出信号” |
| **Online Model** | 现役士兵 | 当前参与预测的模型（可动态替换） |
| **Updater** | 后勤补给 | 当新数据到来时，更新预测/标签 |

### “在线策略”的工作流程

```mermaid
flowchart TD
    %% ==== 样式定义 ====
    classDef strategy fill:#7ED321,stroke:#5BA818,color:#fff
    classDef manager fill:#4A90E2,stroke:#2E6DA4,color:#fff
    classDef model fill:#F5A623,stroke:#D4840E,color:#fff
    classDef signal fill:#50E3C2,stroke:#35B290,color:#fff
    classDef data fill:#9013FE,stroke:#6A0DAD,color:#fff
    classDef decision fill:#FF6B6B,stroke:#D64545,color:#fff

    %% ==== 主流程 ====
    A["每日开盘前"] --> B["新数据到来"]
    B --> C["OnlineManager 触发 routine"]

    subgraph 在线策略核心决策 [在线策略（OnlineStrategy）]
        C --> D{"是否需要训练新模型?"}
        D -->|是| E["prepare_tasks(cur_time)"]
        E --> F["生成训练任务列表"]
        F --> G["Trainer 训练新模型"]
        D -->|否| H["直接进入上线阶段"]

        G --> I{"如何选择上线模型?"}
        H --> I
        I --> J["prepare_online_models(trained_models)"]
        J --> K["标记为 'online'"]
        K --> L["替换/加入现役模型池"]
    end

    L --> M["Updater 更新在线模型预测"]
    M --> N["使用最新数据补全预测"]
    N --> O["Manager 收集所有在线模型预测"]

    O --> P["prepare_signals(集成方式)"]
    P --> Q["生成下一交易日交易信号"]
    Q --> R["输出信号 DataFrame/Series"]
    R --> S["等待下一周期"]

    %% ==== 首次启动 ====
    T["系统首次启动"] --> U["Manager.first_train()"]
    U --> V["Strategy.first_tasks()"]
    V --> W["Trainer 训练初始模型"]
    W --> X["上线初始模型"]
    X --> S

    %% ==== 样式绑定 ====
    class D,I decision
    class E,J,P strategy
    class C,U manager
    class G,W model
    class M data
    class Q,R signal
```

| 决策 | 问题 | 典型实现 |
|------|------|----------|
| **任务生成** | 今天要不要训练新模型？ | 每 30 天滚动训练一次 |
| **模型上线** | 训练好了，哪个模型上场？ | 选 IC 最高的 / 全部平均 |
| **信号准备** | 怎么把预测变成交易信号？ | 取预测分数的平均值 |

### 运行模式对比

| 模式 | 用途 | 训练时机 | 推荐 Trainer |
|------|------|----------|--------------|
| **实时模式（Online）** | 实盘部署 | 每日例行时训练 | `Trainer`（逐一训练） |
| **模拟模式（Simulation）** | 历史回测验证 | 所有任务准备好后**一次性训练** | `DelayTrainer`（并行加速） |

### 关键概念

#### **Online Model（现役模型）**
- **定义**：当前参与预测的模型集合。
- **特点**：
  - 数量可变（1个或多个）。
  - 随时可替换（老模型下线，新模型上线）。
  - 有标签标记：`online` / `offline`。
- **比喻**：足球队中的首发阵容。

#### **Routine（例行流程）**
- **定义**：一个固定周期（如每天）的更新流程。
- **内容**：
  1. 检查新数据
  2. 准备训练任务
  3. 训练模型
  4. 上线模型
  5. 生成信号
- **频率**：默认 `'day'`，可支持分钟级。

#### **Updater（更新器）**
- **作用**：当**新数据到来**时，自动用现役模型**补预测**。
- **场景**：
  - 昨晚收盘数据更新了
  - 需要预测今天的因子值
  - Updater 自动加载模型，补全预测
- **支持类型**：`PredUpdater`（预测）、`LabelUpdater`（标签）

### 小结

| 普通回测 | 在线策略 |
|----------|----------|
| 静态模型 | 动态模型 |
| 一次性训练 | 滚动训练 |
| 历史信号 | 实时信号 |
| 无法实盘 | 可部署实盘 |

| 场景 | 在线策略设计 |
|------|--------------|
| **风格轮动** | 每月训练一个新 LightGBM，上线 IC 最高者 |
| **多模型集成** | 每周训练 3 个模型（GBDT、NN、LR），上线后取平均 |
| **高频信号** | 每 5 分钟滚动训练，生成分钟级买卖信号 |

1. **在线策略 = 模型的动态管理规则**  
2. **核心任务：训练 → 上线 → 出信号**  
3. **随市场变化，模型也要“呼吸”**  
4. **Manager 管流程，Strategy 管决策**  
5. **Updater 是“补货员”，保证预测不缺货**

## 构建公式化 Alpha

在量化交易实践中，能解释并预测未来资产收益的新型因子（alpha factors）对策略盈利至关重要。公式化 Alpha（formulaic alpha）顾名思义，就是能用数学公式/表达式表示的 Alpha。Qlib 支持用户便捷地构建这类 Alpha。

### 用表达式构建 MACD
MACD（移动平均收敛-发散指标）用于捕捉价格趋势强度、方向、动量和持续时间的变化。

$$
MACD = 2 \times (DIF - DEA)
$$

其中

$$
DIF = \frac{EMA(CLOSE,12) - EMA(CLOSE,26)}{CLOSE},\quad
DEA = \frac{EMA(DIF,9)}{CLOSE}
$$

Qlib 表达式与加载方式

```python
from qlib.data.dataset.loader import QlibDataLoader

MACD_EXP = '(EMA($close,12)-EMA($close,26))/$close - EMA((EMA($close,12)-EMA($close,26))/$close,9)/$close'
fields = [MACD_EXP]; names = ['MACD']
labels = ['Ref($close,-2)/Ref($close,-1)-1']; label_names = ['LABEL']

data_loader = QlibDataLoader(config={"feature":(fields,names), "label":(labels,label_names)})

df = data_loader.load(instruments='csi300', start_time='2010-01-01', end_time='2017-12-31')
```

运行后将得到带有 `MACD`（特征）与 `LABEL`（标签）的多层索引 DataFrame。
* `$close`：内置“收盘价”特征字段的占位符（所有原始特征一般以 `$` 前缀）。
* `EMA(x, n)`：对序列 `x` 做 n 期指数移动平均。
* `Ref(x, k)`：对 `x` 进行时间位移（示例里用来构造“相邻两期之间”的收益率标签）。标签表达式 `Ref($close,-2)/Ref($close,-1)-1` 来自官方示例，用作演示；在实盘研究时应选择与你课题相符的标签定义与对齐方式。
* `QlibDataLoader`：把“**feature（特征表达式+命名）**”和“**label（标签表达式+命名）**”封装为加载配置，一次性拉取一篮子标的与时间段的数据。

## 序列化

Qlib 支持将 DataHandler、DataSet、Processor、Model 等对象的状态序列化到磁盘，并在之后重新加载。用途包括：跨进程/机器复用对象状态、加速实验重复、固化训练时的归一化统计、管控可复现实验等。

### qlib.utils.serial.Serializable

- 可序列化基类：Serializable 的实例可以用 pickle 格式dump 到磁盘或load 回内存。
- 对象中名字不以下划线 _ 开头的属性会被保存。也可以通过 config() 方法或覆写 default_dump_all 来关闭这一默认行为。
- 选择后端：可覆写 pickle_backend 选择后端：
  - "pickle"（默认、通用）；
  - "dill"（可序列化更多对象，如函数等复杂结构）

```py
# qlib.data.dataset.DatasetH
# ============= dump =============
dataset.to_pickle(path="dataset.pkl")    # dataset 是 qlib.data.dataset.DatasetH 的实例

# ============= reload =============
import pickle
with open("dataset.pkl", "rb") as f:
    dataset = pickle.load(f)
```

只保存“状态”，不保存“数据”。例如用于归一化的均值/方差等统计是“状态”，应被保存；而真正的数据矩阵不是状态，不会随对象一起落盘。回载后需要重新初始化（reinit）：可以在回载后重设 instruments / start_time / end_time / segments 等，再据此重新生成数据。这也是“状态与数据分离”的设计要点。
- 研究用的数据通常很大（分钟/日频多标的、多因子矩阵）。如果把这些数据随对象一起打包进 pickle，文件会巨无霸、重复存储同一份数据，传输和版本管理都很痛苦。Qlib 的做法是：数据独立存放为 Qlib 格式（.bin 等），序列化只保存“如何生成数据的状态”（如归一化的统计量、处理流水线配置等）。这样对象很轻，随处拷、随处复用。
- DatasetH/DataHandler 等是可序列化的，但默认不把数据随对象落盘；对象回载后，应通过 config()/setup_data() 重新指定 instruments/start_time/end_time/segments，再让它按状态重算/装载数据（满足你当前窗口与标的的需求）。而且以 _ 开头的数据属性不会被保存，进一步确保“只存状态”。
- 金融数据会被事后修订。如果把“某一时刻导出的整块数据”封死在 pickle 里，很容易和点时数据库（PIT）的目标冲突（回测与线上一致、历史时点取到当时版本）。Qlib 用独立的数据层 + PIT 机制保证“按时间点取对版本”；序列化只保留处理状态，回载时再从数据层读取当下需要的时段/标的，才能持续拿到正确版本的数据。

## 任务管理

![任务管理步骤](img/qlib-1762369488486.png)

基于 Qlib 最新文档（qlib.readthedocs.io/en/latest/component/workflow.html#task-management，截至 2025 年 11 月 6 日）和 GitHub 示例（microsoft/qlib/examples/model_rolling/task_manager_rolling.py）。Task Management 是 Qlib Workflow 的扩展，用于自动化批量任务（如滚动回测、多超参对比），支持分布式训练和 Online Serving 集成。核心流程：生成 → 存储 → 训练 → 收集。笔记按流程组织，精炼关键概念、代码示例和实战要点；补充最小用法、常见坑位及价值分析。重点：依赖 MongoDB 作为任务池，适用于量化研究的任务编排。优化后结构更紧凑、逻辑清晰，突出可操作性。

- **核心功能**：扩展 Workflow/qrun 的单任务限制，提供端到端流程：任务生成、存储、训练、收集。自动化批量实验（如不同时间段/loss/模型对比），适用于 Online Serving（如模型切换/融合）。
- **流程解读**：任务队列系统，上游模板批量产命令（TaskGen），中游 MongoDB 池管理状态，下游 Worker 分布式消费，最后 Collector 汇总评估。状态机：WAITING → RUNNING → PART_DONE → DONE。
- **优势**：标准化复现实验，支持分布式（多机共享 Mongo），集成 Qlib 生态。最新更新：Python 3.12+ 支持，DelayTrainer 异步优化，Ensemble 动态权重。
- **适用场景**：滚动回测、多模型 A/B 测试、离线 → 在线闭环。
- **示例**：GitHub examples/model_rolling/task_manager_rolling.py（LightGBM 滚动 + IC/IR 汇总）。

### 任务生成（Task Generating）
- **任务定义**：dict 结构，包括 Model、Dataset、Record（评估器）和自定义组件（文档 Task Section）。
- **生成机制**：自定义 TaskGen.generate(task: dict) → List[dict]。内置 RollingGen：按日期切片批量生成，测试时间段影响（e.g., 滚动 segments）。
- **实用扩展**：固定模型/特征，变 segments/loss 列表产任务；多预测期用 MultiHorizonGenBase（API/源码）。
- **关键点**：模板化批量，减少手动重复配置。

### 任务存储（Task Storing）
- **配置准备**：
  ```python
  from qlib.config import C
  C["mongo"] = {"task_url": "mongodb://localhost:27017/", "task_db_name": "rolling_db"}
  ```
- **核心工具**：TaskManager(task_pool: str)，对应 Mongo Collection。管理生命周期、并发安全（原子操作）。
- **任务结构**（Mongo 文档）：
  ```json
  {
    "def":    // pickle 序列化任务定义
    "filter": // JSON 筛选键（去重/查询）
    "status": "waiting" | "running" | "part_done" | "done",
    "res":    // pickle 序列化结果
  }
  ```
- **CLI 命令**：
  - `-h`：帮助。
  - `-t <pool> wait`：等待消费。
  - `-t <pool> task_stat`：状态统计。
- **常用方法**：create_task（批量插入，去重）；replace_task（替换）；list()（列池）。
- **注意**：首次排查连接/权限/索引；状态机确保流程流转。

### 任务训练（Task Training）
- **执行入口**：run_task(task_pool, task_func=qlib.model.trainer.task_train)，消费 WAITING/PART_DONE 任务，执行 Model/Dataset/Record。
- **关键参数**：query（筛选）、before/after_status（状态控制）、force_release（资源释放）、**kwargs（传 func）。
- **Trainer 扩展**：
  - Trainer.train(tasks) → list（批量训，返回 Recorder/路径）。
  - DelayTrainer：分预处理 + 训练，支持异步。
- **分布式实战**：
  - 单机：多进程 run_task。
  - 集群：多机共享 Mongo + pool（GitHub Issue 确认）。
- **关键点**：Worker 消费模式，易水平扩展。

### 任务收集（Task Collecting）
- **准备**：qlib.init(mlruns_path) 指定实验记录目录。
- **工具集**：
  - Collector：汇总预测/指标成结果集（可扩展）。
  - Group：按规则分组（e.g., 窗口统计）。
  - Ensemble：集成（如加权融合，最新动态权重）。
- **应用**：Rolling 对比、指标曲线、离线 → 在线闭环。
- **关键点**：端到端评估，集成 Qlib Recorder。

### 最小可运行套路（落地清单）
1. **生成任务**：定义 base_task dict，用 RollingGen.generate(base_task) → task_defs。
2. **存储**：
   ```python
   from qlib.workflow.task.manage import TaskManager
   tm = TaskManager("pool_name")
   tm.create_task(task_defs)  // 批量插入
   ```
3. **训练**：
   - CLI：`python -m qlib.workflow.task.manage -t pool_name wait`。
   - 脚本：`run_task(task_pool="pool_name", task_func=qlib.model.trainer.task_train)`。
   - 多机：多 Worker 共享 Mongo。
4. **收集**：qlib.init() 后，用 Collector/Group/Ensemble 产报表/集成。
5. **完整参考**：examples/model_rolling/task_manager_rolling.py。

### 常见坑位与排查
- **Mongo 问题**：未配置/连接失败 → 查权限、网络、库存在；优化索引提升并发。
- **状态异常**：卡住/未更新 → 验证 before/after_status/query；查 Worker 日志（异常退出/冲突）。
- **重复/替换**：create_task 自动去重；用 replace_task 避免浪费。
- **多机同步**：不一致 → 统一 Qlib 版本/数据路径/环境；参考 GitHub “回测接口”讨论。
- **收集失败**：mlruns 未设/产物缺失 → 确认训练写 Recorder（预测/指标）。
- **通用**：日志优先；版本兼容（PyMongo 4.x+）。

### 小结
- **核心优势**：模板批量标准化复现；Mongo 池支持队列/并发/扩容；Trainer/Collector 端到端，省样板代码；无缝接 Qlib Workflow/qrun/Recorder/Online Serving。
- **场景收益**：批量/分布式实验效率翻倍，减少错误。
- **生态扩展**：API/源码自定义；若需，可基于模型/特征定制脚本模板。

## Qlib Point-In-Time (PIT) Database

基于 Qlib 最新文档（qlib.readthedocs.io/en/latest/advanced/PIT.html，截至 2025 年 11 月 6 日）。PIT 数据库是 Qlib 的高级功能，用于处理金融数据的时点一致性，避免历史回测中的数据泄露（data leakage）。笔记按文档逻辑组织，精炼关键概念、数据格式和实战要点；补充解读、落地提示及局限性。重点：文件-based 设计，适用于季度/年度因子数据（如财务报表）。

### 概述（Introduction）
- **核心概念**：PIT 数据确保历史分析（如回测）时，只使用当时可得的版本。金融数据（如报表）常被多次修正，使用最新版会导致数据泄露（e.g., 回测 2020 年 1 月 1 日策略，只用截至该日的旧数据）。
- **问题解决**：保持在线交易与历史回测性能一致，避免未来信息污染过去决策。
- **解读**：像“时间旅行”数据库，回测时“回到过去”取数据。适用于任何历史市场分析，尤其量化策略开发。
- **适用场景**：回测交易策略、因子研究，使用多年历史数据时。

### 数据准备（Data Preparation）
- **获取方式**：Qlib 提供爬虫下载金融数据 + 转换器转 Qlib 格式。
- **步骤**：参考 scripts/data_collector/pit/README.md 下载/转换数据。README 中有额外示例（如自定义数据源）。
- **落地提示**：先安装 Qlib，运行爬虫脚本抓取（如股票报表），然后转换。确保数据源可靠（e.g., Yahoo Finance 或官方 API），处理时区/格式一致性。

### 文件-based 设计（File-based Design）
- **存储格式**：每个特征（feature）用文件存储，包含 4 列：date、period、value、_next。每行对应一个报表声明（statement）。
- **文件名规则**：如 XXX_a.data（年度数据）、XXX_q.data（季度数据）。Qlib 通过文件名识别数据类型（季度/年度）。
- **列含义**（以 XXX_a.data 示例）：
  - date：报表发布日期（e.g., 20070428）。
  - period：报表周期（e.g., 年度用年份整数；季度用 <年><季度索引>，如 200701 表示 2007 年 Q1）。
  - value：实际值（float）。
  - _next：下一条同字段的字节索引（uint32，4294967295 表示无下一条）。
- **数据排序**：按 date 升序排序，便于查询。
- **示例数据**（结构化数组，dtype=[('date', '<u4'), ('period', '<u4'), ('value', '<f8'), ('_next', '<u4')]）：
  ```python
  array([(20070428, 200701, 0.090219  , 4294967295),
         (20070817, 200702, 0.13933   , 4294967295),
         # ... 更多行
         (20191016, 201903, 0.25581899, 4294967295)],
        dtype=[('date', '<u4'), ('period', '<u4'), ('value', '<f8'), ('_next', '<u4')])
  ```
  - 每行 20 字节，便于字节级访问。
- **索引文件**（XXX_a.index）：加速查询。
  - 第一部分：数据起始年份（e.g., 2007）。
  - 剩余：周期首条记录的字节索引数组（uint32）。
  - 示例：
    ```python
    array([0, 20, 40, 60, 100, ..., 1060, 4294967295], dtype=uint32)
    ```
    - 表示每个周期（如 200704）的首条字节位置（e.g., 字节 100 和 80 都对应 200704，记录最早的 100）。
- **关键点**：链式结构（_next）处理多次修正；索引优化查询性能。

### 已知局限性（Known Limitations）
- **适用范围**：当前设计限于季度/年度因子，覆盖多数市场财务报表。但不适合更高频数据（如日频）。
- **文件名依赖**：通过后缀 (_a/_q) 区分类型，需严格遵守。
- **性能优化**：PIT 计算非最优，有潜力提升（e.g., 通过并行或更好算法）。
- **排查提示**：数据泄露疑虑时，检查 date/period 是否匹配历史时点；查询慢时，验证索引完整性。

### 最小可运行套路（落地清单）
1. **准备数据**：运行 scripts/data_collector/pit/README.md 中的爬虫 + 转换脚本，生成 PIT 文件（e.g., features/XXX_a.data 和 .index）。
2. **集成 Qlib**：在策略代码中加载 PIT 数据：
   ```python
   from qlib.data import D  # D 是 DatasetProvider
   pit_data = D.features(instruments=['SH600000'], fields=['XXX_a'], start_time='2020-01-01', end_time='2020-01-01', freq='day', pit=True)  # 指定 pit=True 使用 PIT 模式
   ```
3. **回测验证**：在回测循环中，确保每个交易日只取当时 PIT 值，避免泄露。
4. **扩展**：自定义数据时，遵循 4 列格式 + 排序 + 索引生成（参考 Qlib 源码 qlib/data/storage.py）。

### 小结
- **优势**：消除数据泄露，确保回测真实性（online/offline 一致）；文件-based 简单高效，易扩展。
- **相对普通数据**：非 PIT 用最新版易高估策略性能；PIT 更真实，但文件管理稍复杂。
- **生态扩展**：与 Qlib Workflow/Task Management 集成，用于批量回测；参考 README 示例自定义爬虫。