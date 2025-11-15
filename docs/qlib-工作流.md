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