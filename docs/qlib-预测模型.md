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