## 安装

python 3.12

```shell
pip uninstall -y gym
pip install -U gymnasium
pip install -U "pyqlib[all]"

# 验证
python -c "import qlib; print(qlib.__version__)"

# 安装训练库，关闭warning
pip install "pyqlib==0.9.7" "lightgbm==4.4.0" "xgboost==2.1.4" "catboost==1.2.8"
pip install "torch>=2.2,<3" --index-url https://download.pytorch.org/whl/cpu

#
pip install statsmodels
```

## 数据准本

### 默认方法

数据缺失较多，且只更新到2020/09/25，不建议使用
```sh
# 下载A股日线数据
python src/get_data.py qlib_data --target_dir data/qlib_data/cn_data --region cn
# 下载A股分钟线数据
python src/get_data.py qlib_data --target_dir data/qlib_data/cn_data_1min --region cn --interval 1min
# 下载美股数据
python src/get_data.py qlib_data --target_dir data/qlib_data/us_data --region us
```

### 补充方法

> 任何数据源都需要在使用过程中反复验证，并记录可能的缺陷

日线数据
- https://github.com/chenditc/investment_data/releases
   - 更完整的日线数据，每日更新
   - 可能存在的问题：https://github.com/chenditc/investment_data/issues


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

### 常用 API

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