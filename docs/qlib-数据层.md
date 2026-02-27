# Data Layer 数据层

参考官方文档：[Qlib Data Layer](https://qlib.readthedocs.io/en/latest/component/data.html)。

Qlib 的数据层（Data Layer）解决的核心问题是：**把“金融市场数据 + 因子表达式 + 训练/回测所需切片”统一成一套高性能、可复用、可缓存的数据访问与加工链路**。在你的研究中，数据层通常决定了两件事：

- 你能不能**稳定、可复现**地拿到同一份特征/标签数据（避免对齐错误、未来函数、不同机器结果不一致）。
- 你能不能在**大规模标的×长时间窗**下仍然“拉得动数据”（缓存、磁盘格式、表达式复用）。

下面按“架构 → 概念 → 文件格式 → API 用法 → Loader/Handler/Dataset → 缓存 → 常见坑”把 Data Layer 补全。

## 1. 架构一览：从原始数据到模型输入

可以把数据层理解为一条流水线：

1. **Provider（数据提供者）**：负责提供“日历 / 股票池 / 原始字段（OHLCV 等）/ 财务字段”等基础数据。
2. **Expression Engine（表达式引擎）**：把 `$close`、`Ref($close, -1)`、`Mean($close, 5)` 这类表达式解析成可计算的序列。
3. **Data API（查询入口）**：`qlib.data.D` 提供统一查询：拉字段、拉日历、拉股票池、按表达式计算因子。
4. **DataLoader（批量加载器）**：把“特征表达式集合 + 标签表达式集合 + 时间段 + 股票池”一次性拉成 DataFrame。
5. **DataHandler（数据处理器）**：在 DataLoader 之上增加可复用的数据处理链（缺失值、标准化、中性化、Winsorize 等），并区分 train/infer 的拟合与应用。
6. **Dataset（数据集）**：把 Handler 输出切成 train/valid/test segments，提供给模型训练/评估/回测。
7. **Cache（缓存）**：缓存表达式结果、缓存数据集切片，避免重复计算/重复 IO。

## 2. 核心概念：看懂 Qlib 的数据“坐标系”

### 2.1 Calendar（交易日历）

- Qlib 的大多数数据都**按统一日历对齐**，例如日频用 `calendars/day.txt`。
- 日历既影响“能查到哪些日期”，也影响 `.bin` 文件里每个位置代表哪一天。
- 常见文件：`day.txt`（历史交易日）、`day_future.txt`（可能包含未来交易日，用于扩展/占位）。

### 2.2 Instrument（标的）与股票池（Universe）

- “标的”是单个代码；“股票池”是标的集合（例如 `csi300`、`csi500`、`all`）。
- 股票池文件通常是三列 TSV：`instrument<TAB>start_date<TAB>end_date`，表示该标的在该池里的有效区间（避免把未上市/退市期间混入）。

例如本仓库 `data/qlib_data/cn_data/instruments/all.txt` 的前几行就是这种格式。

### 2.3 Field / Feature（字段 / 特征）

- “字段”通常指原始字段（如 open/close/volume/factor）。在表达式里以 `$close` 这类占位符出现。
- “特征”更广：可以是原始字段，也可以是表达式计算出来的序列（如 `Mean($close, 5)`）。

### 2.4 Frequency（频率）

- 最常用：日频（`day`）、分钟（`1min`）。不同频率对应不同日历/文件后缀。
- 同一套 API，频率不同会影响：数据粒度、对齐规则、性能与缓存体积。

### 2.5 Label（标签）与对齐（避免未来函数）

Qlib 官方示例里经常用：`Ref($close, -2) / Ref($close, -1) - 1` 作为 label。

- `Ref(x, k)` 是时间位移：`k<0` 代表“向未来取值”（更晚的时间点），因此更容易引入未来信息。
- 正确与否取决于你的“交易假设”：例如你是否允许在 T 日收盘后看到 T 日收盘价、在 T+1 开盘交易等。

建议：在文档/实验里明确写清楚“特征截止到何时、标签对应哪段收益、买卖发生在何时”，否则很容易对齐错。

## 3. Qlib 数据目录结构：.bin、股票池、日历、财务数据

以本仓库的 CN 日频数据为例：`data/qlib_data/cn_data/` 典型结构：

```
data/qlib_data/cn_data/
  calendars/
    day.txt
    day_future.txt
  instruments/
    all.txt
    csi300.txt
    csi500.txt
    ...
  features/
    sh600000/
      open.day.bin
      close.day.bin
      volume.day.bin
      factor.day.bin
      ...
    ...
  financial/
    sh600000/
      xxx_q.data
      xxx_q.index
      ...
```

### 3.1 features 下的 `.day.bin` 是什么

- 每个标的一个目录，每个字段一个 `.bin` 文件（例如 `close.day.bin`）。
- `.bin` 本质上是一个**按交易日历对齐的定长数值序列**（通常可被内存映射读取），这样可以避免频繁解析 CSV，速度很高。
- 缺失值（停牌、无成交等）一般用 `NaN` 表示。

> 小提示：Windows 文件系统大小写不敏感，所以你看到的 `SZ000001` / `sz000001` 目录看起来“都存在”；在 Linux/macOS 上通常会统一成小写路径。

### 3.2 instruments 的 start/end 有什么用

当你查询 `instruments="csi300"` 时，Qlib 不只是在“挑出一批代码”，还会用 start/end 把不可交易区间剔除（例如未上市期间），避免训练集里出现大量无意义 NaN。

### 3.3 financial 下的 `.data/.index`

财务数据往往是**低频、稀疏更新**（季度/年报）。Qlib 常用 `.data + .index` 的形式存储，便于在日频/分钟频上做对齐或 forward-fill（具体对齐方式取决于 provider/表达式）。

## 4. 数据准备与更新（把外部数据变成 Qlib 可用的数据）

如果你只是想跑通研究/回测，本仓库已经准备了 `data/qlib_data/*`，可以直接跳到下一节初始化。

如果你要**下载示例数据**或**把自己的 CSV/Parquet 转成 Qlib .bin**，本仓库在 `backend/qlib/scripts/` 下提供了一套脚本（多数来自 Qlib 官方实现并用 `fire` 封装）。

### 4.1 下载示例数据（CN/US、日频/分钟）

在仓库根目录执行：

```bash
# CN 日频
python backend/qlib/scripts/get_data.py qlib_data --target_dir data/qlib_data/cn_data --region cn

# CN 1min（可选）
python backend/qlib/scripts/get_data.py qlib_data --target_dir data/qlib_data/cn_data_1min --region cn --interval 1min

# US 日频
python backend/qlib/scripts/get_data.py qlib_data --target_dir data/qlib_data/us_data --region us
```

> 说明：示例数据属于“公开样例”，更新频率与完整性取决于上游；如果你需要更完整/更高频/更新更及时的数据，通常需要接入自己的数据源并走 dump 流程。

### 4.2 从 CSV/Parquet dump 成 Qlib 数据目录（.bin + 日历 + 股票池）

你可以把外部数据文件（按标的拆分的 `.csv` 或 `.parquet`）dump 到一个新的 `provider_uri` 目录：

```bash
python backend/qlib/scripts/dump_bin.py dump_all --data_path data/stock_data --qlib_dir data/qlib_data/my_data --date_field_name date --symbol_field_name symbol --include_fields open,high,low,close,volume,factor --freq day
```

关键约定（非常重要）：

- 每条记录至少要包含 `symbol`、`date`（字段名可用参数指定）。
- OHLCV 建议用数值列；停牌/无成交用 `NaN` 表示。
- 如果你要做复权口径一致的研究，建议提供 `factor`（或你自己的调整因子），并在研究里写清楚口径。

### 4.3 数据健康检查（缺失、跳变、因子缺失）

dump 后建议先做一次快速体检：

```bash
python backend/qlib/scripts/check_data_health.py --qlib_dir data/qlib_data/my_data --freq day
```

脚本会检查：OHLCV 字段缺失、缺失值数量、异常跳变（价格/成交量）、`factor` 缺失等。

### 4.4 PIT 与财务数据（可选，高阶）

如果你使用财务/公告类数据，建议引入 PIT（Point-In-Time）思想：**在每个交易日只能看到当时已公开的数据**，避免把未来披露的信息泄漏到历史训练集。本仓库提供了 `backend/qlib/scripts/dump_pit.py` 用于构建 PIT 版本的数据（具体字段与口径需结合你的数据源设计）。

## 5. 初始化与最小可用查询（Data API）

### 5.1 init：告诉 Qlib 数据在哪里

在本仓库根目录下，最小初始化示例：

```python
import qlib
from qlib.constant import REG_CN

qlib.init(provider_uri="data/qlib_data/cn_data", region=REG_CN)
```

如果你使用的是你自己下载到 `~/.qlib/qlib_data/cn_data` 的数据，把 `provider_uri` 换成对应路径即可。

### 5.2 常用查询：日历、股票池、字段/表达式

```python
from qlib.data import D

# 交易日历
cal = D.calendar(start_time="2020-01-01", end_time="2020-01-10", freq="day")

# 股票池（返回 instrument 列表）
inst = D.list_instruments(instruments="csi300", start_time="2020-01-01", end_time="2020-01-10")

# 拉原始字段 + 表达式字段
df = D.features(
    instruments=["SH600000", "SZ000001"],
    fields=["$open", "$close", "Ref($close, -1) / $close - 1", "Mean($volume, 5)"],
    start_time="2020-01-01",
    end_time="2020-03-01",
    freq="day",
)
```

`D.features` 的返回通常是 MultiIndex（`datetime`, `instrument`）的 DataFrame：这是 Qlib 把时间序列与截面数据统一在一个表里的关键设计。

### 5.3 如何判断“数据更新到哪天”

不要依赖网上的“预置数据截止日期”描述，直接看日历文件即可：`calendars/day.txt` 的最后一行就是当前数据的最新交易日（本仓库数据会随着你下载/更新而变化）。

## 6. 表达式引擎：写因子时你需要知道的规则

### 6.1 `$field` 占位符与常用算子

- `$open/$close/$high/$low/$volume/$factor`：典型原始字段（不同数据集字段可能略有差异）。
- `Ref(x, k)`：时间位移。
- `Mean(x, n)`、`Std(x, n)`、`Max(x, n)`、`Min(x, n)`：滚动窗口算子。
- 一些表达式支持截面运算（如 rank/归一化），但具体可用算子以你安装的 Qlib 版本为准。

### 6.2 价格复权与 `$factor`

很多数据集会提供一个调整因子（`$factor`）。常见约定是：原始字段已经做了某种复权/归一化；你需要“还原价格”或“统一到可比尺度”时再结合 `$factor` 做转换。

建议：在研究里明确你使用的是**前复权/后复权/不复权**以及标签收益的计算口径，否则同一个因子在不同口径下可能表现差异很大。

## 7. DataLoader：把“表达式集合”批量拉成训练表

`QlibDataLoader` 的核心价值是：一次性配置 feature/label 表达式，统一加载。

```python
from qlib.data.dataset.loader import QlibDataLoader

fields = ["$close", "Ref($close, -1) / $close - 1", "Mean($volume, 5)"]
names = ["CLOSE", "RET_1D", "VOL_MA5"]
labels = ["Ref($close, -2) / Ref($close, -1) - 1"]
label_names = ["LABEL"]

dl = QlibDataLoader(config={"feature": (fields, names), "label": (labels, label_names)})
df = dl.load(instruments="csi300", start_time="2017-01-01", end_time="2020-12-31")
```

当你的特征/标签数量很多时（如 Alpha158/Alpha360），用 Loader 能避免重复写 `D.features(...)`。

## 8. DataHandler：在 Loader 之上复用“加工逻辑”

以 `DataHandlerLP` 为代表的一类 Handler 会把处理逻辑拆为两步：

- **fit（学习）**：只在训练区间统计参数（例如均值方差、分位点、行业中性化回归系数等）。
- **process（应用）**：把学到的参数应用到 train/valid/test（避免用 test 的分布信息“泄漏”到训练）。

最常见的用法是直接使用 Qlib 内置 Handler（如 `Alpha158`、`Alpha360`），它们已经把一套特征表达式、标签表达式和部分处理器串好了。

```python
from qlib.contrib.data.handler import Alpha158

h = Alpha158(
    start_time="2008-01-01",
    end_time="2020-08-01",
    fit_start_time="2008-01-01",
    fit_end_time="2014-12-31",
    instruments="csi300",
)

feat = h.fetch(col_set="feature")
lab = h.fetch(col_set="label")
```

## 9. Dataset：切分 train/valid/test，并喂给模型

最常见的是 `DatasetH`（Handler + Segments）。

```python
from qlib.data.dataset import DatasetH

ds = DatasetH(
    handler=h,
    segments={
        "train": ("2008-01-01", "2014-12-31"),
        "valid": ("2015-01-01", "2016-12-31"),
        "test": ("2017-01-01", "2020-08-01"),
    },
)
train_df = ds.prepare("train")
```

`prepare()` 取出来的就是“模型输入表”（特征列 + 标签列），你可以直接喂给传统 ML（GBDT/Linear）或进一步 reshape 给深度模型。

## 10. Cache：为什么你反复跑实验会越来越快

Qlib 在数据层提供多级缓存，常见的两个方向：

- **Expression cache**：把表达式计算结果缓存下来（尤其是滚动窗口因子）。
- **Dataset cache**：把某次 `Dataset.prepare(...)` 的结果缓存下来（尤其在回测/调参时重复取同一段数据）。

如果你发现“第一次很慢、第二次明显变快”，通常就是缓存生效了。研究时建议：缓存目录与数据目录分开管理，便于清理与复现。

## 11. 常见坑与排查清单

1. **查不到数据 / 返回空表**：优先检查 `qlib.init(provider_uri=...)` 指向的数据目录是否正确、`region` 是否匹配。
2. **股票池为空或数量异常**：检查 `instruments/*.txt` 的 start/end 是否覆盖你的查询区间。
3. **特征全是 NaN**：检查日历是否对齐、是否把停牌视为 NaN、是否字段名写错（如 `$vwap` 数据集未必有）。
4. **标签疑似“未来函数”**：重点检查 `Ref(..., k)` 的方向、交易时点假设、以及你在训练时是否把未来信息漏进了特征。
5. **数据截止日期与预期不一致**：直接查看 `calendars/day.txt` 最后一行，而不是依赖外部说明。
