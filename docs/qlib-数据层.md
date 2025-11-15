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