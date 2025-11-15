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

## Qlib Point-In-Time (PIT) Database

PIT 数据库是 Qlib 的高级功能，用于处理金融数据的时点一致性，避免历史回测中的数据泄露（data leakage）。

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