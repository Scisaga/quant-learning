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