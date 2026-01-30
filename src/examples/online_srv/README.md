# online_srv（在线更新与在线管理示例）

`examples/online_srv/` 这一组示例关注的是“**研究流程的在线化**”：当交易日不断推进时，如何持续更新模型预测、按规则切换/重训模型，并为下一交易日准备交易信号（signals）。

它不是一个对外提供 HTTP/gRPC 的“线上推理服务”，而是面向 **离线/准实时**（例如日频收盘后跑一遍）的在线管理与自动化流程示例。

---

## 你会学到什么

online_srv 主要覆盖两类能力：

1. **只更新预测（不换模型）**：模型已训练好，并被标记为“online”，每天只需要把预测补到最新日期。
2. **在线管理（会换模型/会重训）**：用 `OnlineManager` 管理一组“在线策略（OnlineStrategy）”，按固定频率生成新任务、训练新模型、切换 online 模型，并产出交易信号；同时支持历史回放（simulate）来验证流程。

---

## 核心概念（对应到 Qlib 代码）

- **online model**：在某个时点被选中用于对外输出的“决策模型”。在 Qlib 中通常表现为某个实验（experiment）下的若干 recorder，被打上 tag：`online_status=online`。
  - 相关实现：`qlib/workflow/online/utils.py` 里的 `OnlineToolR`（基于 Recorder/MLflow tags 管理 online/offline）。
- **在线预测更新**：当数据更新到新交易日时，对“online 模型”补充新增区间的 `pred.pkl`（不必重训）。
  - 相关实现：`OnlineToolR.update_online_pred()` → `PredUpdater`（见 `qlib/workflow/online/update.py`）。
- **OnlineManager**：管理多个 `OnlineStrategy`，并提供：
  - `first_train()`：初始训练并设定第一批 online 模型
  - `routine()`：一次例行更新（典型顺序：准备新任务→训练→切换 online→更新 online 预测→准备 signals）
  - `simulate()`：在历史时间轴上回放 routine，用于验证/回测
  - 相关实现：`qlib/workflow/online/manager.py`
- **RollingStrategy**：一个内置 OnlineStrategy 示例，使用 `RollingGen` 按时间滚动生成任务，并通常选择“最新窗口”模型作为 online 模型。
  - 相关实现：`qlib/workflow/online/strategy.py`

---

## 示例 1：只更新在线预测（update_online_pred.py）

文件：`examples/online_srv/update_online_pred.py`

适用场景：你已经训练好了一个模型，希望每天只做“把预测补到最新日期”，用于日频信号/盘后选股等。

它做了两件事：

1. `first_train()`：训练一次模型，并把对应 recorder 标记为 online（`reset_online_tag`）。
2. `update_online_pred()`：对所有 online recorder 调用 `PredUpdater`，把 `pred.pkl` 更新到最新交易日。

运行方式（在仓库根目录）：

```bash
python src/examples/online_srv/update_online_pred.py first_train
python src/examples/online_srv/update_online_pred.py update_online_pred
```

参数提示：

- `--provider_uri` / `--region`：数据路径与市场区域（默认 `~/.qlib/qlib_data/cn_data` / `REG_CN`）。
- `--experiment_name`：实验名（默认 `online_srv`，会写入 `mlruns/`）。

---

## 示例 2：在线管理（滚动任务 + 例行 routine）（rolling_online_management.py）

文件：`examples/online_srv/rolling_online_management.py`

这是一个更“工程化”的例子：通过 `OnlineManager + RollingStrategy` 管理滚动任务，覆盖以下过程：

1. **first_run**：重置旧结果 → 初始训练 → 设定 online 模型 → dump manager 状态到本地文件
2. **routine**：加载上次保存的 manager → 执行一次例行更新（生成新任务→训练→切换 online→更新 online 预测→准备 signals）→ 再次 dump
3. **add_strategy**：向 OnlineManager 增加新的策略（例如新增一种模型），并做新策略的 first_train

运行方式：

```bash
python src/examples/online_srv/rolling_online_management.py first_run
python src/examples/online_srv/rolling_online_management.py routine
python src/examples/online_srv/rolling_online_management.py add_strategy
```

重要说明（MongoDB / Trainer 选择）：

- 该脚本默认 `trainer=DelayTrainerRM()` 且 `task_url="mongodb://10.0.0.4:27017/"`。
  - `*RM` 结尾的 Trainer 通常用于 **任务管理/分布式训练**（依赖 MongoDB 作为任务池后端）。
  - 如果你本机没有 MongoDB 或 URL 不可用，可以：
    - 直接改参数为本机 Mongo：`--task_url="mongodb://localhost:27017/"`；或
    - 改用不依赖 Mongo 的 Trainer（例如 `TrainerR` / `DelayTrainerR`），需要在代码里替换默认 trainer（脚本注释已说明可选 Trainer 类型）。

状态文件：

- 脚本会把 OnlineManager dump 到 `.RollingOnlineExample`（当前工作目录下），以便下一次 `routine` 继续增量运行。

---

## 示例 3：历史回放模拟 online 流程并回测（online_management_simulate.py）

文件：`examples/online_srv/online_management_simulate.py`

这个脚本用于“像真实 online 一样”沿历史日历推进：

- 调用 `OnlineManager.simulate(end_time=...)` 自动跑完从 `begin_time` 到 `end_time` 的一系列 routine。
- `get_signals()` 得到最终信号，并用 `TopkDropoutStrategy + backtest_daily` 做一个示例回测与风险分析。

运行方式：

```bash
python src/examples/online_srv/online_management_simulate.py main
```

同样注意 MongoDB 配置与 Trainer 选择（默认代码里提供 `TrainerR` / `TrainerRM` 选项）。

---

## 结果在哪里看（mlruns / signals）

online_srv 的结果通常分两部分：

- **训练/预测产物**：写入 `mlruns/`（MLflow tracking），包含模型参数、`pred.pkl`、task 配置等。
- **signals**：由 `OnlineManager.get_signals()` 生成（通常是一个以 `datetime` 为索引的分数序列/面板），用于下一周期的交易决策。

如果你想把 signals 接入自己的回测/交易逻辑，优先从 `get_signals()` 的输出开始对接。

---

## 常见坑与建议

- **实验名/路径**：这些脚本会创建/读取 experiment（如模型类名、`exp_name` 等），建议你先清理旧的 `mlruns/` 或换一个 `--experiment_name/--exp_name`，避免混淆。
- **在线预测更新的前提**：只有 recorder 中存在 `pred.pkl` 时，`PredUpdater` 才能增量更新；否则会被跳过（日志会提示）。
- **滚动更新频率**：`RollingGen(step=...)` 决定多久生成一次新窗口任务；`step` 越小越“频繁重训”，开销也越大。
- **Windows 清理**：本目录提供的 `Makefile` 主要面向 Linux/macOS；Windows 上清理可用 PowerShell：
  - `Remove-Item -Recurse -Force .\\mlruns, .\\*.pkl -ErrorAction SilentlyContinue`
