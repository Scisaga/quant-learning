# Qlib Reinforcement Learning 强化学习

与分类、回归等监督学习不同，强化学习（RL）是一种重要范式：在马尔可夫决策过程（MDP）等假设下，通过与环境直接交互，优化累计数值型奖励。一个完整的 RL 系统包含 4 个要素：
   - 智能体（agent）；
   - 环境（environment）；
   - 策略（policy）；
   - 来自环境的奖励信号（reward）。

![强化学习基本框架](img/qlib-1762358151751.png)
  
智能体能够感知并解释环境、采取动作、通过奖励学习，以追求长期、整体的最大奖励。RL 通过“试错”学习：采样动作→观察结果→学习可产生最优行动的策略。与监督学习不同，RL 并非从标签学习，而是从延迟奖励（time-delayed label）中学习，其目标是“采取能让奖励最大化的行动”。QlibRL 是面向量化投资的 RL 平台，使 RL 算法能在 Qlib 中落地。

补充：
- **为何强调“累计/长期奖励”**：量化交易目标（如回撤、成本、完成率、收益稳定性）往往在多期才体现，RL 的累计回报正契合“跨时段最优”的诉求。
- **为什么是“延迟”**：下单后对收益/冲击的真实影响，常在若干步之后才显现。RL 通过折扣回报（discounted return）将未来反馈传回到当前决策。
- Agent：你的交易策略（例如执行机器人或资产配置器）；
- Environment：市场/撮合机制/可交易约束；
- Policy：具体算法（如 PPO、DQN 等）；
- Reward：把“价格优势、成本、冲击、风控目标”等聚合为标量优化目标。

## Order Execution 订单执行

目标是在给定时间框内，高效执行订单，同时权衡：更优价格、最小化交易成本、降低市场冲击、最大化完成率。可将这些目标编码进奖励函数与动作选择。智能体与市场交互，从市场信息获得状态并决定下一步执行；算法通过试错学习最优执行策略，以最大化期望累计奖励。

**一般设置（General Setting）**：

- **环境**：金融市场（盘口动态、流动性、价格变动、市场状态等）；
- **状态**：给定时间步可获得的信息（盘口价差与深度、历史价格/成交量、波动率等）；
- **动作**：基于状态做出的执行决策（下单数量、价格、时点）；
- **奖励**：衡量执行表现的标量（鼓励价格优势、低成本〔含手续费与滑点〕、低冲击与高完成率）。
- 场景细分：
   - **单资产执行**：关注某一资产的一笔或一批订单，目标同上；
   - **多资产执行**：要同时/序贯地执行多个资产的订单，并处理资产间相互依赖、资金约束等，目标是在组合层面平衡整体表现。

执行是强时序/强约束问题（剩余时间/剩余量/成交量曲线/冲击函数等），奖励与动作耦合复杂，RL 的闭环交互 + 奖励塑形非常匹配。
- **状态/动作建模**：
   - **状态**建议包含：订单剩余量/剩余步数、历史成交速率、盘口微结构特征、波动/价差/流动性代理等；
   - **动作**可离散化为“当前步成交比例/速率档位”，或连续化为“下单量/价格偏移”。离散更稳、连续更细。
- **奖励设计**：常见做法是线性或分段加权：
  - 价格优势（如相对 TWAP/VWAP 的改进）、
  - 成本（手续费、滑点/冲击惩罚）、
  - 完成率/超时惩罚。
  - 不同市场状态下（高/低流动性、波动 regimes）权重可自适应或分桶。

## Portfolio Construction 投资组合构建

组合构建是“选择并分配资产权重”的过程。RL 通过与市场环境交互，在考虑风险管理的同时最大化长期回报。

**一般设置（General Setting）**：
- **状态**：关于市场与组合的当前信息（历史价量、技术指标、其他相关数据）；
- **动作**：对各资产分配的资本权重/权重变动；
- **奖励**：衡量组合表现的指标，可定义为总收益、风险调整收益（如夏普），或最小回撤等。
- 市场实例：
  - 股票、加密货币、外汇等市场均可适用，算法/设置需随市场特性调整。
- 与监督式“打分选股”关系：监督学习多输出“预测分数/预期收益”，再由规则/优化器转成权重；RL 则直接在“权重/换手/成本/风险约束”的闭环中学习策略，能够把交易 frictions 与风控内化到奖励中。
- 建模提示：
  - 状态建议加入：持仓、可用资金、换手与成本预算、行业/风格/风险因子暴露；
  - 动作可为“目标权重向量”或“增量调仓”；
  - 奖励除收益类指标外，建议显式扣除成本/冲击，并加入风险约束的罚项（波动、回撤、因子超限等）。

## QlibRL

![The Framework of QlibRL](img/qlib-1762360750595.png)

QlibRL 覆盖一个 RL 流水线的完整生命周期：从市场模拟器的构建、状态/动作塑形，到策略（Policy）训练，最后在模拟环境里回测策略。它主要基于 Tianshou 与 Gym 实现。
- **环境/状态/动作/奖励** → EnvWrapper 四件套：在 QlibRL 中通过 Simulator / StateInterpreter / ActionInterpreter / Reward 组合得到 gym.Env 兼容环境；这让你能在同一市场模拟下快速替换不同的状态抽取、动作空间与奖励权重，做“变量分离”的实验设计。
- **策略（Policy）**：通常直接复用 Tianshou 的 PPO、DQN 等策略实现；适合先用离散动作 + PPO做稳健基线，再尝试连续动作或自研策略。
- **训练与回测**：用 Trainer 进行并行采样与训练（dummy/subproc/shmem 三种向量化方式），保存 checkpoint 后切到回测组件评估效果（收益、成本、完成率、冲击、回撤等维度齐看）。

### EnvWrapper 环境封装
QlibRL 中的 EnvWrapper 是 gym.Env 的子类，封装了强化学习环境；开发者通常不需要“另写一个 Env”，而是通过替换组成部件（模拟器/解释器/奖励）来塑形环境行为。

- EnvWrapper 是“胶水层”，把下面这些可插拔组件拼成一个标准 Gym 环境：
  - Simulator（市场模拟器）
  - StateInterpreter（状态解释器）
  - ActionInterpreter（动作解释器）
  - Reward（奖励函数）
- 这种分层能把“市场逻辑（撮合/成交/限制）”与“特征/动作/目标”解耦，便于快速做 A/B 实验：换奖励即可改优化目标，换解释器即可改输入/动作空间，Simulator 不必重写。

### Simulator 市场模拟器

Simulator 负责还原交易环境的核心动态，例如**价量演化、流动性限制、撮合/成交规则、成交量阈值**等；它决定 `step()` 如何推进时间、如何更新“剩余订单/剩余时间”等内部状态。
  * **时间粒度**（如 `time_per_step=30min`）、
  * **成交量/价格规则**（VWAP/收盘价/限价）、
  * **约束**（涨跌停/可交易量/成交量阈值）等。
  * 复杂模拟器越贴近实盘，但训练更慢；简化模拟器训练快，现实度略差，可用于快速迭代。

### StateInterpreter 状态解释器

StateInterpreter 负责把原始数据（分钟线、盘口、订单剩余量等）转成**策略网络可用的张量**，例如固定长度的时序窗口；官方示例使用 `FullHistoryStateInterpreter`，并通过 **ProcessedDataProvider** 从本地特征目录读取数据（pickle/缓存）。

* 关键超参：`data_dim`（特征维度）、`data_ticks`（日内采样点，比如 240 分钟）、`max_step`（订单执行允许的最大步数估计）。
* 建议遵循 **PIT（Point-in-Time）** 原则，确保状态里只包含当时可见的信息，避免前视偏差。

### ActionInterpreter 动作解释器

ActionInterpreter 将策略输出（离散类别或连续向量）转换为**可执行的交易动作**，如“当前步下单比例/速率”。示例里常用 `CategoricalActionInterpreter`，支持用 **整数 n** 自动生成等分档位（如 `[0, 1/n, ..., 1]`），或自定义具体网格。

* **离散动作**更稳、更易收敛，适合建立基线（如 15 档成交比例）；
* **连续动作**表达力强，但探索更难、超参更敏感；
* `max_step` 也会在解释器侧用于边界/归一化。

### Reward 奖励

Reward 将多目标（价格优势、交易成本/冲击、完成率、时限惩罚）压缩成**单一标量**来优化。示例里有 `PAPenaltyReward`：通过 `penalty` 系数抑制短时过度成交，把**冲击/成本**内化进优化目标。
* `Reward = +价格优势 − 成本/冲击 − 超时罚 + 完成率奖励`；
* 欠拟合→降低惩罚/简化动作空间；过拟合→早停/正则/更长验证序列。

### Policy & Network 策略与网络

QlibRL 直接复用 **Tianshou** 的 Policy（如 **PPO、DQN**），网络侧示例常用 **Recurrent（RNN）** 以利用时序依赖。策略与网络的构造参数通过配置传入。
* 起步推荐“**离散动作 + PPO + RNN**”；
* 若要替换为自研方法，只需提供兼容 Tianshou Policy 的实现即可插入流水线。

### TrainingVessel / Trainer 训练容器与训练器

**TrainingVessel** 保存构造环境所需的组件“配方”，便于在并行采样时**复制出多个环境**；**Trainer** 负责训练调度（采样、学习、验证、早停、检查点、指标写入）。配置里可设 `concurrency` 与 `parallel_mode`（`dummy/subproc/shmem`）来控制向量化环境与并行策略。
* `dummy` 最稳、便于调试；`subproc/shmem` 能显著提升采样吞吐，但要关注随机性与进程间资源；
* **Metrics/Checkpoint/EarlyStopping** 是一等公民：建议固定验证周期、设置早停耐心，避免无效训练。
  * 当你监控的验证指标（如 val_loss、val_reward、Sharpe 等）连续若干次评估都没有变好，就提前停止训练；这个“若干次”的阈值就叫耐心。

### Backtest 回测

训练后的策略可在**同一模拟环境**回测；回测配置中需保持**解释器/网络/策略**与训练侧一致，并用 `weight_file` 指向已保存权重；也可并行运行多只订单/多策略（如与 TWAP 对照）。
* 回测层还可设置**成交价规则**（如 `If($close==0,$vwap,$close)`）与**成交量阈值**（累计/当前），约束更贴近交易所微结构

### 自定义新场景步骤
1. **定 Simulator**：选时间粒度、成交/撮合规则、流动性/阈值、可交易性限制；
2. **做 StateInterpreter**：明确 `data_dim/data_ticks/max_step` 与 PIT 特征来源；
3. **定 ActionInterpreter**：先离散（等分网格/业务自定义档位），再酌情尝试连续；
4. **写 Reward**：把收益、成本/冲击、完成率/回撤、时限等组合为标量；
5. **选 Policy/Network**：PPO+RNN 起步；
6. **设 Trainer**：并行模式、采样节奏、早停、检查点；
7. **做 Backtest**：与训练配置一致，加载 `weight_file`，并与规则基线（TWAP/VWAP）对照。
上述每一步都与框架的四大部件和训练/回测器一一对应，改一处不必重写全栈。

## 单资产订单执行示例 Single-Asset Order Execution, SAOE

为什么使用`离散动作 + RNN + PPO`？
- **离散动作**：搜索空间更稳、训练更容易收敛；先得到可用基线，再考虑连续动作（如高斯策略）。
- **RNN**（Recurrent）：执行任务强时序依赖；RNN 能利用历史轨迹。
- **PPO**：鲁棒的 on-policy 算法，Tianshou 生态成熟，易于复用与替换。
- 这些设计与 QlibRL 的“解释器分层 + Tianshou 策略”的框架天然契合。

### 训练配置 train_config.yml
```yml
simulator:
  # 每个 step 覆盖的时间长度（分钟）。与日内分钟数共同决定一次订单可分成多少步
  time_per_step: 30
  # 成交量上限；null 表示不限。若设为 0~1 的浮点，则以上市成交量百分比为上限（越小越“稳健”）
  vol_limit: null

env:
  # 并发的环境实例数（VectorEnv）；先用 1 跑通，再提高实现采样加速
  concurrency: 1
  # 环境并行模式：dummy（单进程，稳定）/ subproc 或 shmem（多进程/共享内存，更快）
  parallel_mode: dummy

action_interpreter:
  # 把策略输出的“离散动作索引”映射为实际可执行的“下单比例/速率”
  class: CategoricalActionInterpreter
  kwargs:
    # 候选动作集合。可给定一个列表 [a1,...,aL]；也可给定整数 n，自动生成 [0, 1/n, 2/n, ..., 1]
    values: 14
    # 订单执行允许的最大步数（上界估计，用于归一化/边界处理）
    max_step: 8
  module_path: qlib.rl.order_execution.interpreter

state_interpreter:
  # 将行情/上下文加工成可喂给策略网络的时序张量
  class: FullHistoryStateInterpreter
  kwargs:
    # 每个时间点的特征维度（需与你的特征文件一致）
    data_dim: 6
    # 日内总记录数（分钟级常见为 240；若有盘前/盘后则据实调整）
    data_ticks: 240
    # 订单执行的最大步数上界（例如 390/30≈13，可按 time_per_step 估算）
    max_step: 8
    # “已处理特征”的提供者（本地 pickle 目录）
    processed_data_provider:
      class: PickleProcessedDataProvider
      module_path: qlib.rl.data.pickle_styled
      kwargs:
        # 你的特征数据目录（需要你预先生成）
        data_dir: ./data/pickle_dataframe/feature
  module_path: qlib.rl.order_execution.interpreter

reward:
  # 奖励函数：对“短时大成交量”进行惩罚（可理解为把冲击/成本写入奖励）
  class: PAPenaltyReward
  kwargs:
    # 惩罚系数；越大越抑制瞬时成交量（可与真实滑点/冲击标定）
    penalty: 100.0
  module_path: qlib.rl.order_execution.reward

data:
  source:
    # 训练用订单切片目录（需你准备好若干订单样本）
    order_dir: ./data/training_order_split
    # 训练回放所需的行情/特征数据目录
    data_dir: ./data/pickle_dataframe/backtest
    # 日内时间索引总数（与 data_ticks 一致）
    total_time: 240
    # 日内开始/结束索引（0 到 total_time）
    default_start_time: 0
    default_end_time: 240
    # 处理后特征维度（与 data_dim 一致）
    proc_data_dim: 6
  # 数据加载并发与队列大小（先设保守值，稳定后再调大）
  num_workers: 0
  queue_size: 20

network:
  # 序列模型（RNN），适合订单执行的强时序依赖
  class: Recurrent
  module_path: qlib.rl.order_execution.network

policy:
  # 策略算法：PPO（也可替换为 DQN 等；建议先用 PPO 跑稳基线）
  class: PPO
  kwargs:
    # 学习率（根据稳定性/收敛速度调优）
    lr: 0.0001
  module_path: qlib.rl.order_execution.policy

runtime:
  # 随机种子（便于复现实验）与是否启用 CUDA（先 CPU 跑通，再切 GPU）
  seed: 42
  use_cuda: false

trainer:
  # 训练总 epoch（示例给的是 2，真实训练需更大）
  max_epoch: 2
  # 每次采样后重复更新次数（稳定性与速度的权衡）
  repeat_per_collect: 5
  # 早停耐心值（若验证集长时间不提升则提前停止）
  earlystop_patience: 2
  # 每轮采样的 episode 数量（采样越多更新越稳；同时训练更耗时）
  episode_per_collect: 20
  # 小批量大小
  batch_size: 16
  # 每隔多少个 epoch 做一次验证
  val_every_n_epoch: 1
  # 检查点保存目录与保存频率（迭代粒度）
  checkpoint_path: ./checkpoints
  checkpoint_every_n_iters: 1
```

### 回测配置 backtest_config.yml
```yml
# 回测订单清单（CSV），需先准备（可与训练不同的订单集作泛化验证）
order_file: ./data/backtest_orders.csv

# 日内回测时间窗（字符串形式，需与行情时间轴一致）
start_time: "9:45"
end_time: "14:44"

qlib:
  # 1 分钟级原始数据（二进制/缓存目录）
  provider_uri_1min: ./data/bin
  # 回测检索用的特征根目录（pickle）
  feature_root_dir: ./data/pickle
  # “今日即可获得的信息”对应的特征列（务必保证信息时点一致 PIT）
  feature_columns_today: [
    "$open", "$high", "$low", "$close", "$vwap", "$volume",
  ]
  # “昨日信息”生成的特征列（_v1 表示向后平移 1 天）
  feature_columns_yesterday: [
    "$open_v1", "$high_v1", "$low_v1", "$close_v1", "$vwap_v1", "$volume_v1",
  ]

exchange:
  # 买卖限制表达式（示例：若收盘价为 0 则限制）
  limit_threshold: ['$close == 0', '$close == 0']
  # 买卖成交价规则（示例：若 $close 为 0 则用 vwap，否则用收盘价）
  deal_price: ["If($close == 0, $vwap, $close)", "If($close == 0, $vwap, $close)"]

volume_threshold:
  # 买卖共同的成交量限制。"cum" 表示按时间累计（示例：不超过日内 20% 累计量）
  all: ["cum", "0.2 * DayCumsum($volume, '9:45', '14:44')"]
  # 买入的“当前时刻”成交量阈值（"current" 表示非累计、实时）
  buy: ["current", "$close"]
  # 卖出的“当前时刻”阈值（同上）
  sell: ["current", "$close"]

strategies:
  # 规则基线：30 分钟 TWAP（用于与 RL 策略对照）
  30min:
    class: TWAPStrategy
    module_path: qlib.contrib.strategy.rule_strategy
    kwargs: {}

  # RL 策略：SAOEIntStrategy（需与训练侧的解释器/网络/策略保持一致）
  1day:
    class: SAOEIntStrategy
    module_path: qlib.rl.order_execution.strategy
    kwargs:
      # 状态解释器（参数需与训练时一致）
      state_interpreter:
        class: FullHistoryStateInterpreter
        module_path: qlib.rl.order_execution.interpreter
        kwargs:
          max_step: 8
          data_ticks: 240
          data_dim: 6
          processed_data_provider:
            class: PickleProcessedDataProvider
            module_path: qlib.rl.data.pickle_styled
            kwargs:
              data_dir: ./data/pickle_dataframe/feature

      # 动作解释器（需与训练时一致）
      action_interpreter:
        class: CategoricalActionInterpreter
        module_path: qlib.rl.order_execution.interpreter
        kwargs:
          values: 14
          max_step: 8

      # 网络（与训练保持一致；通常无需在回测侧再传其他超参）
      network:
        class: Recurrent
        module_path: qlib.rl.order_execution.network
        kwargs: {}

      # 策略（PPO），权重要指向训练产物；如删除 weight_file 则为“随机策略”对照
      policy:
        class: PPO
        module_path: qlib.rl.order_execution.policy
        kwargs:
          lr: 1.0e-4
          # 训练完成后生成的最新模型（确保路径正确）
          weight_file: ./checkpoints/latest.pth

# 回测并发环境数（提高吞吐；先确保正确性再调大）
concurrency: 5
```

### 训练 / 回测

* **数据目录**
  1. **训练侧**

     * `./data/training_order_split/`：训练订单样本（按日/按单拆分）。
     * `./data/pickle_dataframe/feature/`：状态解释器要用的**特征 pickle**。
     * `./data/pickle_dataframe/backtest/`：训练数据段的回放行情（示例配置引用）。

  2. **回测侧**

     * `./data/bin/`：1 分钟级原始数据（二进制/缓存格式）。
     * `./data/pickle/`：与回测特征检索相关的 pickle。
     * `./data/backtest_orders.csv`：回测订单清单。

* **训练**

  ```bash
  python -m qlib.rl.contrib.train_onpolicy.py --config_path train_config.yml
  ```
* **回测**

  ```bash
  python -m qlib.rl.contrib.backtest.py --config_path backtest_config.yml
  ```

  训练完成后，回测会在 `SAOEIntStrategy` 中加载 `./checkpoints/latest.pth` 来评估 RL 策略表现

* **执行建议**

  1. **先就地复制官方示例配置**，仅改动：

     * `data_dir` / `order_dir` / `provider_uri_1min` / `feature_root_dir` / `order_file` 等本地路径；
     * 保持 `FullHistoryStateInterpreter + CategoricalActionInterpreter + PAPenaltyReward + Recurrent + PPO` 组合不变。
  2. **先 `dummy + concurrency=1` 跑通**，确认能产出 `./checkpoints/latest.pth`；
  3. **回测** 引用该权重，对比 `TWAPStrategy`；
  4. 再逐步：放大 `max_epoch`、提高并行、尝试 `subproc/shmem`、调 `penalty/values/max_step` 等超参。

## 常见误区 / 对策 / 排查清单

通用：
1. **只追单步最优，忽略长期目标** → 奖励要体现跨期指标（例如期末完成率/最终成交均价相对基准的优势、最大回撤等），并配合折扣因子。
2. **状态信息不全** → 扩充到“盘口/流动性/波动/剩余量/剩余时间”等关键变量，必要时加入 regime（市场状态）提示，减少“同态不同价”的学习难度。
3. **忽略交易 frictions** → 将手续费、滑点/冲击、资金/仓位上限，显式写入奖励或约束，避免策略“纸面最优”。
4. **多资产协同欠缺** → 在多资产执行/配置时，把相关性/资金联动/风险预算纳入状态与奖励，才能真正学到组合层最优。

执行层：
1. **配置字段与版本不匹配**

   * 现象：示例 yml 报未知字段/路径不存在。
   * 排查：对照你安装的 qlib 版本、Tianshou/Gym 版本；路径按本机数据改；必要时参考社区 issue（例如“示例 yml 过时/数据缺失”问题）。
2. **数据目录未建/未生成**

   * 现象：`processed_data_provider` 读不到数据；回测拿不到 1 分钟 bin 或 pickle。
   * 排查：先跑你的特征生成脚本/CSV→Qlib 格式转换→健康检查，再启动 RL。
3. **并行导致随机性**

   * 现象：`subproc/shmem` 下结果有波动。
   * 排查：固定 `seed`、记录环境；先用 `dummy` 验证稳定性，再提升并行度。
4. **训练无收敛/奖励波动大**

   * 调低 `values`（减少离散动作档位）→ 降低 `penalty` 或做 reward shaping → 增加 `episode_per_collect` 以稳定更新。
5. **回测策略不生效**

   * 检查 `policy.kwargs.weight_file` 是否指向训练产物；若为空即为“随机策略”对照实验。

## 参考
- [Reinforcement Learning in Quantitative Trading](https://qlib.readthedocs.io/en/latest/component/rl/overall.html)
- [QlibRL Quick Start](https://qlib.readthedocs.io/en/latest/component/rl/quickstart.html)
- [The Framework of QlibRL](https://qlib.readthedocs.io/en/latest/component/rl/framework.html)