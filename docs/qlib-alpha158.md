# Alpha158

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
* `QlibDataLoader`：把**feature（特征表达式+命名）**和**label（标签表达式+命名）**封装为加载配置，一次性拉取一篮子标的与时间段的数据。

## K 线形态（单日 9 个硬编码特征）

| 因子名   | 表达式（Qlib 语法）                                       | 物理意义（直观解释）              | 典型使用场景         |
| ----- | -------------------------------------------------- | ----------------------- | -------------- |
| KMID  | `($close-$open)/$open`                             | 实体长短（收阳/收阴幅度），反映多空当日净力量 | 短线强弱、次日延续/反转判别 |
| KLEN  | `($high-$low)/$open`                               | 当日总振幅/开盘，波动活跃度          | 波动过滤、风控阈值、形态确认 |
| KMID2 | `($close-$open)/($high-$low+1e-12)`                | 实体占振幅的比例（真假突破）          | 假阳/假阴甄别        |
| KUP   | `($high-Greater($open,$close))/$open`              | 上影线相对开盘                 | 冲高回落/抛压识别      |
| KUP2  | `($high-Greater($open,$close))/($high-$low+1e-12)` | 上影线占振幅比                 | 阻力强弱           |
| KLOW  | `(Less($open,$close)-$low)/$open`                  | 下影线相对开盘                 | 探底回升/承接力       |
| KLOW2 | `(Less($open,$close)-$low)/($high-$low+1e-12)`     | 下影线占振幅比                 | 支撑强弱           |
| KSFT  | `(2*$close-$high-$low)/$open`                      | 收盘相对高低中点偏移（未归一）         | 尾盘拉抬/打压侦测      |
| KSFT2 | `(2*$close-$high-$low)/($high-$low+1e-12)`         | 上式对振幅归一                 | 形态强弱横向对比       |

**示例**

* 若 `KUP2` 很大（长上影），第二天倾向震荡或回落 ⇒ 可与 `VSTD`（量波动）联合作为“冲高回落过滤器”。
* `KSFT2>0.6` 且 `KMID2` 也高，常见于尾盘资金主动拉升 ⇒ 次日看多但需结合 `VSUMD`（量能偏置）确认。

## 原始价/量的滞后相对位（按 `d` 展开）

| 因子族                           | 命名规则                       | 表达式模板                                         | 物理意义                 | 典型使用场景            |
| ----------------------------- | -------------------------- | --------------------------------------------- | -------------------- | ----------------- |
| OPENd/HIGHd/LOWd/CLOSEd/VWAPd | `FIELDd`（d∈windows；0 表示当期） | `Ref($field,d)/$close`（d=0 时 `$field/$close`） | 过去/当期价位相对“今日收盘”的定标位置 | 跨标的可比、相对强弱/价位结构对齐 |
| VOLUMEd                       | `VOLUMEd`                  | `Ref($volume,d)/($volume+1e-12)`（d=0 时约为 1）   | 过去量能相对“今日量”的定标位置     | 放量/缩量对比、量能异常检测    |

**示例**

* `CLOSE5<1`、`CLOSE20<1` 同时成立 ⇒ 5/20 日相对弱势；若 `VSUMD5>0`（量偏正）说明或在弱势末端有承接。
* `VWAP10≈1.02`、`VWAP60≈1.10` ⇒ 近 10/60 日机构成交价相对高位，偏强；可与 `RSQR20` 联合判断趋势“干净”程度。

## 滚动算子族（每个算子按 `d∈{5,10,20,30,60}` 等窗口自动展开）

> 说明：下表中的 `ret = $close/Ref($close,1) - 1`；
> `pos_ret = Greater(ret, 0) * ret`，`neg_ret = Less(ret, 0) * (-ret)`；
> `dV = $volume - Ref($volume,1)`，`pos_dV = Greater(dV,0)*dV`，`neg_dV = Less(dV,0)*(-dV)`。

| 因子族   | 命名       | 表达式模板（以 d 为窗口）                                                 | 物理意义                | 典型使用场景        |
| ----- | -------- | -------------------------------------------------------------- | ------------------- | ------------- |
| ROC   | `ROCd`   | `Ref($close,d)/$close`                                         | d 日动量（反向写法；<1 表示上涨） | 动量/反转基元、持有期映射 |
| MA    | `MAd`    | `Mean($close,d)/$close`                                        | 均线相对位（>1 高于均线）      | 多周期均线结构、趋势判定  |
| STD   | `STDd`   | `Std($close,d)/$close`                                         | 相对波动率               | 风险控制、仓位/止损阈值  |
| BETA  | `BETAd`  | `Slope($close,d)/$close`                                       | 线性趋势斜率（相对位）         | 趋势强度刻画、动量打分   |
| RSQR  | `RSQRd`  | `Rsquare($close,d)`                                            | 趋势“干净”程度            | 噪声过滤、动量质量     |
| RESI  | `RESId`  | `Resi($close,d)/$close`                                        | 去趋势残差幅度             | 均值回归信号、波动聚集   |
| MAX   | `MAXd`   | `Max($high,d)/$close`                                          | 区间最高位相对位            | 突破监测、止盈/压顶    |
| MIN   | `MINd`   | `Min($low,d)/$close`                                           | 区间最低位相对位            | 支撑/回撤监测、止损    |
| QTLU  | `QTLUd`  | `Quantile($close,d,0.8)/$close`                                | 上 20% 分位相对位         | 通道上沿、强势筛选     |
| QTLD  | `QTLDd`  | `Quantile($close,d,0.2)/$close`                                | 下 20% 分位相对位         | 通道下沿、防噪声      |
| RANK  | `RANKd`  | `Rank($close,d)`                                               | d 日时序秩位（0–1）        | 归一/稳健排序、非参数打分 |
| RSV   | `RSVd`   | `($close-Min($low,d))/(Max($high,d)-Min($low,d)+1e-12)`        | 区间位置（KDJ 的 RSV 思想）  | 突破/回撤信号、通道择时  |
| IMAX  | `IMAXd`  | `IdxMax($high,d)/d`                                            | 最高价出现的相对时点（0→近；1→远） | 波段节奏/先后次序     |
| IMIN  | `IMINd`  | `IdxMin($low,d)/d`                                             | 最低价出现的相对时点          | 波段节奏/先后次序     |
| IMXD  | `IMXDd`  | `(IdxMax($high,d)-IdxMin($low,d))/d`                           | 高低点先后关系             | 结构顺序判别        |
| CORR  | `CORRd`  | `Corr($close, Log($volume+1), d)`                              | 价位与量能相关性            | 放量上涨/缩量上涨区分   |
| CORD  | `CORDd`  | `Corr($close/Ref($close,1), Log($volume/Ref($volume,1)+1), d)` | 收益与量变化率相关           | 价量共振/背离       |
| CNTP  | `CNTPd`  | `Mean($close>Ref($close,1), d)`                                | 上涨天数占比（频次动量）        | 弱趋势更稳的动量刻画    |
| CNTN  | `CNTNd`  | `Mean($close<Ref($close,1), d)`                                | 下跌天数占比              | 回撤/风险提示       |
| CNTD  | `CNTDd`  | `CNTPd - CNTNd`                                                | 上下次数偏置              | 情绪强弱          |
| SUMP  | `SUMPd`  | `Sum(pos_ret, d)/Sum(Abs(ret), d)`                             | 上涨幅度占比（幅度动量）        | 单边走强甄别        |
| SUMN  | `SUMNd`  | `Sum(neg_ret, d)/Sum(Abs(ret), d)`                             | 下跌幅度占比              | 衰弱/风险警示       |
| SUMD  | `SUMDd`  | `SUMPd - SUMNd`                                                | 幅度偏置（净强度）           | 趋势强弱评分        |
| VMA   | `VMAd`   | `Mean($volume, d)/($volume+1e-12)`                             | 成交量均值相对当期量          | 活跃度、异常量识别     |
| VSTD  | `VSTDd`  | `Std($volume, d)/($volume+1e-12)`                              | 成交量波动率（相对）          | 量的不确定性        |
| WVMA  | `WVMAd`  | `Std(Abs(ret)*$volume, d) / Mean(Abs(ret)*$volume, d)`         | “波动×量”的波动比          | 剧烈波动筛查、风险预警   |
| VSUMP | `VSUMPd` | `Sum(pos_dV, d) / Sum(Abs(dV), d)`                             | 量的上行增量占比            | 健康放量、筹码上移     |
| VSUMN | `VSUMNd` | `Sum(neg_dV, d) / Sum(Abs(dV), d)`                             | 量的下行增量占比            | 恐慌放量、筹码下移     |
| VSUMD | `VSUMDd` | `VSUMPd - VSUMNd`                                              | 量能偏置净值              | 价量共振/背离判别     |

### 物理含义与使用场景（按族补充一些“怎么搭配更好用”的思路）

* **趋势族（ROC / MA / BETA / RSQR）**

  * **含义**：同是“趋势”，但关注点不同：幅度（ROC）、相对位（MA）、方向与强度（BETA）、趋势纯净度（RSQR）。
  * **组合建议**：`BETA20↑ & RSQR20↑` ⇒ 趋势“又直又顺”；再看 `VSUMD20>0`（量偏正）强化置信度。
  * **应用**：中短线动量选股、持有期映射（d 与持有天数同频）。

* **波动/均值回归（STD / RESI）**

  * **含义**：`STD` 是总波动，`RESI` 是去趋势后的剩余波动（更纯的“噪声/回归潜力”）。
  * **组合建议**：`RESI10↑ & RSQR20↓` ⇒ 噪声占比高，回归信号更稳健。
  * **应用**：区间震荡策略、卖出期权（高波动）择时、仓位/止损调节。

* **区间位置/通道（MAX / MIN / QTLU / QTLD / RSV / RANK）**

  * **含义**：把当前价放在近 d 天的“通道/分位”上观察位置与秩。
  * **组合建议**：`RSV20>0.8 & VSUMP20>0.5` ⇒ 放量逼近上沿的突破更可信；`RSV20<0.2 & VSUMN20>0.5` ⇒ 破位风险高。
  * **应用**：通道突破、箱体回撤、分位化过滤（减少噪声）。

* **节奏/结构（IMAX / IMIN / IMXD）**

  * **含义**：极值发生的次序/间隔，刻画“先高后低/先低后高”的波段节奏。
  * **组合建议**：`IMXD20>0`（先创高后创低） + `CNTD20↓` ⇒ 多头衰减；`IMXD20<0`（先低后高） + `CNTP20↑` ⇒ 反转成型。
  * **应用**：波段切换识别、拉回买点/冲高减仓点提示。

* **价量关系（CORR / CORD）**

  * **含义**：价位×量能、收益×量变化率的相关程度。
  * **组合建议**：`CORD20>0.5 & BETA20>0` ⇒ 价量共振的上升趋势；若 `CORD20<0` 则留意“无量空涨”。
  * **应用**：过滤“脆弱趋势”、确认“放量上涨/缩量下跌”等健康结构。

* **频次与幅度动量（CNTP / CNTN / CNTD / SUMP / SUMN / SUMD）**

  * **含义**：分别从“上涨/下跌的次数”和“上涨/下跌的幅度”描述动量。
  * **组合建议**：`CNTP20↑ & SUMP20↑` ⇒ 频次与幅度同向更可靠；若 `CNTP↑ 但 SUMP≈0` ⇒ 多为小阳堆积，警惕一根大阴回吐。
  * **应用**：不同市场状态下切换“频次派/幅度派”的动量。

* **量能活跃与偏置（VMA / VSTD / WVMA / VSUMP / VSUMN / VSUMD）**

  * **含义**：量的均值/波动、量的上/下行增量偏置、波动×量的“能量指标”。
  * **组合建议**：`WVMA20↑` 伴随 `VSUMP20↑` ⇒ 高波动+正向放量，若 `RSV20>0.8` 则关注放量突破；`WVMA20↑ & VSUMN20↑` ⇒ 恐慌放量，优先回避。
  * **应用**：突破确认、衰竭/恐慌检测、风控触发。

## 完整列清单（建议本地导出，确保与你的配置一致）

```python
import qlib
from qlib.constant import REG_CN
from qlib.contrib.data.loader import Alpha158DL
import pandas as pd

qlib.init(provider_uri="data/qlib_data/cn_data", region=REG_CN)
exprs, names = Alpha158DL.get_feature_config()  # 默认模板 + 默认窗口展开
pd.DataFrame({"feature_name": names, "expression": exprs}).to_csv("alpha158_features.csv", index=False)
```

> 备注：Alpha158 真实列数取决于 `conf`（选择的价/量字段、窗口列表、包含/排除的算子）。如切到分钟频，`d` 的含义也随之改变，记得重设窗口并做特征稳定性检查（IC/ICIR、VIF、Permutation Importance）。

### 快速落地建议

1. **先小而准**：从“趋势（BETA/RSQR）+ 动量（ROC/MA）+ 量偏置（VSUMD）+ 风险（STD/RESI）”起步，窗口用 10/20/60 做三线结构。
2. **再做过滤**：突破要配量（`VSUMP↑`）、回撤看承接（`VSUMN↑`）、脆弱趋势要剔除（`CORD<0`、`RSQR↓`）。
3. **最后调仓/风控**：用 `STD/RESI/WVMA` 触发减仓或止盈止损阈值，避免在高噪声期“硬扛”。

## 补充指标

### 1. **振荡器类（Oscillators） - 补充超买/超卖信号，Alpha158的RSV/RANK已部分覆盖，但缺少平滑版本**
- **RSI (Relative Strength Index)**: 相对强弱指数，测量速度和变化。
  - 表达式模板: `100 - 100 / (1 + Mean(Greater(ret,0)*ret, d) / (Mean(Less(ret,0)*(-ret), d) + 1e-12))`
  - 物理含义: 0-100区间，>70超买，<30超卖。
  - 为什么添加: Alpha158有SUMP/SUMN（幅度偏置），但RSI使用平滑均值，更鲁棒于噪声；可与RSQR结合过滤“干净趋势中的超买”。
  - 典型场景: 均值回归策略、反转择时。
- **Stochastic Oscillator (KDJ的完整版)**: 随机指标。
  - 表达式模板: `%K = RSVd; %D = Mean(RSVd, 3); %J = 3*%K - 2*%D`（基于你的RSVd）。
  - 物理含义: %K敏感，%D/%J平滑；交叉信号提示反转。
  - 为什么添加: Alpha158有RSVd，但缺少%D/%J的平滑和交叉逻辑；增强对短期波动的捕捉。
  - 典型场景: 与KUP2/KLOW2结合判断影线后的反转。
- **CCI (Commodity Channel Index)**: 商品通道指数。
  - 表达式模板: `($close - Mean($close, d)) / (0.015 * Mean(Abs($close - Mean($close, d)), d))`
  - 物理含义: >100强势，<-100弱势。
  - 为什么添加: 结合均值和偏差，补充RESId的残差概念，但更标准化；适合高波动过滤。
  - 典型场景: 与WVMA搭配，识别“高能量”突破。

### 2. **趋势确认类（Trend Confirmation） - Alpha158有BETA/RSQR，但缺少方向强度和转折点**
- **MACD (Moving Average Convergence Divergence)**: 移动平均收敛散度。
  - 表达式模板: `MACD = EMA($close,12) - EMA($close,26); Signal = EMA(MACD,9); Hist = MACD - Signal`（注: Qlib需自定义EMA，如迭代Mean with alpha）。
  - 物理含义: 正/负表示趋势，Hist柱状图示背离。
  - 为什么添加: Alpha158有MA/BETA（简单均线/斜率），但MACD用指数平滑，更敏感于转速变化；补充CORD的价量背离。
  - 典型场景: 与VSUMD结合确认“量价共振的MACD金叉”。
- **ADX (Average Directional Index)**: 平均方向运动指数。
  - 表达式模板: `DX = Abs((Max($high,d)-Ref($close,1)) - (Ref($close,1)-Min($low,d))) / (Max($high,d)-Min($low,d)+1e-12) * 100; ADX = Mean(DX, d)`
  - 物理含义: >25强趋势，<20无趋势。
  - 为什么添加: 量化趋势强度，补充RSQR的“干净度”；Alpha158缺少纯方向指标。
  - 典型场景: `ADX20>25 & BETA20>0` ⇒ 强趋势选股。
- **Parabolic SAR (Stop and Reverse)**: 抛物线转向。
  - 表达式模板: 需迭代计算（初始SAR=前低/高，AF=0.02递增）；Qlib中可自定义。
  - 物理含义: 趋势跟踪止损点。
  - 为什么添加: 动态止损，补充MAXd/MINd的静态通道；适合风控。
  - 典型场景: 与STDd结合调整仓位。

### 3. **波动/范围类（Volatility/Range） - Alpha158有STD/RESI，但可加真实范围和带宽**
- **ATR (Average True Range)**: 平均真实范围。
  - 表达式模板: `TR = Max($high-$low, Abs($high-Ref($close,1)), Abs($low-Ref($close,1))); ATR = Mean(TR, d)`
  - 物理含义: 真实波动幅度。
  - 为什么添加: 比KLEN/STD更全面（考虑跳空），补充VSTD的量波动；标准化止损。
  - 典型场景: `ATR14 / $close`作为相对波动，与RESI20结合预测回归潜力。
- **Bollinger Bands Bandwidth**: 布林带宽度。
  - 表达式模板: `(Upper - Lower) / Middle; Upper = MAd + 2*STDd; Lower = MAd - 2*STDd; Middle = MAd`
  - 物理含义: 波动收缩/扩张。
  - 为什么添加: 直接用你的MA/STD构建，补充QTLU/QTLD的通道；识别“挤压”后突破。
  - 典型场景: 宽度低+VSUMP↑ ⇒ 放量扩张机会。

### 4. **量价高级类（Volume-Price Advanced） - Alpha158有CORR/VSUMD，但可加累积指标**
- **OBV (On-Balance Volume)**: 能量潮。
  - 表达式模板: `CumSum(If($close>Ref($close,1), $volume, If($close<Ref($close,1), -$volume, 0)), d)`
  - 物理含义: 累积量能方向。
  - 为什么添加: 补充VSUMDd的短期偏置，提供长期累积视角；检测背离。
  - 典型场景: OBV新高但价格未新高 ⇒ 潜在上涨。
- **Chaikin Money Flow (CMF)**: 蔡金资金流。
  - 表达式模板: `Sum((($close-$low)-($high-$close))/($high-$low+1e-12)*$volume, d) / Sum($volume, d)`
  - 物理含义: 资金流入/流出强度。
  - 为什么添加: 结合K线位置（类似KSFT2）和量，补充CORD的价量相关；更注重日内资金。
  - 典型场景: 与KMID2结合判断“真阳/假阳”的资金支持。