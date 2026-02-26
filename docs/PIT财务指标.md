## PIT 财务指标

### 1 `pit_` 是什么？

`pit_` 前缀表示 **Point-in-Time（按披露时点对齐）** 的财务指标特征：
- 数据源：PIT 财务指标（字段命名沿用 Baostock 的常见缩写风格）
- 输出路径：`features/<inst_fname>/pit_<field>_q.day.bin`
- `_q` 后缀：本数据集对该类财务指标特征的统一命名约定（无论来自季度表/快报/预告，文件名都统一追加 `_q`）

### 2 日频对齐口径（数据集口径）

PIT raw 行的关键列是：
- `date`：公告/披露日（Baostock `pubDate` 或同义字段）
- `period`：财务期末（Baostock `statDate`）
- `value`：指标值

写入 `.day.bin` 时的核心规则：
- 若 `date` 缺失：用 `period + 45 天` 推断公告日（仅用于补齐缺失，不做额外兜底逻辑）
- `period` 会被转换为季度整数 `YYYYQQ`（例：2020-12-31 → 202004）
- 对每个交易日做 LOCF（向前填充）：取“在该日及之前已披露的最新指标值”
- 同一天内若出现多个 `period`：以 **更大的 period（更新的财务期）** 为准

因此，`pit_*.day.bin` 的含义可以理解为：
> “截至该交易日（含），市场当时可获得的、最新财务期的该指标值”

### 3 字段与 bin 文件名对照表

说明
- 下表的 `bin 文件名` 省略了后缀 `.day.bin`，完整文件名形如：`pit_roeavg_q.day.bin`
- `Baostock 字段` 与 `接口` 为本数据集采用的字段命名与分组方式

#### 3.1 盈利能力（Baostock：`query_profit_data`）

| bin 文件名 | Baostock 字段 | 说明（按字段名直译/常用缩写） |
|---|---|---|
| pit_roeavg_q | roeAvg | ROE 平均值 |
| pit_npmargin_q | npMargin | 净利率 |
| pit_gpmargin_q | gpMargin | 毛利率 |
| pit_netprofit_q | netProfit | 净利润 |
| pit_epsttm_q | epsTTM | 每股收益（TTM） |

#### 3.2 运营效率（Baostock：`query_operation_data`）

| bin 文件名 | Baostock 字段 | 说明（按字段名直译/常用缩写） |
|---|---|---|
| pit_nrturnratio_q | NRTurnRatio | 应收账款周转率（NR Turn Ratio） |
| pit_nrturndays_q | NRTurnDays | 应收账款周转天数（NR Turn Days） |
| pit_invturnratio_q | INVTurnRatio | 存货周转率（INV Turn Ratio） |
| pit_invturndays_q | INVTurnDays | 存货周转天数（INV Turn Days） |
| pit_caturnratio_q | CATurnRatio | 流动资产周转率（CA Turn Ratio） |
| pit_assetturnratio_q | AssetTurnRatio | 总资产周转率（Asset Turn Ratio） |

#### 3.3 成长能力（Baostock：`query_growth_data`）

| bin 文件名 | Baostock 字段 | 说明（按字段名直译/常用缩写） |
|---|---|---|
| pit_yoyequity_q | YOYEquity | 净资产同比（YoY Equity） |
| pit_yoyasset_q | YOYAsset | 总资产同比（YoY Asset） |
| pit_yoyni_q | YOYNI | 净利润同比（YoY NI） |
| pit_yoyeps_q | YOYEPSBasic | EPS 同比（YoY EPS Basic） |
| pit_yoypni_q | YOYPNI | 归母净利润同比（YoY PNI） |

#### 3.4 偿债能力/资产负债结构（Baostock：`query_balance_data`）

| bin 文件名 | Baostock 字段 | 说明（按字段名直译/常用缩写） |
|---|---|---|
| pit_currentratio_q | currentRatio | 流动比率 |
| pit_quickratio_q | quickRatio | 速动比率 |
| pit_cashratio_q | cashRatio | 现金比率 |
| pit_liabilitytoasset_q | liabilityToAsset | 资产负债率（负债/资产） |
| pit_assettequity_q | assetToEquity | 权益乘数（资产/权益） |

#### 3.5 现金流/结构指标（Baostock：`query_cash_flow_data`）

| bin 文件名 | Baostock 字段 | 说明（按字段名直译/常用缩写） |
|---|---|---|
| pit_catoasset_q | CAToAsset | 流动资产/总资产 |
| pit_ncatoasset_q | NCAToAsset | 非流动资产/总资产 |
| pit_tangassettoasset_q | tangibleAssetToAsset | 有形资产/总资产 |
| pit_ebittointerest_q | ebitToInterest | EBIT/利息支出 |
| pit_cfotoor_q | CFOToOR | 经营现金流/营业收入 |
| pit_cfotonp_q | CFOToNP | 经营现金流/净利润 |
| pit_cfotogr_q | CFOToGr | 经营现金流/（口径见 Baostock 字段定义：ToGr） |

#### 3.6 杜邦分析（Baostock：`query_dupont_data`）

| bin 文件名 | Baostock 字段 | 说明（按字段名直译/常用缩写） |
|---|---|---|
| pit_dup_roe_q | dupontROE | 杜邦 ROE |
| pit_dup_margin_q | dupontNitogr | 净利润率（Ni to Gr） |
| pit_dup_assetturn_q | dupontAssetTurn | 资产周转率 |
| pit_dup_leverage_q | dupontAssetStoEquity | 权益乘数（Assets to Equity） |
| pit_dup_taxburden_q | dupontTaxBurden | 税负（Tax Burden） |
| pit_dup_intburden_q | dupontIntburden | 利息负担（Int Burden） |
| pit_dup_ebitmargin_q | dupontEbittogr | EBIT 利润率（EBIT to Gr） |

#### 3.7 业绩快报（Baostock：`query_performance_express_report`）

| bin 文件名 | Baostock 字段 | 说明（按字段名直译/常用缩写） |
|---|---|---|
| pit_ex_roewa_q | performanceExpressROEWa | 快报 ROE（加权） |
| pit_ex_eps_q | performanceExpressEPSDiluted | 快报 EPS（稀释） |
| pit_ex_epschg_q | performanceExpressEPSChgPct | 快报 EPS 变动幅度 |
| pit_ex_gryoy_q | performanceExpressGRYOY | 快报 营业收入同比（GR YoY） |
| pit_ex_opyoy_q | performanceExpressOPYOY | 快报 营业利润同比（OP YoY） |
| pit_ex_totalasset_q | performanceExpressTotalAsset | 快报 总资产 |
| pit_ex_netasset_q | performanceExpressNetAsset | 快报 净资产 |

#### 3.8 业绩预告（Baostock：`query_forecast_report`）

| bin 文件名 | Baostock 字段 | 说明（按字段名直译/常用缩写） |
|---|---|---|
| pit_fc_rangeup_q | profitForcastChgPctUp | 预告变动幅度上限（ChgPct Up） |
| pit_fc_rangedown_q | profitForcastChgPctDwn | 预告变动幅度下限（ChgPct Down） |
| pit_fc_rangemid_q | forecastMid | 上下限均值（(Up+Down)/2） |
