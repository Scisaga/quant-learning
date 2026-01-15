## Qlib 实验产物（MLflow artifacts）

这些 `.pkl` 基本都是 Qlib 的工作流 `RecordTemp` 在运行时通过 `recorder.save_objects(...)` 自动存到 MLflow artifacts 里的（底层就是 `mlruns/.../artifacts/...`）。

- `pred.pkl`：`SignalRecord.generate()` 存的预测结果（通常是 `DataFrame(index=(datetime, instrument), columns=['score'])`），见 `record_temp.py` line 162
- `label.pkl`：`SignalRecord.generate()` 同时存的标签（`dataset.prepare(..., col_set='label')`），用于 IC/回测评估，见 `record_temp.py` line 162
- `trained_model`：你手动 `R.save_objects(trained_model=model)` 存的模型对象（pickle 后的 `LGBModel` 等）
- `sig_analysis/`（`SigAnaRecord` 产物，见 `record_temp.py` line 296）
  - `ic.pkl`：逐期 IC 序列（预测分数 vs label 的相关系数）
  - `ric.pkl`：逐期 Rank IC 序列（按排名相关）
- `portfolio_analysis/`（`PortAnaRecord` 产物，见 `record_temp.py` line 359）
  - `report_normal_1day.pkl`：回测“组合收益报告”（包含 return/bench/cost 等列）
  - `positions_normal_1day.pkl`：回测持仓明细（每期持有哪些标的/权重/市值等，结构取决于回测配置）
  - `port_analysis_1day.pkl`：对 `report_normal_1day.pkl` 的风险分析结果（excess return、有无成本等的均值/波动/IR/MDD…）
  - `indicators_normal_1day.pkl`：交易执行层面的逐期指标序列（如 ffr/pa/pos/count/deal_amount/value 等）
  - `indicators_normal_1day_obj.pkl`：更“原始”的指标对象（用于进一步分析/可视化/复现）
  - `indicator_analysis_1day.pkl`：对 `indicators_normal_1day.pkl` 的汇总统计（`indicator_analysis` 的输出）
- `code_status.txt`：非 pkl，Qlib/MLflow 为了复现实验自动记录的代码状态快照

## data/qlib_data 目录结构

`data/qlib_data` 是 Qlib 运行时的数据根目录，常见结构如下：

- `cn_data/`：A 股日频数据
  - `calendars/`：交易日历
    - `day.txt`：日频交易日历（按交易日顺序，每行一个日期）
    - `day_future.txt`：更长时间范围的日历（用于未来日期对齐）
  - `instruments/`：股票池定义（文本）
    - 文件格式：`instrument<TAB>start_date<TAB>end_date`，用于定义成分生效区间
    - 代码样式：`SH600000`、`SZ000001`、`BJ430017` 等
    - `all.txt`：数据中所有可交易标的（包含沪深北，随数据集而变）
    - `csi300.txt`：中证 300 成分股（大盘蓝筹）
    - `csi500.txt`：中证 500 成分股（中盘）
    - `csi800.txt`：中证 800（通常是 csi300 + csi500）
    - `csi1000.txt`：中证 1000 成分股（小盘）
    - `csiall.txt`：中证全指（全市场 A 股，具体成分以数据源为准）
  - `features/`：行情特征（基础行情字段 + PIT 财务特征）
    - 按标的分目录（如 `sh601515/`）
    - 文件以 `*.day.bin` 存储，每个文件对应一个字段、与 `calendars/day.txt` 对齐
    - 基础行情字段：
      - `open.day.bin`：开盘价
      - `high.day.bin`：最高价
      - `low.day.bin`：最低价
      - `close.day.bin`：收盘价
      - `adjclose.day.bin`：复权收盘价（通常由 `close` 与 `factor` 计算得到）
      - `volume.day.bin`：成交量
      - `amount.day.bin`：成交额
      - `vwap.day.bin`：成交量加权均价
      - `change.day.bin`：涨跌幅（通常是日收益率，如 `close/prev_close - 1`）
      - `factor.day.bin`：复权因子（用于价格复权）
    - PIT 财务字段（`pit_*`）：point-in-time 对齐后的财务指标，按公告/生效日对齐并前向填充到日频，避免未来信息泄露
      - 命名规则：`pit_<指标名>_q.day.bin`（季度口径），与 `financial/<instrument>/<指标名>_q` 口径一致
      - 字段说明（示例以 `sh601515/` 为准，具体口径以数据源为准）：
        - `pit_assettequity_q.day.bin`：资产/股东权益比（资产权益比）
        - `pit_assetturnratio_q.day.bin`：总资产周转率
        - `pit_cashratio_q.day.bin`：现金比率
        - `pit_catoasset_q.day.bin`：流动资产占总资产比
        - `pit_caturnratio_q.day.bin`：流动资产周转率
        - `pit_cfotogr_q.day.bin`：经营现金流/营业收入（现金流含量）
        - `pit_cfotonp_q.day.bin`：经营现金流/净利润
        - `pit_cfotoor_q.day.bin`：经营现金流/营业利润
        - `pit_currentratio_q.day.bin`：流动比率
        - `pit_dup_assetturn_q.day.bin`：杜邦分解-资产周转率
        - `pit_dup_ebitmargin_q.day.bin`：杜邦分解-EBIT 利润率
        - `pit_dup_intburden_q.day.bin`：杜邦分解-利息负担
        - `pit_dup_leverage_q.day.bin`：杜邦分解-财务杠杆
        - `pit_dup_margin_q.day.bin`：杜邦分解-净利率
        - `pit_dup_roe_q.day.bin`：杜邦分解-ROE
        - `pit_dup_taxburden_q.day.bin`：杜邦分解-税负
        - `pit_ebittointerest_q.day.bin`：EBIT/利息费用覆盖倍数
        - `pit_epsttm_q.day.bin`：每股收益（TTM）
        - `pit_ex_eps_q.day.bin`：一致预期/扩展口径 EPS
        - `pit_ex_epschg_q.day.bin`：一致预期 EPS 变动
        - `pit_ex_gryoy_q.day.bin`：一致预期营收同比增速
        - `pit_ex_netasset_q.day.bin`：一致预期净资产
        - `pit_ex_opyoy_q.day.bin`：一致预期营业利润同比增速
        - `pit_ex_roewa_q.day.bin`：一致预期 ROE（加权平均）
        - `pit_ex_totalasset_q.day.bin`：一致预期总资产
        - `pit_fc_rangedown_q.day.bin`：一致预期区间-下限
        - `pit_fc_rangemid_q.day.bin`：一致预期区间-中位
        - `pit_fc_rangeup_q.day.bin`：一致预期区间-上限
        - `pit_gpmargin_q.day.bin`：毛利率
        - `pit_invturndays_q.day.bin`：存货周转天数
        - `pit_invturnratio_q.day.bin`：存货周转率
        - `pit_liabilitytoasset_q.day.bin`：资产负债率
        - `pit_ncatoasset_q.day.bin`：非流动资产占总资产比
        - `pit_netprofit_q.day.bin`：净利润
        - `pit_npmargin_q.day.bin`：净利率
        - `pit_nrturndays_q.day.bin`：应收账款周转天数
        - `pit_nrturnratio_q.day.bin`：应收账款周转率
        - `pit_quickratio_q.day.bin`：速动比率
        - `pit_roeavg_q.day.bin`：平均 ROE
        - `pit_tangassettoasset_q.day.bin`：有形资产占总资产比
        - `pit_yoyasset_q.day.bin`：资产同比增速
        - `pit_yoyeps_q.day.bin`：EPS 同比增速
        - `pit_yoyequity_q.day.bin`：权益同比增速
        - `pit_yoyni_q.day.bin`：净利润同比增速
        - `pit_yoypni_q.day.bin`：净利润（口径依数据源）同比增速
    - 使用方式：
      - 直接作为特征字段引用（与普通行情字段相同）
      - 示例：
        ```python
        from qlib.data import D
        fields = ["$open", "$close", "$pit_roeavg_q", "$pit_yoyni_q"]
        df = D.features(
            instruments="csi300",
            fields=fields,
            start_time="2020-01-01",
            end_time="2020-12-31",
            freq="day",
        )
        ```
      - 说明：这里的 `pit_*.day.bin` 已是 PIT 对齐后的日频特征，一般不需要 `pit=True`；只有在直接读取 PIT 原始库时才需要 `pit=True`
    - 说明：字段的精确定义、复权口径、量价单位依赖数据源配置
  - `financial/`：财务因子
    - 按标的分目录（如 `sh600000/`）
    - 每个字段通常以 `*.data` 和 `*.index` 成对存在：
      - `*.index` 记录财报日期（或发布/生效日期）
      - `*.data` 记录对应指标的数值序列
    - 常见财务指标（示例，具体以数据集为准）：
      - 盈利能力：`gpmargin_q`（毛利率）、`npmargin_q`（净利率）、`dup_ebitmargin_q`
      - ROE/杜邦：`roeavg_q`、`dup_roe_q`、`dup_leverage_q`、`dup_margin_q`、`dup_assetturn_q`
      - 偿债/流动性：`currentratio_q`、`quickratio_q`、`cashratio_q`、`liabilitytoasset_q`
      - 运营效率：`assetturnratio_q`、`invturnratio_q`、`nrturnratio_q`、`invturndays_q`、`nrturndays_q`
      - 现金流：`cfotogr_q`、`cfotonp_q`、`cfotoor_q`
      - 成长性：`yoyasset_q`、`yoyeps_q`、`yoyni_q`、`yoypni_q`、`yoyequity_q`
      - 每股/TTM：`epsttm_q`
      - 预期区间：`fc_rangeup_q`、`fc_rangemid_q`、`fc_rangedown_q`
      - 扩展口径：以 `ex_` 前缀开头的指标（具体含义以数据源说明为准）
- `cn_data_1min/`：A 股 1 分钟数据包（本仓库为 `*_latest.zip` 压缩包）
- `us_data/`：美股日频数据包（本仓库为 `*_latest.zip` 压缩包）
