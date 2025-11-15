# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
增强版 PIT 采集器：基于 Baostock 获取多维度的财务指标，并统一为 qlib 可用的字段与格式。

覆盖的指标分组：
    - 盈利能力（seasonProfit）：ROE、净利率、毛利率、净利润、EPS 等。
    - 运营效率（seasonOperation）：应收/票据周转、存货周转、应收账款周转等。
    - 成长能力（seasonGrowth）：股东权益、资产、净利润、EPS 等同比增速。
    - 资产负债（seasonBalance）：流动/速动/现金比率、资产负债率、权益乘数等。
    - 现金流量（seasonCashFlow）：经营/投资/筹资现金流、现金净变化、自由现金流等。
    - 杜邦分解（seasonDupont）：ROE 分解为利润率、周转率、杠杆、税负/利息负担等。
    - 业绩快报（seasonExpress）：加权 ROE、EPS、BPS、营业收入、净利润等快报口径。
    - 业绩预告（seasonForecast）：净利润同比增速的上下限与中位值预测等。
"""

import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import baostock as bs
import fire
import pandas as pd
from loguru import logger

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR.parent.parent))

from data_collector.base import BaseCollector, BaseNormalize, BaseRun  # noqa: E402
from data_collector.utils import get_calendar_list  # noqa: E402

FieldSpec = Dict[str, Any]


def _convert_numeric_preserve_non_numeric(
    series: pd.Series, numeric_transform: Optional[Callable[[pd.Series], pd.Series]] = None
) -> pd.Series:
    """Convert numeric values while keeping non-numeric entries unchanged."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric_transform is not None:
        numeric = numeric_transform(numeric)
    return numeric.where(~numeric.isna(), series)


def _stack_indicator_fields(df: pd.DataFrame, field_specs: List[FieldSpec]) -> pd.DataFrame:
    """Explode wide indicator columns into (date, period, field, value) rows."""
    if df is None or df.empty:
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    for spec in field_specs:
        column = spec["source"]
        if column not in df.columns:
            continue

        series = df[column]
        convert_numeric = spec.get("convert_numeric", True)
        transform = spec.get("numeric_transform")
        if convert_numeric:
            series = _convert_numeric_preserve_non_numeric(series, transform)
        elif transform is not None:
            series = transform(series)

        stacked = pd.DataFrame(
            {
                "date": df["date"],
                "period": df["period"],
                "field": spec["field"],
                "value": series,
            }
        )
        frames.append(stacked)

    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def _finalize_temporal_columns(
    df: pd.DataFrame,
    date_candidates: Optional[List[str]] = None,
    period_candidates: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Ensure df has string-formatted date and period columns."""
    date_candidates = date_candidates or ["pubDate", "performanceExpPubDate", "profitForcastExpPubDate", "date"]
    period_candidates = period_candidates or ["statDate", "performanceExpStatDate", "profitForcastExpStatDate", "period"]

    date_values = None
    for col in date_candidates:
        if col in df.columns:
            date_values = pd.to_datetime(df[col], errors="coerce")
            if date_values.notna().any():
                break
    if date_values is None:
        date_values = pd.Series(pd.NaT, index=df.index)

    period_values = None
    for col in period_candidates:
        if col in df.columns:
            period_values = pd.to_datetime(df[col], errors="coerce")
            if period_values.notna().any():
                break
    if period_values is None:
        period_values = pd.Series(pd.NaT, index=df.index)

    df = df.copy()
    df["date"] = date_values.dt.strftime("%Y-%m-%d")
    df["period"] = period_values.dt.strftime("%Y-%m-%d")
    return df


def _query_quarterly_dataframe(
    fetch_fn: Callable[..., Any], code: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """Fetch paginated quarterly data and keep rows within publish-date range."""
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)
    start_year = start_dt.year - 1
    end_year = end_dt.year + 1
    quarters = [(year, quarter) for year in range(start_year, end_year + 1) for quarter in range(1, 5)]

    records: List[List[str]] = []
    fields: Optional[List[str]] = None

    for year, quarter in quarters:
        resp = fetch_fn(code=code, year=year, quarter=quarter)
        if resp.error_code != "0":
            logger.warning(f"{fetch_fn.__name__}({code}, {year}Q{quarter}) error: {resp.error_msg}")
            continue

        fields = resp.fields
        while resp.next():
            row = resp.get_row_data()
            if not row:
                continue
            if "pubDate" in resp.fields:
                pub_date_raw = row[resp.fields.index("pubDate")]
                if pub_date_raw:
                    pub_ts = pd.to_datetime(pub_date_raw, errors="coerce")
                    if pd.isna(pub_ts) or not (start_dt <= pub_ts <= end_dt):
                        continue
            records.append(row)

    if not records or fields is None:
        return pd.DataFrame()

    df = pd.DataFrame(records, columns=fields)
    return _finalize_temporal_columns(df)


# 盈利能力（seasonProfit）
#   pro.roeAvg：ROE（平均）
#   pro.npMargin：净利率
#   pro.gpMargin：毛利率
#   pro.netProfit：净利润（百万元）
#   pro.epsTTM：每股收益（TTM）
PROFIT_FIELD_SPECS: List[FieldSpec] = [
    {"source": "roeAvg", "field": "pro_roeavg", "desc": "Average ROE."},
    {"source": "npMargin", "field": "pro_npmargin", "desc": "Net profit margin."},
    {"source": "gpMargin", "field": "pro_gpmargin", "desc": "Gross profit margin."},
    {"source": "netProfit", "field": "pro_netprofit", "desc": "Net profit (million CNY)."},
    {"source": "epsTTM", "field": "pro_epsttm", "desc": "TTM EPS."},
]

# 运营效率（seasonOperation）
#   op.nrTurn：应收票据及应收账款周转率
#   op.inventoryTurn：存货周转率
#   op.arTurn：应收账款周转率
OPERATION_FIELD_SPECS: List[FieldSpec] = [
    {"source": "NRTurnRate", "field": "op_nrturn", "desc": "Notes & AR turnover."},
    {"source": "inventoryTurn", "field": "op_inventoryturn", "desc": "Inventory turnover."},
    {"source": "ARTurn", "field": "op_arturn", "desc": "Accounts receivable turnover."},
]

# 成长能力（seasonGrowth）
#   gr.yoyEquity：股东权益同比增速
#   gr.yoyAsset：资产总计同比增速
#   gr.yoyNI：归母净利润同比增速
#   gr.yoyEPS：基本每股收益同比增速
#   gr.yoyPNI：扣非净利润同比增速
GROWTH_FIELD_SPECS: List[FieldSpec] = [
    {"source": "YOYEquity", "field": "gr_yoyequity", "desc": "YoY equity growth."},
    {"source": "YOYAsset", "field": "gr_yoyasset", "desc": "YoY asset growth."},
    {"source": "YOYNI", "field": "gr_yoyni", "desc": "YoY net income growth."},
    {"source": "YOYEPSBasic", "field": "gr_yoyeps", "desc": "YoY basic EPS growth."},
    {"source": "YOYPNI", "field": "gr_yoypni", "desc": "YoY net income excl. non-recurring."},
]

# 资产负债（seasonBalance）
#   bal.currentRatio：流动比率
#   bal.quickRatio：速动比率
#   bal.cashRatio：现金比率
#   bal.liabilityToAsset：资产负债率
#   bal.assetToEquity：权益乘数
BALANCE_FIELD_SPECS: List[FieldSpec] = [
    {"source": "currentRatio", "field": "bal_currentratio", "desc": "Current ratio."},
    {"source": "quickRatio", "field": "bal_quickratio", "desc": "Quick ratio."},
    {"source": "cashRatio", "field": "bal_cashratio", "desc": "Cash ratio."},
    {"source": "liabilityToAsset", "field": "bal_liabilitytoasset", "desc": "Debt-to-asset ratio."},
    {"source": "assetToEquity", "field": "bal_assettoequity", "desc": "Asset-to-equity multiplier."},
]

# 现金流量（seasonCashFlow）
#   cf.cfo：经营活动现金流量净额
#   cf.cfi：投资活动现金流量净额
#   cf.cff：筹资活动现金流量净额
#   cf.netChange：现金及现金等价物净增加额
#   cf.freeCashFlow：自由现金流
CASH_FLOW_FIELD_SPECS: List[FieldSpec] = [
    {"source": "NCFOperateA", "field": "cf_cfo", "desc": "Net cash from operations."},
    {"source": "NCFFrInvestA", "field": "cf_cfi", "desc": "Net cash from investing."},
    {"source": "NCFFrFinanA", "field": "cf_cff", "desc": "Net cash from financing."},
    {"source": "NChangeInCash", "field": "cf_netchange", "desc": "Net change in cash."},
    {"source": "FCF", "field": "cf_freecashflow", "desc": "Free cash flow."},
]

# 杜邦分解（seasonDupont）
#   dup.roe：ROE
#   dup.margin：净利率
#   dup.assetTurn：资产周转率
#   dup.leverage：权益乘数
#   dup.taxBurden / dup.intBurden：税负/利息负担
DUPONT_FIELD_SPECS: List[FieldSpec] = [
    {"source": "dupontROE", "field": "dup_roe", "desc": "ROE from DuPont."},
    {"source": "dupontOperaMargin", "field": "dup_margin", "desc": "Operating profit margin."},
    {"source": "dupontAssetTurn", "field": "dup_assetturn", "desc": "Asset turnover."},
    {"source": "dupontAssetStoEquity", "field": "dup_leverage", "desc": "Equity multiplier."},
    {"source": "dupontTaxBurden", "field": "dup_taxburden", "desc": "Tax burden factor."},
    {"source": "dupontIntburden", "field": "dup_intburden", "desc": "Interest burden factor."},
]

# 业绩快报（seasonExpress）
#   ex.roeWa：加权 ROE（快报）
#   ex.eps：稀释每股收益
#   ex.bps：每股净资产
#   ex.opIncome：营业收入
#   ex.netProfit：净利润
EXPRESS_FIELD_SPECS: List[FieldSpec] = [
    {"source": "performanceExpressROEWa", "field": "ex_roewa", "desc": "Express ROE (weighted)."},
    {"source": "performanceExpressEPSDiluted", "field": "ex_eps", "desc": "Express EPS (diluted)."},
    {"source": "performanceExpressBPS", "field": "ex_bps", "desc": "Express BPS."},
    {"source": "performanceExpressOP", "field": "ex_opincome", "desc": "Express operating income."},
    {"source": "performanceExpressNP", "field": "ex_netprofit", "desc": "Express net profit."},
]

# 业绩预告（seasonForecast）
#   fc_rangeup/fc_rangedown：净利润同比增速上下限（预告）
#   fc_rangemid：上下限中位值（预告）
FORECAST_FIELD_SPECS: List[FieldSpec] = [
    {"source": "profitForcastChgPctUp", "field": "fc_rangeup", "desc": "Forecast YoY growth upper bound."},
    {"source": "profitForcastChgPctDwn", "field": "fc_rangedown", "desc": "Forecast YoY growth lower bound."},
    {"source": "forecastMid", "field": "fc_rangemid", "desc": "Midpoint of YoY growth guidance."},
]


class PitCollectorN1(BaseCollector):
    DEFAULT_START_DATETIME_QUARTERLY = pd.Timestamp("2000-01-01")
    DEFAULT_START_DATETIME_ANNUAL = pd.Timestamp("2000-01-01")
    DEFAULT_END_DATETIME_QUARTERLY = pd.Timestamp(datetime.now() + pd.Timedelta(days=1))
    DEFAULT_END_DATETIME_ANNUAL = pd.Timestamp(datetime.now() + pd.Timedelta(days=1))

    INTERVAL_QUARTERLY = "quarterly"
    INTERVAL_ANNUAL = "annual"

    def __init__(
        self,
        save_dir: Union[str, Path],
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "quarterly",
        max_workers: int = 1,
        max_collector_count: int = 1,
        delay: int = 0,
        check_data_length: bool = False,
        limit_nums: Optional[int] = None,
        symbol_regex: Optional[str] = None,
    ):
        self.symbol_regex = symbol_regex
        super().__init__(
            save_dir=save_dir,
            start=start,
            end=end,
            interval=interval,
            max_workers=max_workers,
            max_collector_count=max_collector_count,
            delay=delay,
            check_data_length=check_data_length,
            limit_nums=limit_nums,
        )

    def get_instrument_list(self) -> List[str]:
        logger.info("load cn stock symbols from local instrument file......")
        instrument_file = BASE_DIR.parent.parent.parent.joinpath("data", "qlib_data", "cn_data", "instruments", "all.txt")
        if not instrument_file.exists():
            raise FileNotFoundError(f"instrument file not found: {instrument_file}")

        def _normalize(code: str) -> Optional[str]:
            exchange = code[:2].upper()
            symbol = code[2:]
            if exchange == "SH":
                return f"{symbol}.ss"
            if exchange == "SZ":
                return f"{symbol}.sz"
            return None

        symbols: List[str] = []
        with instrument_file.open("r", encoding="utf-8") as fp:
            for line in fp:
                if not line.strip():
                    continue
                code = line.split()[0]
                normalized = _normalize(code)
                if normalized:
                    symbols.append(normalized)

        if not symbols:
            raise ValueError(f"no valid instruments parsed from {instrument_file}")

        if self.symbol_regex is not None:
            regex_compile = re.compile(self.symbol_regex)
            symbols = [symbol for symbol in symbols if regex_compile.match(symbol)]
        logger.info(f"get {len(symbols)} symbols.")
        return symbols

    def normalize_symbol(self, symbol: str) -> str:
        symbol, exchange = symbol.split(".")
        exchange = "sh" if exchange == "ss" else "sz"
        return f"{exchange}{symbol}"

    def _collect_profitability(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        df = _query_quarterly_dataframe(bs.query_profit_data, code, start_date, end_date)
        return _stack_indicator_fields(df, PROFIT_FIELD_SPECS)

    def _collect_operation(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        df = _query_quarterly_dataframe(bs.query_operation_data, code, start_date, end_date)
        return _stack_indicator_fields(df, OPERATION_FIELD_SPECS)

    def _collect_growth(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        df = _query_quarterly_dataframe(bs.query_growth_data, code, start_date, end_date)
        return _stack_indicator_fields(df, GROWTH_FIELD_SPECS)

    def _collect_balance(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        df = _query_quarterly_dataframe(bs.query_balance_data, code, start_date, end_date)
        return _stack_indicator_fields(df, BALANCE_FIELD_SPECS)

    def _collect_cash_flow(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        df = _query_quarterly_dataframe(bs.query_cash_flow_data, code, start_date, end_date)
        return _stack_indicator_fields(df, CASH_FLOW_FIELD_SPECS)

    def _collect_dupont(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        df = _query_quarterly_dataframe(bs.query_dupont_data, code, start_date, end_date)
        return _stack_indicator_fields(df, DUPONT_FIELD_SPECS)

    def _collect_express(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        resp = bs.query_performance_express_report(code=code, start_date=start_date, end_date=end_date)
        rows = []
        while resp.error_code == "0" and resp.next():
            rows.append(resp.get_row_data())
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=resp.fields)
        df = _finalize_temporal_columns(
            df,
            date_candidates=["performanceExpPubDate", "pubDate", "date"],
            period_candidates=["performanceExpStatDate", "statDate", "period"],
        )
        return _stack_indicator_fields(df, EXPRESS_FIELD_SPECS)

    def _collect_forecast(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        resp = bs.query_forecast_report(code=code, start_date=start_date, end_date=end_date)
        rows = []
        while resp.error_code == "0" and resp.next():
            rows.append(resp.get_row_data())
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=resp.fields)
        df = _finalize_temporal_columns(
            df,
            date_candidates=["profitForcastExpPubDate", "pubDate", "date"],
            period_candidates=["profitForcastExpStatDate", "statDate", "period"],
        )
        if {"profitForcastChgPctUp", "profitForcastChgPctDwn"}.issubset(df.columns):
            up = pd.to_numeric(df["profitForcastChgPctUp"], errors="coerce")
            down = pd.to_numeric(df["profitForcastChgPctDwn"], errors="coerce")
            df["forecastMid"] = ((up + down) / 2).where(~(up.isna() | down.isna()))
        return _stack_indicator_fields(df, FORECAST_FIELD_SPECS)

    def get_data(
        self,
        symbol: str,
        interval: str,
        start_datetime: pd.Timestamp,
        end_datetime: pd.Timestamp,
    ) -> pd.DataFrame:
        if interval != self.INTERVAL_QUARTERLY:
            raise ValueError(f"cannot support {interval}")

        symbol_code, exchange = symbol.split(".")
        exchange = "sh" if exchange == "ss" else "sz"
        code = f"{exchange}.{symbol_code}"
        start_date = start_datetime.strftime("%Y-%m-%d")
        end_date = end_datetime.strftime("%Y-%m-%d")

        collectors = [
            self._collect_profitability,
            self._collect_operation,
            self._collect_growth,
            self._collect_balance,
            self._collect_cash_flow,
            self._collect_dupont,
            self._collect_express,
            self._collect_forecast,
        ]

        frames = []
        for collector in collectors:
            df = collector(code, start_date, end_date)
            if df is not None and not df.empty:
                frames.append(df)

        if not frames:
            return pd.DataFrame(columns=["date", "period", "field", "value"])

        result = (
            pd.concat(frames, ignore_index=True)
            .dropna(subset=["date", "period", "field"])
            .drop_duplicates(subset=["date", "period", "field", "value"])
        )
        return result


class PitNormalizeN1(BaseNormalize):
    """Convert enriched PIT CSV files into qlib-friendly format."""

    def __init__(self, interval: str = PitCollectorN1.INTERVAL_QUARTERLY, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.interval = interval

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        df = df.copy()
        period_ts = pd.to_datetime(df["period"], errors="coerce")
        offset_days = 45 if self.interval == PitCollectorN1.INTERVAL_QUARTERLY else 90
        inferred_dates = (period_ts + pd.to_timedelta(offset_days, unit="D")).dt.strftime("%Y-%m-%d")
        df["date"] = df["date"].fillna(inferred_dates)

        df["period"] = period_ts.apply(
            lambda x: x.year if self.interval == PitCollectorN1.INTERVAL_ANNUAL else x.year * 100 + (x.month - 1) // 3 + 1
            if pd.notna(x)
            else None
        )
        return df

    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        return get_calendar_list()


class Run(BaseRun):
    def __init__(
        self,
        source_dir: Optional[Union[str, Path]] = None,
        normalize_dir: Optional[Union[str, Path]] = None,
        max_workers: int = 1,
        interval: str = "1d",
    ):
        super().__init__(
            source_dir=source_dir,
            normalize_dir=normalize_dir,
            max_workers=max_workers,
            interval=interval,
        )
        self._cur_module = sys.modules[__name__]

    @property
    def collector_class_name(self) -> str:
        return "PitCollectorN1"

    @property
    def normalize_class_name(self) -> str:
        return "PitNormalizeN1"

    @property
    def default_base_dir(self) -> Union[Path, str]:
        return BASE_DIR


if __name__ == "__main__":
    bs.login()
    try:
        fire.Fire(Run)
    finally:
        bs.logout()
