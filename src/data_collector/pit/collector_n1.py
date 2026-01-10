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

本版本在原有基础上增加了：
    - “季度级断点续传”能力：以 CSV 中的最大 period（财报季度）为进度，只对未覆盖季度发起请求；
    - 对快报 / 预告采用“按日期增量”的方式，避免重复写入；
    - 支持长时间运行中途中断后，重新执行同一命令自动续跑，同时保证 CSV 幂等；
    - 对单次季频请求增加超时/重试/重登录，减少“卡死”风险。
"""

import re
import sys
import socket
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import baostock as bs
import fire
import pandas as pd
from loguru import logger

# 所有 socket 调用最多等待 30 秒，避免网络层无限挂起
socket.setdefaulttimeout(30)

# tqdm 做“每次接口请求”的进度条；不存在时自动降级
try:
    from tqdm.auto import tqdm  # type: ignore
except ImportError:  # pragma: no cover
    tqdm = None

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR.parent.parent))

from data_collector.base import BaseCollector, BaseNormalize, BaseRun  # noqa: E402
from data_collector.utils import get_calendar_list  # noqa: E402

FieldSpec = Dict[str, Any]

# ----------------------------------------------------------------------
# 全局调试/诊断控制
# ----------------------------------------------------------------------

# 是否保存 baostock 接口返回的原始数据到 tmp 目录（文本格式）
SAVE_RAW_API_RESPONSE: bool = False
RAW_API_TMP_DIR: Path = BASE_DIR / "tmp"

# 是否启用“按请求粒度”的 tqdm 进度条
# 若觉得输出太乱，可改为 False，只保留外层“股票进度条”
ENABLE_REQUEST_PROGRESS: bool = True

# 按“单次请求”粒度的进度条
_REQUEST_PBAR = None  # type: ignore[var-annotated]


def _ensure_tmp_dir() -> Path:
    RAW_API_TMP_DIR.mkdir(parents=True, exist_ok=True)
    return RAW_API_TMP_DIR


def _save_raw_response(
    fetch_name: str,
    code: str,
    fields: List[str],
    rows: List[List[Any]],
    suffix: str = "",
) -> None:
    """
    将一次接口调用的原始 rows 简单写成文本文件，方便诊断“数据不全”问题。
    每一行是逗号分隔，第一行为字段名。
    """
    if not SAVE_RAW_API_RESPONSE:
        return
    if not rows or not fields:
        return

    out_dir = _ensure_tmp_dir()
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    safe_code = code.replace(".", "_")
    suffix_part = f"_{suffix}" if suffix else ""
    filename = out_dir / f"{fetch_name}_{safe_code}{suffix_part}_{ts}.txt"

    try:
        with filename.open("w", encoding="utf-8") as f:
            f.write(",".join(fields) + "\n")
            for row in rows:
                f.write(",".join("" if v is None else str(v) for v in row) + "\n")
    except Exception as exc:
        # 写文件失败不影响主流程，只打日志
        logger.error(f"failed to save raw response to {filename}: {exc}")


def _update_request_progress(desc: str) -> None:
    """
    进度条：按“单次 baostock 请求”更新，而不是按股票。
    desc 会显示当前在跑哪个接口 / 代码 / 区间。
    """
    global _REQUEST_PBAR
    if not ENABLE_REQUEST_PROGRESS or tqdm is None:
        return

    try:
        if _REQUEST_PBAR is None:
            _REQUEST_PBAR = tqdm(
                total=None,
                desc="PIT requests",
                dynamic_ncols=True,
            )
        _REQUEST_PBAR.set_postfix_str(desc)
        _REQUEST_PBAR.update(1)
    except Exception:
        # tqdm 自身报错也不影响主逻辑
        pass


# ----------------------------------------------------------------------
# 指标定义
# ----------------------------------------------------------------------

# 盈利能力（seasonProfit）
PROFIT_FIELD_SPECS: List[FieldSpec] = [
    {"source": "roeAvg", "field": "roeavg", "desc": "Average ROE."},
    {"source": "npMargin", "field": "npmargin", "desc": "Net profit margin."},
    {"source": "gpMargin", "field": "gpmargin", "desc": "Gross profit margin."},
    {"source": "netProfit", "field": "netprofit", "desc": "Net profit (million CNY)."},
    {"source": "epsTTM", "field": "epsttm", "desc": "TTM EPS."},
]

# 运营效率（seasonOperation）
OPERATION_FIELD_SPECS: List[FieldSpec] = [
    {"source": "NRTurnRatio", "field": "nrturnratio", "desc": "Notes & AR turnover ratio."},
    {"source": "NRTurnDays", "field": "nrturndays", "desc": "Notes & AR turnover days."},
    {"source": "INVTurnRatio", "field": "invturnratio", "desc": "Inventory turnover ratio."},
    {"source": "INVTurnDays", "field": "invturndays", "desc": "Inventory turnover days."},
    {"source": "CATurnRatio", "field": "caturnratio", "desc": "Current asset turnover ratio."},
    {"source": "AssetTurnRatio", "field": "assetturnratio", "desc": "Total asset turnover ratio."},
]

# 成长能力（seasonGrowth）
GROWTH_FIELD_SPECS: List[FieldSpec] = [
    {"source": "YOYEquity", "field": "yoyequity", "desc": "YoY equity growth."},
    {"source": "YOYAsset", "field": "yoyasset", "desc": "YoY asset growth."},
    {"source": "YOYNI", "field": "yoyni", "desc": "YoY net income growth."},
    {"source": "YOYEPSBasic", "field": "yoyeps", "desc": "YoY basic EPS growth."},
    {"source": "YOYPNI", "field": "yoypni", "desc": "YoY net income excl. non-recurring."},
]

# 资产负债（seasonBalance）
BALANCE_FIELD_SPECS: List[FieldSpec] = [
    {"source": "currentRatio", "field": "currentratio", "desc": "Current ratio."},
    {"source": "quickRatio", "field": "quickratio", "desc": "Quick ratio."},
    {"source": "cashRatio", "field": "cashratio", "desc": "Cash ratio."},
    {"source": "liabilityToAsset", "field": "liabilitytoasset", "desc": "Debt-to-asset ratio."},
    {"source": "assetToEquity", "field": "assettequity", "desc": "Asset-to-equity multiplier."},
]

# 现金流量能力 / 资本结构比率（seasonCashFlow）
CASH_FLOW_FIELD_SPECS: List[FieldSpec] = [
    {"source": "CAToAsset", "field": "catoasset", "desc": "Current assets / Total assets."},
    {"source": "NCAToAsset", "field": "ncatoasset", "desc": "Non-current assets / Total assets."},
    {"source": "tangibleAssetToAsset", "field": "tangassettoasset", "desc": "Tangible assets / Total assets."},
    {"source": "ebitToInterest", "field": "ebittointerest", "desc": "EBIT / Interest expense."},
    {"source": "CFOToOR", "field": "cfotoor", "desc": "Operating CF / Operating revenue."},
    {"source": "CFOToNP", "field": "cfotonp", "desc": "Operating CF / Net profit."},
    {"source": "CFOToGr", "field": "cfotogr", "desc": "Operating CF / Gross revenue."},
]

# 杜邦分解（seasonDupont）
DUPONT_FIELD_SPECS: List[FieldSpec] = [
    {"source": "dupontROE", "field": "dup_roe", "desc": "ROE from DuPont."},
    {"source": "dupontNitogr", "field": "dup_margin", "desc": "Net profit margin."},
    {"source": "dupontAssetTurn", "field": "dup_assetturn", "desc": "Asset turnover."},
    {"source": "dupontAssetStoEquity", "field": "dup_leverage", "desc": "Equity multiplier."},
    {"source": "dupontTaxBurden", "field": "dup_taxburden", "desc": "Tax burden factor."},
    {"source": "dupontIntburden", "field": "dup_intburden", "desc": "Interest burden factor."},
    {"source": "dupontEbittogr", "field": "dup_ebitmargin", "desc": "EBIT margin."},
]

# 业绩快报（seasonExpress）
EXPRESS_FIELD_SPECS: List[FieldSpec] = [
    {"source": "performanceExpressROEWa", "field": "ex_roewa", "desc": "Express ROE (weighted)."},
    {"source": "performanceExpressEPSDiluted", "field": "ex_eps", "desc": "Express EPS (diluted)."},
    {"source": "performanceExpressEPSChgPct", "field": "ex_epschg", "desc": "EPS growth rate (YoY)."},
    {"source": "performanceExpressGRYOY", "field": "ex_gryoy", "desc": "Total revenue YoY."},
    {"source": "performanceExpressOPYOY", "field": "ex_opyoy", "desc": "Operating profit YoY."},
    {"source": "performanceExpressTotalAsset", "field": "ex_totalasset", "desc": "Total assets (express)."},
    {"source": "performanceExpressNetAsset", "field": "ex_netasset", "desc": "Net assets (express)."},
]

# 业绩预告（seasonForecast）
FORECAST_FIELD_SPECS: List[FieldSpec] = [
    {"source": "profitForcastChgPctUp", "field": "fc_rangeup", "desc": "Forecast YoY growth upper bound."},
    {"source": "profitForcastChgPctDwn", "field": "fc_rangedown", "desc": "Forecast YoY growth lower bound."},
    {"source": "forecastMid", "field": "fc_rangemid", "desc": "Midpoint of YoY growth guidance."},
]

# 汇总所有指标定义，方便统一导出 / 查询
ALL_FIELD_SPECS: List[FieldSpec] = (
    PROFIT_FIELD_SPECS
    + OPERATION_FIELD_SPECS
    + GROWTH_FIELD_SPECS
    + BALANCE_FIELD_SPECS
    + CASH_FLOW_FIELD_SPECS
    + DUPONT_FIELD_SPECS
    + EXPRESS_FIELD_SPECS
    + FORECAST_FIELD_SPECS
)
ALL_FIELD_NAMES: List[str] = [spec["field"] for spec in ALL_FIELD_SPECS]

# 直接拼接 $$ 前缀，方便在 Notebook / 脚本中导入 D.features 用到的字段
INDICATOR_FIELD_NAMES = [f"P($${name}_q)" for name in ALL_FIELD_NAMES]


# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------

def _convert_numeric_preserve_non_numeric(
    series: pd.Series,
    numeric_transform: Optional[Callable[[pd.Series], pd.Series]] = None,
) -> pd.Series:
    """Convert numeric values while keeping non-numeric entries unchanged."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric_transform is not None:
        numeric = numeric_transform(numeric)
    return numeric.where(~numeric.isna(), series)


def _stack_indicator_fields(
    df: pd.DataFrame,
    field_specs: List[FieldSpec],
    context: str = "",
) -> pd.DataFrame:
    """将宽表的指标列展开为 (date, period, field, value) 形式。"""

    if df is None or df.empty:
        return pd.DataFrame()

    available_cols = set(df.columns)
    expected_sources = [spec["source"] for spec in field_specs]
    missing = [col for col in expected_sources if col not in available_cols]

    if missing:
        ctx = f" [{context}]" if context else ""
        logger.warning(
            f"missing expected columns{ctx}: {missing}; "
            f"available={sorted(available_cols)}"
        )

    frames: List[pd.DataFrame] = []
    for spec in field_specs:
        column = spec["source"]
        if column not in available_cols:
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
    """
    统一处理“发布日期”和“对应财报期间”的列：
    - date：用于 qlib 中的 “date”，通常对应公告发布日期；
    - period：用于 qlib 中的 “period”，此处先保持为 YYYY-MM-DD 字符串，
              后续在 Normalize 阶段再转成 YYYYQ / YYYY。
    """
    date_candidates = date_candidates or [
        "pubDate",
        "performanceExpPubDate",
        "profitForcastExpPubDate",
        "date",
    ]
    period_candidates = period_candidates or [
        "statDate",
        "performanceExpStatDate",
        "profitForcastExpStatDate",
        "period",
    ]

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

    def _format_ymd(ts: Any) -> Optional[str]:
        if ts is None or pd.isna(ts):
            return None
        return pd.Timestamp(ts).strftime("%Y-%m-%d")

    df = df.copy()
    # Avoid relying on pandas' `.dt.strftime` stubs (Pylance can mis-type `.dt`).
    df["date"] = date_values.map(_format_ymd)
    df["period"] = period_values.map(_format_ymd)
    return df


# ---------------------- 断点续传辅助：季度 & 日期 ----------------------


def _date_to_quarter_period(dt: pd.Timestamp) -> int:
    """将日期转换为整数 YYYYQ（Q ∈ {1,2,3,4}），用于“季度级进度”比较。"""
    if pd.isna(dt):
        raise ValueError("NaT is not allowed in _date_to_quarter_period")
    quarter = (dt.month - 1) // 3 + 1
    return dt.year * 100 + quarter


def _period_to_year_quarter(period: int) -> Tuple[int, int]:
    """_date_to_quarter_period 的反函数：例如 20241 -> (2024, 1)。"""
    year = period // 100
    quarter = period % 100
    if quarter not in (1, 2, 3, 4):
        raise ValueError(f"invalid quarter in period: {period}")
    return year, quarter


def _next_quarter_period(period: int) -> int:
    """
    给定 YYYYQ，返回下一个季度（自动跨年）：
    20241 -> 20242, 20242 -> 20243, 20243 -> 20244, 20244 -> 20251
    """
    year = period // 100
    quarter = period % 100
    if quarter not in (1, 2, 3, 4):
        raise ValueError(f"invalid quarter in period: {period}")
    if quarter < 4:
        return year * 100 + (quarter + 1)
    return (year + 1) * 100 + 1


def _shift_quarter_period(period: int, delta_quarters: int) -> int:
    """
    将 YYYYQQ 形式的季度 period 平移 delta_quarters 个季度（delta 可为负）。
    例如：202601 + (-1) => 202504；202504 + (-2) => 202502
    """
    if delta_quarters == 0:
        return period
    year, quarter = _period_to_year_quarter(period)
    idx = year * 4 + (quarter - 1)
    idx += delta_quarters
    new_year = idx // 4
    new_quarter = idx % 4 + 1
    return new_year * 100 + new_quarter


def _get_existing_progress(
    save_dir: Path,
    normalized_symbol: str,
) -> Tuple[Optional[pd.Timestamp], Optional[int]]:
    """
    从已有 CSV 中推断这只股票的采集“进度”：

    - max_date:   已有数据中最大的 date（公告发布日期），用于快报/预告增量拉取；
    - max_period: 已有数据中最大的财报期间（按 period 列推断季度），用于季度类接口增量拉取。

    若文件不存在或格式异常，返回 (None, None)。
    """
    csv_path = save_dir / f"{normalized_symbol}.csv"
    if not csv_path.exists():
        return None, None

    try:
        df = pd.read_csv(csv_path, usecols=["date", "period"])
    except Exception as exc:
        logger.warning(f"failed to read {csv_path} for resume info: {exc}")
        return None, None

    if df.empty:
        return None, None

    max_date: Optional[pd.Timestamp] = None
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce")
        if not dates.isna().all():
            max_date = dates.max()

    max_period: Optional[int] = None
    if "period" in df.columns:
        periods = pd.to_datetime(df["period"], errors="coerce")
        if not periods.isna().all():
            max_period_ts = periods.max()
            if pd.notna(max_period_ts):
                max_period = _date_to_quarter_period(max_period_ts)

    return max_date, max_period


def _safe_fetch(
    fetch_fn: Callable[..., Any],
    *,
    code: str,
    year: int,
    quarter: int,
    max_retry: int = 3,
):
    """
    对单次季度查询做：
    - error_code 非 0 时自动重试 + 重登录；
    - 捕获网络异常（包括超时），避免整个任务直接卡死。
    """
    last_resp = None
    for attempt in range(1, max_retry + 1):
        try:
            resp = fetch_fn(code=code, year=year, quarter=quarter)
        except Exception as exc:
            logger.warning(
                f"{fetch_fn.__name__}({code}, {year}Q{quarter}) exception on attempt {attempt}: {exc}"
            )
            # 简单粗暴：重登后再试
            try:
                bs.logout()
            except Exception:
                pass
            lg = bs.login()
            logger.info(
                f"re-login after exception, error_code={lg.error_code}, msg={lg.error_msg}"
            )
            last_resp = None
            continue

        last_resp = resp
        if resp.error_code == "0":
            return resp

        logger.warning(
            f"{fetch_fn.__name__}({code}, {year}Q{quarter}) "
            f"error on attempt {attempt}: {resp.error_code}, {resp.error_msg}"
        )
        try:
            bs.logout()
        except Exception:
            pass
        lg = bs.login()
        logger.info(
            f"re-login after error_code, error_code={lg.error_code}, msg={lg.error_msg}"
        )

    # 多次重试失败，返回最后一个 resp，让上层决定跳过这个季度
    return last_resp


# ----------------------------------------------------------------------
# 按季度拉取 baostock 数据（支持“只从某个季度之后开始”的增量模式）
# ----------------------------------------------------------------------


def _query_quarterly_dataframe(
    fetch_fn: Callable[..., Any],
    code: str,
    start_date: str,
    end_date: str,
    min_period: Optional[int] = None,
) -> pd.DataFrame:
    """
    按季度拉取 baostock 数据，并根据发布日期过滤在 [start_date, end_date] 内的记录。

    这里是“单次请求粒度”的核心：
    - 默认（min_period 为 None）：
        以 start_date 所在季度向前回看 2 个季度作为起点，
        以 end_date 所在季度作为终点（不再向未来扫到 Q4），
        再用 pubDate ∈ [start_date, end_date] 做过滤；
    - 增量模式（设置 min_period）：
        从给定的财报季度 YYYYQ 开始向后枚举，不再扫描历史所有季度，
        结合“max_period” 使用，可实现真正的“季度级断点续传”。
    """
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)
    end_period = _date_to_quarter_period(end_dt)

    if min_period is None:
        # 全量（但针对给定日期区间做精确枚举）：
        # 从 start_date 所在季度向前回看 2 个季度，避免“向前漏报/迟报”；
        # 不再向未来扫到 Q4，避免大量无效请求。
        start_period = _shift_quarter_period(_date_to_quarter_period(start_dt), -2)
        start_year, start_quarter = _period_to_year_quarter(start_period)
    else:
        # 增量：只从尚未覆盖的季度开始
        start_year, start_quarter = _period_to_year_quarter(min_period)

    records: List[List[str]] = []
    fields: Optional[List[str]] = None

    year = start_year
    quarter = start_quarter

    while (year * 100 + quarter) <= end_period:
        _update_request_progress(f"{fetch_fn.__name__} {code} {year}Q{quarter}")

        resp = _safe_fetch(fetch_fn, code=code, year=year, quarter=quarter)
        if resp is None:
            logger.warning(f"{fetch_fn.__name__}({code}, {year}Q{quarter}) returns None")
            # 下一季度
            if quarter == 4:
                year += 1
                quarter = 1
            else:
                quarter += 1
            continue

        if resp.error_code != "0":
            logger.warning(
                f"{fetch_fn.__name__}({code}, {year}Q{quarter}) error: {resp.error_msg}"
            )
            # 继续下一个季度
            if quarter == 4:
                year += 1
                quarter = 1
            else:
                quarter += 1
            continue

        if fields is None:
            fields = resp.fields

        quarter_raw_rows: List[List[str]] = []

        pubdate_idx: Optional[int] = None
        if "pubDate" in resp.fields:
            pubdate_idx = resp.fields.index("pubDate")

        while resp.next():
            row = resp.get_row_data()
            if not row:
                continue

            # 原始 rows（不做过滤，用于诊断保存）
            quarter_raw_rows.append(row)

            # 发布日过滤
            if pubdate_idx is not None:
                pub_date_raw = row[pubdate_idx]
                if pub_date_raw:
                    pub_ts = pd.to_datetime(pub_date_raw, errors="coerce")
                    if pd.isna(pub_ts) or not (start_dt <= pub_ts <= end_dt):
                        continue

            records.append(row)

        if quarter_raw_rows and fields:
            _save_raw_response(
                fetch_fn.__name__,
                code,
                fields,
                quarter_raw_rows,
                suffix=f"{year}Q{quarter}",
            )

        # 下一个季度
        if quarter == 4:
            year += 1
            quarter = 1
        else:
            quarter += 1

    if not records or fields is None:
        return pd.DataFrame()

    df = pd.DataFrame(records, columns=fields)
    return _finalize_temporal_columns(df)


# ----------------------------------------------------------------------
# Collector / Normalize / Runner
# ----------------------------------------------------------------------


class PitCollectorN1(BaseCollector):
    """
    增强版 PIT Collector：
    - 支持季度类接口（profit / operation / growth / balance / cashflow / dupont）的“季度级断点续传”，
      通过 CSV 中的最大 period 推断已覆盖的最后一个财报季度；
    - 支持业绩快报 / 预告的“按公告日期增量拉取”，通过 CSV 中的最大 date 推断已覆盖的最新公告日。
    """

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
        # 保存原始 CSV 目录路径，用于“断点续传”时读取已有进度
        self.save_dir = Path(save_dir)

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
        """
        从本地 qlib instruments 文件中加载股票列表，并按 symbol_regex 进行过滤。
        注意：断点续传逻辑不在此处“按文件是否存在”过滤，
        而是交给 get_data 结合 CSV 内容精细控制（按季度/公告日增量）。
        """
        logger.info("load cn stock symbols from local instrument file......")
        instrument_file = (
            BASE_DIR.parent.parent.parent
            .joinpath("data", "qlib_data", "cn_data", "instruments", "all.txt")
        )
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

    # ---------------------- 各类指标采集（支持 min_period） ----------------------

    def _collect_profitability(
        self,
        code: str,
        start_date: str,
        end_date: str,
        min_period: Optional[int] = None,
    ) -> pd.DataFrame:
        df = _query_quarterly_dataframe(
            bs.query_profit_data,
            code,
            start_date,
            end_date,
            min_period=min_period,
        )
        return _stack_indicator_fields(df, PROFIT_FIELD_SPECS, context=f"{code}-profit")

    def _collect_operation(
        self,
        code: str,
        start_date: str,
        end_date: str,
        min_period: Optional[int] = None,
    ) -> pd.DataFrame:
        df = _query_quarterly_dataframe(
            bs.query_operation_data,
            code,
            start_date,
            end_date,
            min_period=min_period,
        )
        return _stack_indicator_fields(df, OPERATION_FIELD_SPECS, context=f"{code}-operation")

    def _collect_growth(
        self,
        code: str,
        start_date: str,
        end_date: str,
        min_period: Optional[int] = None,
    ) -> pd.DataFrame:
        df = _query_quarterly_dataframe(
            bs.query_growth_data,
            code,
            start_date,
            end_date,
            min_period=min_period,
        )
        return _stack_indicator_fields(df, GROWTH_FIELD_SPECS, context=f"{code}-growth")

    def _collect_balance(
        self,
        code: str,
        start_date: str,
        end_date: str,
        min_period: Optional[int] = None,
    ) -> pd.DataFrame:
        df = _query_quarterly_dataframe(
            bs.query_balance_data,
            code,
            start_date,
            end_date,
            min_period=min_period,
        )
        return _stack_indicator_fields(df, BALANCE_FIELD_SPECS, context=f"{code}-balance")

    def _collect_cash_flow(
        self,
        code: str,
        start_date: str,
        end_date: str,
        min_period: Optional[int] = None,
    ) -> pd.DataFrame:
        df = _query_quarterly_dataframe(
            bs.query_cash_flow_data,
            code,
            start_date,
            end_date,
            min_period=min_period,
        )
        return _stack_indicator_fields(df, CASH_FLOW_FIELD_SPECS, context=f"{code}-cashflow")

    def _collect_dupont(
        self,
        code: str,
        start_date: str,
        end_date: str,
        min_period: Optional[int] = None,
    ) -> pd.DataFrame:
        df = _query_quarterly_dataframe(
            bs.query_dupont_data,
            code,
            start_date,
            end_date,
            min_period=min_period,
        )
        return _stack_indicator_fields(df, DUPONT_FIELD_SPECS, context=f"{code}-dupont")

    # ---------------------- 业绩快报 & 预告（按日期增量） ----------------------

    def _collect_express(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        _update_request_progress(
            f"query_performance_express_report {code} {start_date}~{end_date}"
        )

        resp = bs.query_performance_express_report(code=code, start_date=start_date, end_date=end_date)
        if resp is None:
            logger.warning(
                f"query_performance_express_report({code}, {start_date}, {end_date}) returned None; skip."
            )
            return pd.DataFrame()

        if resp.error_code != "0":
            logger.warning(
                f"query_performance_express_report({code}, {start_date}, {end_date}) error: {resp.error_msg}"
            )
            return pd.DataFrame()

        rows: List[List[str]] = []
        while resp.error_code == "0" and resp.next():
            rows.append(resp.get_row_data())

        if not rows:
            logger.info(
                f"no performance express report data for {code} between {start_date} and {end_date}"
            )
            return pd.DataFrame()

        _save_raw_response(
            "query_performance_express_report",
            code,
            resp.fields,
            rows,
            suffix=f"{start_date}_{end_date}",
        )

        df = pd.DataFrame(rows, columns=resp.fields)
        df = _finalize_temporal_columns(
            df,
            date_candidates=["performanceExpPubDate", "pubDate", "date"],
            period_candidates=["performanceExpStatDate", "statDate", "period"],
        )
        return _stack_indicator_fields(df, EXPRESS_FIELD_SPECS, context=f"{code}-express")

    def _collect_forecast(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        业绩预告（seasonForecast）增量采集。

        这里对 baostock.query_forecast_report 做了异常防护：
        - 捕获 JSONDecodeError（服务端返回非法 JSON）；
        - 捕获其它异常（网络问题等）；
        - resp.error_code != "0" 时直接返回空表。

        这样可以避免单只股票的坏数据导致整个采集任务崩溃。
        """
        _update_request_progress(f"query_forecast_report {code} {start_date}~{end_date}")

        try:
            resp = bs.query_forecast_report(code=code, start_date=start_date, end_date=end_date)
        except json.JSONDecodeError as exc:
            logger.warning(
                f"query_forecast_report({code}, {start_date}, {end_date}) "
                f"JSONDecodeError: {exc}; skip forecast for this code & range."
            )
            return pd.DataFrame()
        except Exception as exc:
            logger.warning(
                f"query_forecast_report({code}, {start_date}, {end_date}) "
                f"raised exception: {exc}; skip forecast for this code & range."
            )
            return pd.DataFrame()

        if resp is None:
            logger.warning(
                f"query_forecast_report({code}, {start_date}, {end_date}) "
                "returned None; skip."
            )
            return pd.DataFrame()

        if resp.error_code != "0":
            logger.warning(
                f"query_forecast_report({code}, {start_date}, {end_date}) "
                f"error_code={resp.error_code}, msg={resp.error_msg}; skip."
            )
            return pd.DataFrame()

        rows: List[List[str]] = []
        while resp.error_code == "0" and resp.next():
            rows.append(resp.get_row_data())

        if not rows:
            logger.info(
                f"no forecast report data for {code} between {start_date} and {end_date}"
            )
            return pd.DataFrame()

        _save_raw_response(
            "query_forecast_report",
            code,
            resp.fields,
            rows,
            suffix=f"{start_date}_{end_date}",
        )

        df = pd.DataFrame(rows, columns=resp.fields)
        df = _finalize_temporal_columns(
            df,
            date_candidates=["profitForcastExpPubDate", "pubDate", "date"],
            period_candidates=["profitForcastExpStatDate", "statDate", "period"],
        )

        # 计算中位值 forecastMid = (上限 + 下限) / 2
        if {"profitForcastChgPctUp", "profitForcastChgPctDwn"}.issubset(df.columns):
            up = pd.to_numeric(df["profitForcastChgPctUp"], errors="coerce")
            down = pd.to_numeric(df["profitForcastChgPctDwn"], errors="coerce")
            df["forecastMid"] = ((up + down) / 2).where(~(up.isna() | down.isna()))

        return _stack_indicator_fields(df, FORECAST_FIELD_SPECS, context=f"{code}-forecast")


    # ---------------------- 核心：结合 CSV 进度实现“断点续传 + 增量更新” ----------------------

    def get_data(
        self,
        symbol: str,
        interval: str,
        start_datetime: pd.Timestamp,
        end_datetime: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        核心采集逻辑：

        - 对季度类接口：
            根据 CSV 中的最大 period 推断已覆盖到哪一个财报季度，
            若已覆盖到 end 所在季度，则本轮跳过；
            否则仅从“尚未覆盖的下一季度”开始向后拉取。

        - 对快报 / 预告：
            根据 CSV 中的最大 date 推断最新公告日期，
            仅从 (max_date + 1 天) 起按日期增量拉取，避免重复写入。

        这样可以实现：
            - 初次运行全量拉取；
            - 长任务中途中断后重跑自动续传；
            - 之后每天/每周定期运行，仅抓增量数据。
        """
        if interval != self.INTERVAL_QUARTERLY:
            raise ValueError(f"cannot support {interval}")

        # qlib 的 symbol 形如 "000001.sz"，需要转成 baostock 的 "sz.000001"
        symbol_code, exchange = symbol.split(".")
        exchange = "sh" if exchange == "ss" else "sz"
        code = f"{exchange}.{symbol_code}"
        normalized_symbol = f"{exchange}{symbol_code}"  # 用作 CSV 文件名前缀

        start_dt = pd.Timestamp(start_datetime)
        end_dt = pd.Timestamp(end_datetime)
        start_date = start_dt.strftime("%Y-%m-%d")
        end_date = end_dt.strftime("%Y-%m-%d")

        # ---- 1) 从已有 CSV 推断“进度”：最大公告日 & 最大财报季度 ----
        existing_max_date, existing_max_period = _get_existing_progress(
            self.save_dir, normalized_symbol
        )
        end_period = _date_to_quarter_period(end_dt)

        # ---- 2) 决定季度类接口是否需要请求，以及从哪一个季度开始 ----
        if existing_max_period is not None and existing_max_period >= end_period:
            # 已经覆盖到当前 end 所在的季度，本轮无需再对季度类接口发请求
            skip_quarterly = True
            min_period = None
            logger.info(
                f"skip quarterly part for {symbol}: "
                f"existing_max_period={existing_max_period}, end_period={end_period}"
            )
        else:
            skip_quarterly = False
            # 只对“尚未覆盖的下一季度及以后”发请求
            if existing_max_period is None:
                min_period = None
            else:
                min_period = _next_quarter_period(existing_max_period)

        # ---- 3) 决定快报/预告的“增量起点日期” ----
        if existing_max_date is not None:
            express_start_dt = max(start_dt, existing_max_date + pd.Timedelta(days=1))
        else:
            express_start_dt = start_dt

        if express_start_dt > end_dt:
            skip_express = True
            express_start_date = None
            logger.info(
                f"skip express/forecast for {symbol}: "
                f"existing_max_date={existing_max_date.date() if existing_max_date is not None else None}, "
                f"requested_end={end_dt.date()}"
            )
        else:
            skip_express = False
            express_start_date = express_start_dt.strftime("%Y-%m-%d")

        frames: List[pd.DataFrame] = []

        # ---- 4) 季度类接口（profit / operation / growth / balance / cashflow / dupont） ----
        if not skip_quarterly:
            logger.info(
                f"collect quarterly data for {symbol} ({code}) from {start_date} to {end_date}, "
                f"min_period={min_period}, "
                f"existing_max_period={existing_max_period}, end_period={end_period}"
            )

            for collector in [
                self._collect_profitability,
                self._collect_operation,
                self._collect_growth,
                self._collect_balance,
                self._collect_cash_flow,
                self._collect_dupont,
            ]:
                df = collector(code, start_date, end_date, min_period=min_period)
                if df is not None and not df.empty:
                    frames.append(df)

        # ---- 5) 业绩快报 / 预告（按公告日期增量拉取） ----
        if not skip_express and express_start_date is not None:
            logger.info(
                f"collect express/forecast for {symbol} ({code}) from "
                f"{express_start_date} to {end_date}, "
                f"existing_max_date={existing_max_date.date() if existing_max_date is not None else None}"
            )

            df_express = self._collect_express(code, express_start_date, end_date)
            if df_express is not None and not df_express.empty:
                frames.append(df_express)

            df_forecast = self._collect_forecast(code, express_start_date, end_date)
            if df_forecast is not None and not df_forecast.empty:
                frames.append(df_forecast)

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
        """
        将采集阶段生成的 (date, period, field, value) 宽表，转换为 qlib PIT 所需格式：
        - date：若缺失，则在财报期末日基础上加 45/90 天进行推断；
        - period：按季度转换为 YYYYQ 整数，或按年度转换为 YYYY。
        """
        if df is None or df.empty:
            return pd.DataFrame()

        df = df.copy()
        period_ts = pd.to_datetime(df["period"], errors="coerce")
        offset_days = 45 if self.interval == PitCollectorN1.INTERVAL_QUARTERLY else 90
        inferred_dates = (period_ts + pd.to_timedelta(offset_days, unit="D")).dt.strftime("%Y-%m-%d")
        df["date"] = df["date"].fillna(inferred_dates)

        df["period"] = period_ts.apply(
            lambda x: x.year
            if self.interval == PitCollectorN1.INTERVAL_ANNUAL
            else x.year * 100 + (x.month - 1) // 3 + 1
            if pd.notna(x)
            else None
        )
        return df

    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        """优先从本地 qlib calendar 读取，fallback 到网络接口。"""

        local_calendar = (
            BASE_DIR.parent.parent.parent
            / "data"
            / "qlib_data"
            / "cn_data"
            / "calendars"
            / "day.txt"
        )
        if local_calendar.exists():
            dates = pd.read_csv(local_calendar, header=None, names=["date"], dtype=str)["date"]
            dates = pd.to_datetime(dates, errors="coerce").dropna().tolist()
            if dates:
                return dates
            logger.warning(
                "local calendar file exists but empty or invalid, fallback to remote calendar"
            )
        else:
            logger.info(
                f"local calendar file not found: {local_calendar}, fallback to remote calendar"
            )
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
        # 收尾进度条
        if _REQUEST_PBAR is not None:
            try:
                _REQUEST_PBAR.close()
            except Exception:
                pass
