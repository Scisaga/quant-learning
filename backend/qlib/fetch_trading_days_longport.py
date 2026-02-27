from __future__ import annotations

"""
util/fetch_trading_days_longport.py

用途：
  用最小化、可复现的方式验证 Longport/Longbridge Quote API 是否可达：
  - 读取项目根目录的 `.env`（通过导入 `config.py` 触发 load_dotenv）；
  - 使用 `longport.openapi.QuoteContext.trading_days` 拉取交易日；
  - 直接打印接口返回内容（交易日列表/半日列表/原始对象 repr），便于排查“无法访问”的问题。

为什么要单独脚本：
  job/sync_trading_day.py 会涉及 DB 写入、分段、事务等逻辑；当网络/鉴权异常时不利于快速定位。
  本脚本只做一次接口调用 + 打印结果，方便你：
  - 验证代理是否生效、地址是否正确；
  - 验证 LONGPORT_* 凭证是否被正确加载；
  - 验证指定 market/date range 能否返回预期数据。

示例运行（推荐在项目根目录）：
  0) 全默认（市场 CN，未来 1 个月，含今天）：
     python util/fetch_trading_days_longport.py

  1) 直连（不走代理）：
     python util/fetch_trading_days_longport.py --market CN --days 7

  2) 使用 `.env` 中的代理（例如 HTTPS_PROXY=http://127.0.0.1:7890）：
     python util/fetch_trading_days_longport.py --market HK --start-date 2026-02-02 --end-date 2026-02-08

  3) 打印 raw 响应对象（便于看 SDK 字段）：
     python util/fetch_trading_days_longport.py --market US --days 14 --print-raw

输出说明：
  - 会先打印当前进程可见的代理环境变量（HTTPS_PROXY/HTTP_PROXY 及其小写版本）；
  - 再打印请求参数与返回的交易日/半日列表（ISO 日期字符串）。
"""

import argparse
import importlib.metadata
import os
import sys
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

# 让脚本可直接导入项目模块（config/dao/job 等）
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from longport.openapi import Config as LBConfig, Market, QuoteContext

from config import HTTPS_PROXY, logger  # 触发读取 .env

_MARKET_ENUM_MAP = {
    "HK": Market.HK,
    "CN": Market.CN,
    "US": Market.US,
    "SG": Market.SG,
}


def _mask_proxy_url(proxy: str) -> str:
    """
    将代理 URL 中的密码脱敏，避免在日志/控制台泄露敏感信息。

    例：
      http://user:pass@127.0.0.1:7890 -> http://user:***@127.0.0.1:7890
    """
    if "://" not in proxy:
        return proxy

    parts = urlsplit(proxy)
    if parts.username is None or parts.password is None:
        return proxy

    host = parts.hostname or ""
    if parts.port is not None:
        host = f"{host}:{parts.port}"

    netloc = f"{parts.username}:***@{host}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def _apply_proxy_env_from_dotenv() -> None:
    """
    约定：
      - 项目统一用 `.env` 里的 `HTTPS_PROXY` 作为“代理源”；
      - 同步到常见的 HTTP(S) 环境变量键，便于 SDK/底层 HTTP 库识别。
    """
    if not HTTPS_PROXY:
        return
    for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        os.environ[key] = HTTPS_PROXY


def _apply_proxy_from_cli(proxy: str | None, *, no_proxy: bool) -> None:
    """
    允许在命令行里显式控制代理，便于排查：
    - `--no-proxy`：清空 HTTP/HTTPS 代理环境变量（强制直连）
    - `--proxy http://127.0.0.1:7890`：显式设置代理（覆盖现有环境变量）
    """
    keys = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy")
    if no_proxy:
        for k in keys:
            os.environ.pop(k, None)
        return

    if proxy is None or not str(proxy).strip():
        return

    p = str(proxy).strip()
    for k in keys:
        os.environ[k] = p


def _print_proxy_env() -> None:
    keys = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy")
    items: list[str] = []
    for k in keys:
        v = os.environ.get(k)
        if v:
            items.append(f"{k}={_mask_proxy_url(v)}")
    if items:
        print("proxy_env:", ", ".join(items))
    else:
        print("proxy_env: <empty>（未检测到代理环境变量，直连）")


def _normalize_market(market: str) -> str:
    m = market.strip().upper()
    if m in {"SH", "SZ", "SS"}:
        m = "CN"
    return m


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call Longport Quote API trading_days and print result.")
    parser.add_argument(
        "--market",
        default="CN",
        help="市场代码：CN/HK/US/SG（A 股子市场 SH/SZ/SS 会归一为 CN）",
    )
    parser.add_argument("--proxy", type=str, help="显式指定代理，例如 http://127.0.0.1:7890（会覆盖现有代理环境变量）")
    parser.add_argument("--no-proxy", action="store_true", help="强制直连：清空 HTTP/HTTPS 代理环境变量")
    parser.add_argument("--start-date", type=date.fromisoformat, help="开始日期（YYYY-MM-DD），默认今天")
    parser.add_argument("--end-date", type=date.fromisoformat, help="结束日期（YYYY-MM-DD）；若提供则优先于 --days")
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="同步/查询天数（含今天，含端点）；默认 30（约等于未来 1 个月）。若提供 --end-date 则忽略该参数",
    )
    parser.add_argument("--print-raw", action="store_true", help="额外打印 SDK 原始响应对象 repr")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    _apply_proxy_env_from_dotenv()
    _apply_proxy_from_cli(args.proxy, no_proxy=bool(args.no_proxy))

    print("cwd:", str(Path.cwd()))
    print("python:", sys.version.split()[0])
    try:
        print("longport:", importlib.metadata.version("longport"))
    except Exception:
        print("longport: <unknown>")
    print(
        "longport_env:",
        {
            "LONGPORT_HTTP_URL": os.getenv("LONGPORT_HTTP_URL") or "",
            "LONGPORT_QUOTE_WS_URL": os.getenv("LONGPORT_QUOTE_WS_URL") or "",
            "LONGPORT_TRADE_WS_URL": os.getenv("LONGPORT_TRADE_WS_URL") or "",
        },
    )
    _print_proxy_env()

    market = _normalize_market(str(args.market))
    if market not in _MARKET_ENUM_MAP:
        raise ValueError(f"Unsupported market: {market!r} (expect one of {sorted(_MARKET_ENUM_MAP)})")

    start_date = args.start_date or date.today()
    if args.end_date is not None:
        end_date = args.end_date
    else:
        end_date = start_date + timedelta(days=int(args.days) - 1)
    if start_date > end_date:
        start_date, end_date = end_date, start_date

    # 明确打点：方便你对比日志时间与网络侧抓包/代理日志
    logger.info("fetch_trading_days_longport request market=%s start=%s end=%s", market, start_date, end_date)

    config = LBConfig.from_env()
    ctx = QuoteContext(config)
    try:
        resp = ctx.trading_days(_MARKET_ENUM_MAP[market], start_date, end_date)
    finally:
        try:
            ctx.close()
        except Exception:
            pass

    trading_days = [d.isoformat() for d in (resp.trading_days or [])]
    half_days = [d.isoformat() for d in (resp.half_trading_days or [])]

    print("request:", {"market": market, "start_date": start_date.isoformat(), "end_date": end_date.isoformat()})
    print("trading_days_count:", len(trading_days))
    print("half_trading_days_count:", len(half_days))
    print("trading_days:", trading_days)
    print("half_trading_days:", half_days)
    if args.print_raw:
        print("raw_resp:", repr(resp))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
