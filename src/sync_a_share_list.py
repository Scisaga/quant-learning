# -*- coding: utf-8 -*-
"""
每日同步 沪深A股 + 创业板 + 科创板 标的清单（含上市日期/市值）
数据源：东方财富 push2 接口（公开可用）
依赖：pip install requests pandas python-dateutil
可选依赖（socks 代理）: pip install 'requests[socks]'

用法:
  python sync_a_share_list.py
  python sync_a_share_list.py --filter                       # 基础交易性过滤示例 上市满 ~250 交易日 流通市值 >= 30 亿元
  python sync_a_share_list.py --proxy http://127.0.0.1:7890  # 通过代理请求


字段说明: 

| 字段             | 含义          | 类型  | 单位/格式                     | 备注                                              |
| -------------- | ----------- | --- | ------------------------- | ----------------------------------------------- |
| `symbol`       | 标准代码（与长桥一致） | 字符串 | `600036.SH` / `000001.SZ` | 由 `code + "." + exchange` 拼接而成                  |
| `code`         | 证券代码（6 位）   | 字符串 | 如 `600036`                | 不含交易所后缀                                         |
| `name`         | 证券简称        | 字符串 | —                         | 来自东财返回 `f14`                                    |
| `exchange`     | 交易所         | 枚举  | `SH` / `SZ`               | 由东财 `f13`（1=上交所，0=深交所）映射                        |
| `board`        | 板块          | 枚举  | 主板 / 创业板 / 科创板            | 依据代码段推断：`688***`=科创，`300***`=创业，其余=主板           |
| `list_date`    | 上市日期        | 日期  | `YYYY-MM-DD`              | 来自东财 `f26`（原始为 `yyyymmdd`，脚本已转日期） ([CSDN博客][1]) |
| `total_mv`     | 总市值         | 数值  | 元（RMB）                    | 东财 `f20`，注意单位是“元”不是“亿元” ([CSDN博客][1])           |
| `float_mv`     | 流通市值        | 数值  | 元（RMB）                    | 东财 `f21`，同上为“元” ([CSDN博客][1])                   |
| `total_mv_bil` | 总市值（亿元）     | 数值  | 亿元（RMB）                   | 由 `total_mv / 1e8` 计算（便于筛选）                     |
| `float_mv_bil` | 流通市值（亿元）    | 数值  | 亿元（RMB）                   | 由 `float_mv / 1e8` 计算                           |
| `f2`           | 最新价         | 数值  | 元                         | 东财字段定义：最新价（现价） ([CSDN博客][1])                    |
| `f3`           | 涨跌幅         | 数值  | %                         | 直接返回百分数（如 `-1.23` 表示 **-1.23%**） ([CSDN博客][1])  |
| `f6`           | 成交额         | 数值  | 元                         | 当日成交金额（RMB） ([CSDN博客][1])                       |
| `f8`           | 换手率         | 数值  | %                         | 直接为百分数（如 `3.15`= **3.15%**） ([CSDN博客][1])       |

参考: 
[1]: https://blog.csdn.net/weixin_42430074/article/details/125450480?utm_source=chatgpt.com "东方财富f指标对应原创"

"""
import os
import sys
import time
import re
import argparse
import datetime as dt
from dateutil.tz import gettz
from typing import Optional, Dict

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd

# ---------------------- 常量与全局配置 ----------------------
EM_URL = "https://push2.eastmoney.com/api/qt/clist/get"
"""东方财富 clist 列表接口基础 URL。"""

# fs 组合：上证主板 / 科创 / 深证主板(含原中小板) / 创业板
FS_ALL_A = "m:1 t:2,m:1 t:23,m:0 t:6,m:0 t:13,m:0 t:80"
"""东财 fs 参数，覆盖沪深 A 股（含创业/科创）。"""

# 需要的字段：代码/名称/交易所/上市日期/总市值/流通市值/基础行情
EM_FIELDS = ",".join([
    "f12","f13","f14",      # 代码 市场 名称
    "f26",                  # 上市日期 yyyymmdd
    "f20","f21",            # 总市值 流通市值（单位=元）
    "f2","f3","f6","f8"     # 最新价 涨跌幅 成交额 换手率
])
"""clist 返回字段列表，对应脚本里各列的含义见注释。"""

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://quote.eastmoney.com/"
}
"""请求头，适配东财页面来源校验。"""

TIMEOUT = (5, 15)  # (connect, read)
"""HTTP 超时设置：连接 5s，读取 15s。"""


# ---------------------- 网络会话（含代理/重试） ----------------------
def build_session(proxy: Optional[str] = None) -> requests.Session:
    """
    构建带重试能力的 requests 会话；支持代理。
    - 若 `proxy` 传入，则强制使用该代理（http/https 同置）。
    - 若未传入，则沿用系统环境变量中的 HTTP(S)_PROXY/ALL_PROXY（requests 默认行为）。
    """
    sess = requests.Session()
    sess.headers.update(DEFAULT_HEADERS)

    # 自动重试：应对偶发 429/5xx
    retries = Retry(
        total=3,
        backoff_factor=0.3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retries)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)

    if proxy:
        # 支持 http/https/socks5/socks5h; 若用 socks，请安装 requests[socks]
        sess.proxies.update({"http": proxy, "https": proxy})

    return sess


# ---------------------- 数据抓取 ----------------------
def _page_fetch(session: requests.Session, fs: str, pn: int, pz: int = 200) -> list:
    """
    抓取单页列表数据。
    说明：东财对 clist 有“强分页”限制，即使 pz 很大也需翻页；当返回 diff 为空时表示结束。
    """
    params: Dict[str, str] = {
        "pn": str(pn),
        "pz": str(pz),     # 单页条数；若被限，会继续翻页
        "po": "1",
        "np": "1",
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": "2",
        "invt": "2",
        "fid": "f3",
        "fs": fs,
        "fields": EM_FIELDS,
        "_": str(int(time.time() * 1000))
    }
    r = session.get(EM_URL, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    j = r.json()
    data = (j.get("data") or {}).get("diff") or []
    return data


def fetch_em_list(fs: str, session: Optional[requests.Session] = None) -> pd.DataFrame:
    """
    根据 fs 参数抓取东财 A 股列表，并生成标准化字段：
      - symbol(600036.SH) / code / name / exchange(SH|SZ) / board(主板|创业板|科创板)
      - list_date(datetime64[ns]) / total_mv / float_mv / *_mv_bil（亿元）
      - f2(现价) / f3(涨跌幅%) / f6(成交额元) / f8(换手率%)
    """
    sess = session or build_session()
    rows = []
    pn = 1
    while True:
        diff = _page_fetch(sess, fs, pn)
        if not diff:
            break
        rows.extend(diff)
        pn += 1
        time.sleep(0.2)  # 轻微节流，降低被风控概率

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # 交易所映射：f13=0 深交所 -> SZ；1 上交所 -> SH
    exch_map = {0: "SZ", 1: "SH"}
    df["exchange"] = df["f13"].map(exch_map)

    # 代码统一 6 位；生成 Longbridge 规范 symbol
    df["code"] = df["f12"].astype(str).str.strip().str.zfill(6)
    df["symbol"] = df["code"] + "." + df["exchange"]

    # 强校验：600036.SH / 000001.SZ
    pat = re.compile(r"^\d{6}\.(SH|SZ)$")
    df = df[df["symbol"].apply(lambda s: bool(pat.match(s)))]

    df["name"] = df["f14"]

    # 上市日期：yyyymmdd -> datetime64[ns]
    df["list_date"] = pd.to_datetime(df["f26"].astype(str), format="%Y%m%d", errors="coerce")

    # 市值：元；并提供亿元换算列
    df["total_mv"] = pd.to_numeric(df["f20"], errors="coerce")
    df["float_mv"] = pd.to_numeric(df["f21"], errors="coerce")
    df["total_mv_bil"] = df["total_mv"] / 1e8
    df["float_mv_bil"] = df["float_mv"] / 1e8

    # 板块推断：SH 以 688*** 识别科创；SZ 以 300*** 识别创业
    def infer_board(row):
        code, ex = row["code"], row["exchange"]
        if ex == "SH":
            return "科创板" if code.startswith("688") else "主板"
        else:
            return "创业板" if code.startswith("300") else "主板"

    df["board"] = df.apply(infer_board, axis=1)

    keep_cols = [
        "symbol", "code", "name", "exchange", "board",
        "list_date", "total_mv", "float_mv", "total_mv_bil", "float_mv_bil",
        "f2", "f3", "f6", "f8"
    ]
    return df[keep_cols].sort_values(["exchange", "code"]).reset_index(drop=True)


def fetch_all_a_share(session: Optional[requests.Session] = None) -> pd.DataFrame:
    """
    抓取“沪深主板 + 创业板 + 科创板”全量列表。
    """
    return fetch_em_list(FS_ALL_A, session=session)


# ---------------------- 基础过滤（示例） ----------------------
def basic_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    基础交易性过滤示例：
    - 上市满 ~250 交易日（≈360 天）
    - 流通市值 >= 30 亿元
    """
    today = pd.Timestamp.today().normalize()
    list_date = pd.to_datetime(df["list_date"], errors="coerce")
    days = (today - list_date).dt.days.fillna(-9999).astype(int)
    return df[(days >= 360) & (df["float_mv_bil"] >= 30)]


# ---------------------- CLI 入口 ----------------------
def main():
    """
    命令行入口：
    - 默认抓取全量列表并导出 CSV
    - `--filter` 开启基础交易性过滤
    - `--proxy` 强制指定代理（例如 http://127.0.0.1:7890 或 socks5h://user:pass@host:1080）
      若不传，则沿用系统环境变量中的代理设置。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", action="store_true", help="应用基础交易性过滤示例")
    parser.add_argument("--proxy", type=str, default=None,
                        help="HTTP/HTTPS 代理，如 http://127.0.0.1:7890 或 socks5h://127.0.0.1:1080")
    args = parser.parse_args()

    # 构建带代理/重试的会话
    session = build_session(proxy=args.proxy)

    os.makedirs("../data", exist_ok=True)
    df = fetch_all_a_share(session=session)

    if df.empty:
        print("未获取到数据，请稍后重试或检查网络/代理设置。")
        sys.exit(2)

    df_out = basic_filter(df) if args.filter else df
    tag = "filtered" if args.filter else "full"

    # 写出 CSV（保留 list_date 为 Timestamp；如需纯日期，可 df_out["list_date"]=df_out["list_date"].dt.date）
    df_out["list_date"]=df_out["list_date"].dt.date

    today_str = dt.datetime.now(gettz("Asia/Shanghai")).strftime("%Y%m%d")
    out_path = f"data/a_share_list_{tag}_{today_str}.csv"
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"生成：{out_path} （共 {len(df_out)} 条，原始 {len(df)} 条）")


if __name__ == "__main__":
    main()
