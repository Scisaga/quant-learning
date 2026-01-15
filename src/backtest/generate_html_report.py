from __future__ import annotations

"""
从 Qlib 的训练记录（Recorder）生成 HTML 回测报告（prediction / backtest / analysis）。

这个脚本的定位是把 `notebooks/qlib_test.ipynb` 中“prediction, backtest & analysis”那一段
抽出来做成可重复运行的 CLI：给定 `experiment_name` + `recorder_id`（或自动取最新）即可生成
一份自包含的 HTML 报告，默认保存到 `reports/`。

核心输入（来自训练脚本产物）
- `pred.pkl`：模型对 (datetime, instrument) 的预测分数（信号）。
- `trained_model`：训练好的模型对象（可选，仅用于展示 model 名字）。

核心输出
- `reports/qlib_report_<exp>_<rid>_<timestamp>.html`

使用示例
- 指定 recorder：
  `python src/backtest/generate_html_report.py --exp-name tutorial_exp --recorder-id <RID>`
- 不指定 recorder：自动选择该 experiment 下最新的 recorder：
  `python src/backtest/generate_html_report.py --exp-name tutorial_exp`

注意事项
- 这个脚本会“按需补齐”回测/分析产物：如果 recorder 里还没有 portfolio/signal 分析的 pkl，
  会调用 Qlib 的 `PortAnaRecord.generate()` / `SigAnaRecord.generate()` 去生成并写回 recorder。
- 不使用 `R.start(resume=True)`：在 MLflow 后端下，resume 时重复写入 param（比如 cmd-sys.argv）
  可能报错（MLflow 不允许覆盖同名 param 的不同值）。因此这里直接 `R.get_recorder(...)` 读取并补齐。
"""

import argparse
import html as html_lib
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# 兼容性处理：部分 Qlib 依赖（尤其是 RL 相关）可能 import `gym`，而新环境里常用 `gymnasium`。
# 这里把 gymnasium 注入到 sys.modules["gym"]，以避免导入时报错/警告。
try:
    import gymnasium as gym  # type: ignore

    sys.modules["gym"] = gym
except Exception:
    pass

try:
    import pandas as pd
except ModuleNotFoundError as e:
    raise SystemExit("Missing dependency: pandas. Install dependencies via `pip install -r requirements.txt`.") from e

try:
    import qlib
except ModuleNotFoundError as e:
    raise SystemExit("Missing dependency: qlib. Install dependencies via `pip install -r requirements.txt`.") from e
from qlib.constant import REG_CN
from qlib.data import D
from qlib.workflow import R
from qlib.workflow.record_temp import PortAnaRecord, SigAnaRecord

from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.report import analysis_model, analysis_position


@dataclass(frozen=True)
class ReportPaths:
    """报告输出路径集合。"""

    out_dir: Path
    html_path: Path


def _ensure_dt_inst_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    确保 DataFrame 的 MultiIndex 顺序为 (datetime, instrument)。

    Qlib 的一些产物可能是 (instrument, datetime) 或者 index 名称顺序不一致；
    后续的 join / concat 依赖一致的 index 顺序，因此这里统一整理。
    """
    if not isinstance(df.index, pd.MultiIndex):
        return df
    names = list(df.index.names)
    if names == ["datetime", "instrument"]:
        return df.sort_index()
    if set(names) == {"datetime", "instrument"}:
        df = df.copy()
        df.index = df.index.reorder_levels(["datetime", "instrument"])
        return df.sort_index()
    return df.sort_index()


def _infer_pred_col(pred_df: pd.DataFrame) -> str:
    """
    推断 pred_df 里“预测分数”的列名。

    上游训练脚本保存 pred.pkl 时列名可能是 `score`/`pred`/`0` 等，这里取第一个数值列。
    """
    for col in pred_df.columns:
        try:
            if pd.api.types.is_numeric_dtype(pred_df[col]):
                return str(col)
        except Exception:
            continue
    return str(pred_df.columns[0])


def _diagnose_empty_slice(
    *,
    pred_df: pd.DataFrame,
    report_normal_df: Optional[pd.DataFrame],
    positions: Optional[Any],
    backtest_start: str,
    backtest_end: str,
    topk: int,
    max_print_dates: int = 20,
    max_print_insts: int = 20,
) -> tuple[str, str]:
    """
    尝试解释 backtest 过程中反复出现的 `Mean of empty slice`。

    这个 warning 来源于 `np.nanmean` 对“空数组”求均值，一般意味着某些交易日可交易集合为空
    或者关键字段（如 `$close`）在当日对候选集合全是 NaN。

    我们无法从 warning 本身直接拿到“具体哪个标的”，所以这里给出可操作的定位方法：
    - 统计回测区间内 `$close` 缺失最严重的标的（Top N）。
    - 对每个交易日，取 TopK 信号标的，检查其中哪些标的当日 `$close` 缺失。
      若某日 TopK 全缺失，则很容易导致后续均值计算变成“空切片”。

    返回 (console_text, html_table) 两份内容：
    - console_text：写到 stdout，便于你在命令行直接看到结论；
    - html_table：写到报告，便于回看。
    """
    pred_df = _ensure_dt_inst_index(pred_df)
    pred_col = _infer_pred_col(pred_df)

    insts = pred_df.index.get_level_values("instrument").unique().tolist()
    dt_index = pred_df.index.get_level_values("datetime")
    pred_start = pd.to_datetime(dt_index.min()).date()
    pred_end = pd.to_datetime(dt_index.max()).date()

    # 诊断区间：尽量对齐回测区间（如果 pred 覆盖范围更窄，则取交集）。
    bt_start = pd.to_datetime(backtest_start).date()
    bt_end = pd.to_datetime(backtest_end).date()
    diag_start = max(pred_start, bt_start)
    diag_end = min(pred_end, bt_end)
    start = str(diag_start)
    end = str(diag_end)

    # 1) 拉取 close（用于判断“缺行情”）
    close_raw = D.features(insts, ["$close"], start, end, freq="day")
    close_col = close_raw.columns[0]
    close_df = close_raw[[close_col]].rename(columns={close_col: "close"})
    close_df = _ensure_dt_inst_index(close_df.swaplevel("instrument", "datetime").sort_index())

    # 2) 统计每个标的 close 缺失天数（只统计 pred 覆盖的日期范围）
    close_nan_count = close_df["close"].isna().groupby(level="instrument").sum()
    close_total_count = close_df["close"].groupby(level="instrument").size()
    close_nan_rate = (close_nan_count / close_total_count).fillna(0.0)

    top_inst = (
        pd.DataFrame(
            {
                "nan_days": close_nan_count,
                "total_days": close_total_count.reindex(close_nan_count.index),
                "nan_rate": close_nan_rate,
            }
        )
        .reset_index()
        .rename(columns={"instrument": "instrument"})
    )
    top_inst = top_inst.sort_values(["nan_days", "nan_rate", "instrument"], ascending=[False, False, True]).head(max_print_insts)

    # 3) 逐日检查：是否存在“当日信号为空/TopK 为空/TopK 里缺 close”
    pred_score = pred_df[pred_col]
    missing_rows: list[dict[str, Any]] = []
    empty_signal_dates: list[str] = []

    unique_dates = sorted(pd.to_datetime(dt_index.unique()).to_pydatetime().tolist())
    for dt in unique_dates:
        dt_key = pd.Timestamp(dt)
        try:
            day_scores = pred_score.xs(dt_key, level="datetime")
        except Exception:
            continue
        if day_scores.empty:
            continue

        day_scores = day_scores.dropna()
        if day_scores.empty:
            empty_signal_dates.append(str(dt_key.date()))
            continue

        # 取 TopK（如果不足 TopK，就按实际数量）
        day_topk = day_scores.sort_values(ascending=False).head(topk)
        topk_insts = day_topk.index.tolist()

        if not topk_insts:
            empty_signal_dates.append(str(dt_key.date()))
            continue

        # 检查 close 是否缺失（也包括“整条记录不存在”的情况）
        missing_insts: list[str] = []
        present_insts: list[str] = []
        for inst in topk_insts:
            try:
                v = close_df.loc[(dt_key, inst), "close"]
                if pd.isna(v):
                    missing_insts.append(inst)
                else:
                    present_insts.append(inst)
            except Exception:
                missing_insts.append(inst)

        if missing_insts:
            missing_rows.append(
                {
                    "date": str(dt_key.date()),
                    "topk": len(topk_insts),
                    "missing_in_topk": len(missing_insts),
                    "present_in_topk": len(present_insts),
                    "missing_instruments": ", ".join(missing_insts[:50]),
                }
            )

    missing_df = pd.DataFrame(missing_rows)
    # 优先展示“当日 TopK 缺失最多”的日期
    if not missing_df.empty:
        missing_df = missing_df.sort_values(["missing_in_topk", "date"], ascending=[False, True]).head(max_print_dates)

    # 4) 从 positions/report 侧再做一次“空集合”诊断（更贴近 backtest 内部的 mean(empty)）
    empty_position_dates: list[str] = []
    if isinstance(positions, pd.DataFrame) and not positions.empty:
        # 尝试统一 index 为 datetime
        pos_df = positions.copy()
        if isinstance(pos_df.index, pd.MultiIndex) and "datetime" in (pos_df.index.names or []):
            try:
                pos_df = pos_df.reset_index().set_index("datetime")
            except Exception:
                pass
        if isinstance(pos_df.index, pd.DatetimeIndex):
            # 常见结构：index=datetime, columns=instrument, values=position/weight
            try:
                holding_cnt = (pos_df.fillna(0.0).abs() > 0).sum(axis=1)
                empty_position_dates = [str(d.date()) for d in holding_cnt[holding_cnt == 0].index.to_pydatetime().tolist()]
            except Exception:
                pass

    report_nan_summary: Optional[pd.DataFrame] = None
    if isinstance(report_normal_df, pd.DataFrame) and not report_normal_df.empty:
        try:
            cols = [c for c in ["return", "bench", "cost", "turnover"] if c in report_normal_df.columns]
            if cols:
                report_nan_summary = (
                    pd.DataFrame(
                        {
                            "column": cols,
                            "nan_days": [int(report_normal_df[c].isna().sum()) for c in cols],
                            "total_days": [int(len(report_normal_df[c])) for c in cols],
                        }
                    )
                    .assign(nan_rate=lambda x: x["nan_days"] / x["total_days"])
                    .sort_values(["nan_days", "column"], ascending=[False, True])
                )
        except Exception:
            report_nan_summary = None

    # 4) 组织输出
    console_lines: list[str] = []
    console_lines.append("[diagnose] `Mean of empty slice` 常见原因：某些交易日可交易集合为空/关键行情字段全 NaN。")
    console_lines.append(f"[diagnose] pred_col={pred_col!r} backtest={backtest_start}..{backtest_end} pred_range={start}..{end} insts={len(insts)}")

    if not top_inst.empty:
        console_lines.append("[diagnose] `$close` 缺失最多的标的（Top）：")
        for _, row in top_inst.iterrows():
            console_lines.append(
                f"  - {row['instrument']}: nan_days={int(row['nan_days'])} total_days={int(row['total_days'])} nan_rate={float(row['nan_rate']):.2%}"
            )
    else:
        console_lines.append("[diagnose] 未统计到 `$close` 缺失（close_df 为空）。")

    if empty_signal_dates:
        console_lines.append(f"[diagnose] 当日信号全部为空（dropna 后无信号）的日期数={len(empty_signal_dates)}（示例：{', '.join(empty_signal_dates[:max_print_dates])}）")

    if missing_df is not None and not missing_df.empty:
        console_lines.append("[diagnose] TopK 信号集合中 `$close` 缺失的日期样例（缺失越多越靠前）：")
        for _, row in missing_df.iterrows():
            console_lines.append(
                f"  - {row['date']}: missing_in_topk={int(row['missing_in_topk'])}/{int(row['topk'])} missing=[{row['missing_instruments']}]"
            )
    else:
        console_lines.append("[diagnose] 未发现 TopK 集合里 `$close` 缺失的日期（或 pred/close 未对齐）。")

    if empty_position_dates:
        console_lines.append(f"[diagnose] positions 里检测到空仓日数={len(empty_position_dates)}（示例：{', '.join(empty_position_dates[:max_print_dates])}）")

    if report_nan_summary is not None and (report_nan_summary["nan_days"] > 0).any():
        bad = report_nan_summary[report_nan_summary["nan_days"] > 0]
        console_lines.append("[diagnose] report_normal_df 中存在 NaN 的列：")
        for _, row in bad.iterrows():
            console_lines.append(f"  - {row['column']}: nan_days={int(row['nan_days'])}/{int(row['total_days'])} ({float(row['nan_rate']):.2%})")

    html_parts: list[str] = []
    html_parts.append("<div><b>Mean of empty slice</b> 可能原因：某些交易日 TopK 候选集合关键行情字段（如 <code>$close</code>）缺失，导致可交易集合为空。</div>")
    html_parts.append(f"<div>pred_col=<code>{pred_col}</code> pred_range=<code>{start}..{end}</code> insts=<code>{len(insts)}</code></div>")
    if not top_inst.empty:
        html_parts.append("<h3>$close 缺失最多的标的（Top）</h3>")
        html_parts.append(top_inst.to_html(index=False))
    if missing_df is not None and not missing_df.empty:
        html_parts.append("<h3>TopK 信号集合中 $close 缺失的日期样例</h3>")
        html_parts.append(missing_df.to_html(index=False))
    if empty_signal_dates:
        html_parts.append("<h3>当日信号为空（dropna 后无信号）的日期样例</h3>")
        html_parts.append(pd.DataFrame({"date": empty_signal_dates[:max_print_dates]}).to_html(index=False))
    if empty_position_dates:
        html_parts.append("<h3>空仓日（positions 统计）样例</h3>")
        html_parts.append(pd.DataFrame({"date": empty_position_dates[:max_print_dates]}).to_html(index=False))
    if report_nan_summary is not None:
        html_parts.append("<h3>report_normal_df NaN 统计</h3>")
        html_parts.append(report_nan_summary.to_html(index=False))

    return "\n".join(console_lines), "".join(html_parts)


def _select_latest_recorder_id(exp_name: str) -> str:
    """
    在指定 experiment 下选择“最新”的 recorder_id。

    判定依据：recorder 的 start_time（越晚越新）。
    """
    exp = R.get_exp(experiment_name=exp_name, create=False)
    recs = exp.list_recorders()
    if not recs:
        raise RuntimeError(f"no recorders found under experiment={exp_name!r}")

    def _parse_time(x: Any) -> datetime:
        t = getattr(x, "start_time", None)
        if t is None:
            return datetime.min
        return pd.to_datetime(t).to_pydatetime()

    latest_id = max(recs.items(), key=lambda kv: _parse_time(kv[1]))[0]
    return latest_id


def _safe_load(recorder, key: str) -> Optional[Any]:
    """从 recorder 中读取对象；若不存在/读取失败则返回 None（用于“按需生成”逻辑）。"""
    try:
        return recorder.load_object(key)
    except Exception:
        return None


def _make_report_paths(*, exp_name: str, recorder_id: str) -> ReportPaths:
    """生成输出目录与 HTML 文件名（包含 exp、recorder_id 前缀与时间戳）。"""
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / f"qlib_report_{exp_name}_{recorder_id[:8]}.html"
    return ReportPaths(out_dir=out_dir, html_path=html_path)


def _plotly_figs_to_html(figs: list[Any], *, include_plotlyjs: str = "cdn") -> str:
    """
    将一组 plotly Figure 导出为 HTML 片段。

    - 第一张图包含 plotly.js（默认用 CDN），后续图不重复注入，避免 HTML 变大。
    - `displayModeBar=False`：报告里更干净。
    """
    try:
        import plotly.io as pio  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("plotly is required to export figures to HTML") from exc

    # 注意：notebook 里 plotly 默认会带右上角工具栏（缩放/保存图片等）。
    # 这里也保持一致，方便在报告里交互查看细节。
    parts: list[str] = []
    first = True
    for fig in figs:
        parts.append(
            pio.to_html(
                fig,
                include_plotlyjs=(include_plotlyjs if first else False),
                full_html=False,
                config={"displayModeBar": True, "displaylogo": False},
            )
        )
        first = False
    return "\n".join(parts)


def _build_html(*, title: str, sections: list[tuple[str, str]]) -> str:
    """拼装一个简单的单页 HTML（包含少量 CSS + 多个 section）。"""
    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
           margin: 24px; color: #111; }
    h1 { margin: 0 0 10px 0; font-size: 24px; padding-bottom: 10px; border-bottom: 2px solid #ddd; }
    .meta { color: #555; margin-bottom: 18px; }
    h2 { margin-top: 22px; font-size: 18px; padding-bottom: 6px; border-bottom: 1px solid #eee; }
    h3 { margin-top: 18px; font-size: 15px; }
    h4 { margin-top: 14px; font-size: 13px; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #ddd; padding: 6px 8px; font-size: 12px; }
    th { background: #f6f6f6; text-align: left; }
    code { background: #f6f6f6; padding: 2px 4px; border-radius: 4px; }
    .kv td:first-child { width: 260px; background: #fafafa; font-weight: 600; }
    .note { color: #555; font-size: 12px; margin-top: 8px; }
    .kv .kv-section td { background: #eef2ff; font-weight: 700; border-top: 2px solid #ddd; }
    """
    body_parts = [f"<h1>{title}</h1>"]
    body_parts.append(f"<div class='meta'>Generated at {datetime.now().isoformat(timespec='seconds')}</div>")
    for name, html in sections:
        body_parts.append(f"<h2>{name}</h2>")
        body_parts.append(html)
    return f"<!doctype html><html><head><meta charset='utf-8'><style>{css}</style></head><body>{''.join(body_parts)}</body></html>"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _html_escape(v: Any) -> str:
    return html_lib.escape("" if v is None else str(v))


def _render_kv_table(rows: list[tuple[str, Any]]) -> str:
    trs = []
    for k, v in rows:
        trs.append(f"<tr><td>{_html_escape(k)}</td><td>{_html_escape(v)}</td></tr>")
    return "<table class='kv'><tbody>" + "".join(trs) + "</tbody></table>"


def _render_kv_sections(sections: list[tuple[str, list[tuple[str, Any]]]]) -> str:
    trs: list[str] = []
    for title, rows in sections:
        trs.append(f"<tr class='kv-section'><td colspan='2'>{_html_escape(title)}</td></tr>")
        for k, v in rows:
            trs.append(f"<tr><td>{_html_escape(k)}</td><td>{_html_escape(v)}</td></tr>")
    return "<table class='kv'><tbody>" + "".join(trs) + "</tbody></table>"


def _read_local_mlflow_run_dir(*, experiment_id: str, run_id: str) -> Optional[Path]:
    base = Path("mlruns") / str(experiment_id) / str(run_id)
    return base if base.exists() else None


def _read_local_mlflow_kv_dir(dir_path: Path) -> dict[str, str]:
    if not dir_path.exists():
        return {}
    out: dict[str, str] = {}
    for p in dir_path.iterdir():
        if p.is_file():
            out[p.name] = _read_text(p).strip()
    return out


def _try_extract_train_defaults_from_script(script_path: Path) -> dict[str, str]:
    """
    Best-effort: extract argparse defaults from a known training script.
    This is primarily used when the recorder didn't log CLI args/params to MLflow.
    """
    if not script_path.exists() or not script_path.is_file():
        return {}
    txt = _read_text(script_path)
    defaults: dict[str, str] = {}
    for flag in [
        "--provider-uri",
        "--market",
        "--benchmark",
        "--start-time",
        "--end-time",
        "--train",
        "--valid",
        "--test",
        "--label-expr",
        "--pit-fields",
        "--pit-feature-prefix",
    ]:
        m = re.search(
            rf'add_argument\\(\\s*["\\\']{re.escape(flag)}["\\\'][^\\)]*?default\\s*=\\s*(["\\\'])(.*?)\\1',
            txt,
            re.S,
        )
        if m:
            defaults[flag.lstrip("-")] = m.group(2)
    return defaults


def _try_get_model_params(model: Any) -> dict[str, Any]:
    if model is None:
        return {}

    try:
        gp = getattr(model, "get_params", None)
        if callable(gp):
            out = gp()
            if isinstance(out, dict):
                return out
    except Exception:
        pass

    try:
        params = getattr(model, "params", None)
        if isinstance(params, dict):
            return params
    except Exception:
        pass

    # Fallback: expose a small subset of public attributes.
    out: dict[str, Any] = {}
    for k in dir(model):
        if k.startswith("_"):
            continue
        if k in {"model", "booster", "clf"}:
            continue
        try:
            val = getattr(model, k)
        except Exception:
            continue
        if isinstance(val, (str, int, float, bool)) or val is None:
            out[k] = val
    return out


def _try_match_market_from_instruments(*, instruments: list[str], start_time: str, end_time: str) -> Optional[tuple[str, float]]:
    candidates = ["csi300", "csi500", "csi800", "csi1000", "csiall", "all"]
    inst_set = {x.upper() for x in instruments if x}
    if not inst_set:
        return None
    best: Optional[tuple[str, float]] = None
    for c in candidates:
        try:
            cfg = D.instruments(c)
            lst = D.list_instruments(cfg, start_time=start_time, end_time=end_time, freq="day", as_list=True)
            cand_set = {x.upper() for x in lst}
            if not cand_set:
                continue
            score = len(inst_set & cand_set) / max(1, len(inst_set))
            if best is None or score > best[1]:
                best = (c, score)
        except Exception:
            continue
    return best


def _render_experiment_params_html(
    *,
    args: argparse.Namespace,
    recorder: Any,
    recorder_id: str,
    trained_model: Any,
    pred_df: pd.DataFrame,
) -> str:
    dt_index = pred_df.index.get_level_values("datetime")
    pred_start = str(pd.to_datetime(dt_index.min()).date())
    pred_end = str(pd.to_datetime(dt_index.max()).date())
    insts = pred_df.index.get_level_values("instrument").unique().tolist()
    pred_col = _infer_pred_col(pred_df)

    run_config = _safe_load(recorder, "run_config.pkl") or _safe_load(recorder, "run_config")
    if isinstance(run_config, dict):
        cfg_market = run_config.get("market")
        cfg_benchmark = run_config.get("benchmark")
        cfg_provider_uri = run_config.get("provider_uri")
        cfg_start_time = run_config.get("start_time")
        cfg_end_time = run_config.get("end_time")
        cfg_label_expr = run_config.get("label_expr")
        cfg_pit_fields = run_config.get("pit_fields")
        cfg_segments = run_config.get("segments")
    else:
        cfg_market = None
        cfg_benchmark = None
        cfg_provider_uri = None
        cfg_start_time = None
        cfg_end_time = None
        cfg_label_expr = None
        cfg_pit_fields = None
        cfg_segments = None

    run_dir = _read_local_mlflow_run_dir(experiment_id=str(getattr(recorder, "experiment_id", "")), run_id=recorder_id)
    params = _read_local_mlflow_kv_dir(run_dir / "params") if run_dir else {}
    tags = _read_local_mlflow_kv_dir(run_dir / "tags") if run_dir else {}

    train_script = tags.get("mlflow.source.name") or params.get("cmd-sys.argv") or ""
    train_defaults: dict[str, str] = {}
    if train_script:
        try:
            p = Path(train_script)
            if not p.is_absolute():
                p = Path.cwd() / p
            train_defaults = _try_extract_train_defaults_from_script(p)
        except Exception:
            train_defaults = {}

    inferred_train = train_defaults.get("train")
    inferred_valid = train_defaults.get("valid")
    inferred_test = train_defaults.get("test")
    inferred_market = train_defaults.get("market")
    inferred_label_expr = train_defaults.get("label-expr")
    inferred_pit_fields = train_defaults.get("pit-fields")

    model_params = _try_get_model_params(trained_model)
    saved_model_config = _safe_load(recorder, "model_config.pkl") or _safe_load(recorder, "model_config")
    saved_lgb_params = _safe_load(recorder, "lgb_params.pkl") or _safe_load(recorder, "lgb_params")
    if isinstance(saved_lgb_params, dict):
        model_params = saved_lgb_params
    elif isinstance(saved_model_config, dict):
        maybe_kwargs = saved_model_config.get("kwargs")
        if isinstance(maybe_kwargs, dict) and maybe_kwargs:
            model_params = maybe_kwargs

    def _fmt_segment(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, (list, tuple)) and len(v) == 2:
            return f"{v[0]} .. {v[1]}"
        return str(v)

    def _fmt_seg_str(v: Optional[str]) -> str:
        if not v:
            return ""
        parts = [x.strip() for x in v.split(",") if x.strip()]
        if len(parts) == 2:
            return f"{parts[0]} .. {parts[1]}"
        return v

    seg_train = ""
    seg_valid = ""
    seg_test = ""
    if isinstance(cfg_segments, dict):
        seg_train = _fmt_segment(cfg_segments.get("train"))
        seg_valid = _fmt_segment(cfg_segments.get("valid"))
        seg_test = _fmt_segment(cfg_segments.get("test"))
    else:
        seg_train = _fmt_seg_str(inferred_train)
        seg_valid = _fmt_seg_str(inferred_valid)
        seg_test = _fmt_seg_str(inferred_test)

    if not seg_test:
        seg_test = f"{pred_start} .. {pred_end}"

    pit_fields_str = ""
    if isinstance(cfg_pit_fields, (list, tuple)):
        pit_fields_str = ",".join([str(x) for x in cfg_pit_fields])
    elif cfg_pit_fields is not None:
        pit_fields_str = str(cfg_pit_fields)
    if not pit_fields_str:
        pit_fields_str = inferred_pit_fields or params.get("pit_fields", "")

    market = cfg_market or inferred_market or ""
    label_expr_train = cfg_label_expr or inferred_label_expr or ""

    model_rows: list[tuple[str, Any]] = [
        ("model_class", type(trained_model).__name__ if trained_model is not None else "(missing trained_model)"),
    ]
    if isinstance(model_params, dict) and model_params:
        for k in sorted(model_params.keys(), key=lambda x: str(x)):
            model_rows.append((str(k), model_params.get(k)))

    warnings: list[tuple[str, Any]] = []
    if label_expr_train and args.label_expr and label_expr_train != args.label_expr:
        warnings.append(("label_expr_mismatch", f"train={label_expr_train} report_eval={args.label_expr}"))

    sections = [
        (
            "Run",
            [
                ("experiment_name", args.exp_name),
                ("recorder_id", recorder_id),
                ("experiment_id", getattr(recorder, "experiment_id", "")),
                ("train_script", train_script or ""),
            ],
        ),
        (
            "Data",
            [
                ("provider_uri", cfg_provider_uri or args.provider_uri),
                ("market", market),
                ("benchmark", cfg_benchmark or args.benchmark),
                ("handler_range", _fmt_segment((cfg_start_time, cfg_end_time)) if cfg_start_time and cfg_end_time else ""),
                ("segment.train", seg_train),
                ("segment.valid", seg_valid),
                ("segment.test", seg_test),
                ("label_expr", label_expr_train),
                ("pit_fields", pit_fields_str),
            ],
        ),
        (
            "Prediction",
            [
                ("pred_coverage", f"{pred_start} .. {pred_end}"),
                ("pred_instruments", f"{len(insts)} unique"),
                ("pred_column", pred_col),
            ],
        ),
        ("Model", model_rows),
        (
            "Strategy",
            [
                ("strategy.topk", args.topk),
                ("strategy.n_drop", args.n_drop),
                ("strategy.hold_thresh", args.hold_thresh),
            ],
        ),
        (
            "Exchange",
            [
                ("exchange.open_cost", args.open_cost),
                ("exchange.close_cost", args.close_cost),
                ("exchange.min_cost", args.min_cost),
                ("exchange.limit_threshold", args.limit_threshold),
            ],
        ),
    ]

    if warnings:
        sections.append(("Warnings", warnings))

    return _render_kv_sections(sections)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Generate Qlib prediction/backtest/analysis HTML report from a trained recorder.")

    # provider-uri：Qlib 数据目录（cn_data / us_data 等）。这里默认与你仓库的目录一致。
    p.add_argument("--provider-uri", default="data/qlib_data/cn_data")
    # exp-name / recorder-id：训练时写入到 Qlib workflow 的 Experiment/Recorder。
    # 训练脚本（如 src/train/train_lgb_alpha158_pit.py）通常会打印 recorder_id。
    p.add_argument("--exp-name", default="tutorial_exp")
    p.add_argument("--recorder-id", default=None)
    # benchmark：回测基准（如沪深300：SH000300）。
    p.add_argument("--benchmark", default="SH000300")
    # backtest-start/end：回测区间。注意应覆盖 pred 的日期范围（否则图表/统计会缺失）。
    p.add_argument("--backtest-start", default="2019-01-01")
    p.add_argument("--backtest-end", default="2020-12-31")
    # label-expr：用于评估预测效果的标签表达式（D.features 可解析的 Qlib 表达式）。
    # 若不传，则默认使用训练时保存到 recorder 的 label_expr（run_config.pkl）；再不行才回退到“未来 5 日收益率”。
    p.add_argument("--label-expr", default=None)

    # 策略参数：TopkDropoutStrategy（与 notebook/示例保持一致）。
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--n-drop", type=int, default=1)
    p.add_argument("--hold-thresh", type=int, default=5)

    # 账户与交易成本参数（影响回测收益/风险指标）。
    p.add_argument("--account", type=float, default=100_000_000)
    p.add_argument("--open-cost", type=float, default=0.0005)
    p.add_argument("--close-cost", type=float, default=0.0015)
    p.add_argument("--min-cost", type=float, default=5.0)
    p.add_argument("--limit-threshold", type=float, default=0.095)
    p.add_argument("--threads", type=int, default=8)
    # 是否在报告生成时输出“Mean of empty slice”诊断（会额外拉取一次 `$close` 做缺失统计）。
    # Python 3.9+ 支持 BooleanOptionalAction：同时提供 `--diagnose-empty-slice` 与 `--no-diagnose-empty-slice`。
    p.add_argument(
        "--diagnose-empty-slice",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = p.parse_args(argv)

    # 初始化 Qlib（只读数据）。
    qlib.init(provider_uri=args.provider_uri, region=REG_CN)

    # recorder 选择逻辑：不传则取 experiment 下最新的一个（方便“刚训练完直接出报告”）。
    recorder_id = args.recorder_id or _select_latest_recorder_id(args.exp_name)
    paths = _make_report_paths(exp_name=args.exp_name, recorder_id=recorder_id)

    # 直接获取 recorder 读取产物（避免 resume 写入 MLflow param 冲突）。
    recorder = R.get_recorder(recorder_id=recorder_id, experiment_name=args.exp_name)
    trained_model = _safe_load(recorder, "trained_model")

    # Prefer the training label_expr saved on the recorder, unless explicitly overridden by CLI.
    if args.label_expr is None:
        run_config = _safe_load(recorder, "run_config.pkl") or _safe_load(recorder, "run_config")
        if isinstance(run_config, dict):
            saved = run_config.get("label_expr")
            if isinstance(saved, str) and saved.strip():
                args.label_expr = saved.strip()
    if args.label_expr is None:
        args.label_expr = "Ref($close, -5) / $close - 1"

    # 回测/持仓分析配置（Qlib 内部用 dict 表达）。
    port_analysis_config = {
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
        },
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {"signal": "<PRED>", "topk": args.topk, "n_drop": args.n_drop, "hold_thresh": args.hold_thresh},
        },
        "backtest": {
            "start_time": args.backtest_start,
            "end_time": args.backtest_end,
            "account": args.account,
            "benchmark": args.benchmark,
            "exchange_kwargs": {
                "freq": "day",
                "limit_threshold": args.limit_threshold,
                "deal_price": "close",
                "open_cost": args.open_cost,
                "close_cost": args.close_cost,
                "min_cost": args.min_cost,
            },
        },
    }

    # Ensure backtest + analysis artifacts exist on the recorder.
    # NOTE: do NOT use R.start(resume=True) on an existing recorder_id, because MLflow forbids overwriting params
    # like `cmd-sys.argv` and Qlib may attempt to log it again.
    # 这里采用“缺什么补什么”：
    # - portfolio_analysis/report_normal_1day.pkl：回测结果（收益、基准、成本等）
    # - sig_analysis/ic_1day.pkl：信号分析（IC 等）
    if _safe_load(recorder, "portfolio_analysis/report_normal_1day.pkl") is None:
        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()

    if _safe_load(recorder, "sig_analysis/ic_1day.pkl") is None:
        sar = SigAnaRecord(recorder)
        sar.generate()

    # 读取预测信号（必须存在）。
    pred_df = _safe_load(recorder, "pred.pkl")
    if pred_df is None:
        raise RuntimeError(f"missing artifact pred.pkl under recorder_id={recorder_id}")
    pred_df = _ensure_dt_inst_index(pred_df)

    # 读取回测主报告（必须存在；若不存在，上面会生成）。
    report_normal_df = _safe_load(recorder, "portfolio_analysis/report_normal_1day.pkl")
    if report_normal_df is None:
        raise RuntimeError("missing artifact portfolio_analysis/report_normal_1day.pkl")

    # 读取更详细的组合分析表（可选）。不同 Qlib 版本/配置可能没有该产物。
    analysis_df = _safe_load(recorder, "portfolio_analysis/port_analysis_1day.pkl")
    positions = _safe_load(recorder, "portfolio_analysis/positions_normal_1day.pkl")

    diag_console = ""
    diag_html = ""
    if args.diagnose_empty_slice:
        try:
            diag_console, diag_html = _diagnose_empty_slice(
                pred_df=pred_df,
                report_normal_df=report_normal_df,
                positions=positions,
                backtest_start=args.backtest_start,
                backtest_end=args.backtest_end,
                topk=args.topk,
            )
            # 输出到控制台，方便你直接看到“哪些标的/哪些天”缺行情。
            print(diag_console)
        except Exception as exc:
            print(f"[diagnose] failed to diagnose empty-slice causes: {exc}")

    # 生成“回测/持仓”图表（plotly Figures）。
    pos_figs = analysis_position.report_graph(report_normal_df, show_notebook=False)
    for fig in pos_figs:
        try:
            # 有些图默认是 marker，这里更偏向线图展示；并强制 x 轴按日期显示。
            fig.update_traces(mode="lines", selector=dict(type="scatter"))
            fig.update_xaxes(type="date")
            # 与 notebook 保持一致：按季度展示刻度（M3），只显示到“年-月”。
            fig.update_xaxes(
                tickmode="linear",
                tick0=args.backtest_start,
                dtick="M3",
                tickformat="%Y-%m",
                ticklabelmode="period",
                tickangle=0,
            )
        except Exception:
            pass
    pos_html = _plotly_figs_to_html(pos_figs)

    # 生成“模型效果”图表（需要 label 与 pred 对齐）。
    # 做法：
    # 1) 从 pred 的 index 拿到 instruments 与日期范围；
    # 2) 用 D.features 计算 label；
    # 3) 与 pred 做 inner join，得到 pred_label；
    # 4) 调 Qlib 的 analysis_model 绘图。
    insts = pred_df.index.get_level_values("instrument").unique().tolist()
    dt_index = pred_df.index.get_level_values("datetime")
    label_start = str(pd.to_datetime(dt_index.min()).date())
    label_end = str(pd.to_datetime(dt_index.max()).date())
    label_raw = D.features(insts, [args.label_expr], label_start, label_end, freq="day")
    label_col = label_raw.columns[0]
    label_df = label_raw[[label_col]].rename(columns={label_col: "label"})
    # D.features 返回的 index 层级顺序通常是 (instrument, datetime)，这里交换成 (datetime, instrument)。
    label_df = _ensure_dt_inst_index(label_df.swaplevel("instrument", "datetime").sort_index())

    # pred_df 的列名可能是 score / pred / 0（取决于上游训练脚本怎么保存）。
    # 这里不强行改列名，只要与 label 对齐即可。
    pred_label = pd.concat([label_df, pred_df], axis=1, join="inner").sort_index()
    model_figs = analysis_model.model_performance_graph(pred_label, show_notebook=False)
    # 与 notebook 展示对齐：去掉 marker（notebook 里只处理了 report_graph 的 marker；
    # model_performance_graph 不做强制坐标轴格式化，避免影响直方图/热力图/QQ 图）。
    for fig in model_figs:
        try:
            fig.update_traces(mode="lines", selector=dict(type="scatter"))
        except Exception:
            pass
    # plotly.js 只在第一组图里注入一次；这里关闭以避免重复。
    model_html = _plotly_figs_to_html(model_figs, include_plotlyjs=False)

    # Portfolio Analysis Table：尽量把“组合分析”相关内容集中在一个 section 里展示。
    # - 风险指标（基于 excess return）
    # - Qlib 生成的 port_analysis_1day.pkl（若存在）
    risk_blocks: list[str] = []
    try:
        excess_wo = report_normal_df["return"] - report_normal_df["bench"]
        risk_blocks.append("<h3>Excess (w/o cost)</h3>" + pd.DataFrame([risk_analysis(excess_wo)]).to_html(index=False))
        if "cost" in report_normal_df.columns:
            excess_wc = report_normal_df["return"] - report_normal_df["bench"] - report_normal_df["cost"]
            risk_blocks.append(
                "<h3>Excess (with cost)</h3>" + pd.DataFrame([risk_analysis(excess_wc)]).to_html(index=False)
            )
    except Exception:
        pass

    port_analysis_html = "".join(risk_blocks)
    if analysis_df is not None:
        port_analysis_html += "<h3>Portfolio Analysis Table</h3>" + analysis_df.to_html()
    if not port_analysis_html:
        port_analysis_html = "<div class='note'>(no portfolio analysis table found on recorder)</div>"

    data_quality_html = diag_html or "<div class='note'>(disabled or no issues detected)</div>"

    exp_params_html = _render_experiment_params_html(
        args=args,
        recorder=recorder,
        recorder_id=recorder_id,
        trained_model=trained_model,
        pred_df=pred_df,
    )

    # Sections order (as requested):
    # 1. Experiment Parameters
    # 2. Portfolio Analysis Table
    # 3. Backtest & Position
    # 4. Model Performance
    # 5. Data Quality
    sections = [
        ("Experiment Parameters", exp_params_html),
        ("Portfolio Analysis Table", port_analysis_html),
        ("Backtest & Position", pos_html),
        ("Model Performance", model_html),
        ("Data Quality", data_quality_html),
    ]

    # 输出报告。
    html = _build_html(title="Qlib Report", sections=sections)
    paths.html_path.write_text(html, encoding="utf-8")
    print(f"written: {paths.html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
