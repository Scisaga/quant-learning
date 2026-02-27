#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


IR_COL = "excess_return_with_cost.information_ratio"
ANNRET_COL = "excess_return_with_cost.annualized_return"
MAXDD_COL = "excess_return_with_cost.max_drawdown"

IR_NC_COL = "excess_return_without_cost.information_ratio"
ANNRET_NC_COL = "excess_return_without_cost.annualized_return"
MAXDD_NC_COL = "excess_return_without_cost.max_drawdown"


@dataclass(frozen=True)
class RunFiles:
    run_dir: Path
    summary_json: Path
    results_jsonl: Path
    grid_compare_csv: Path
    grid_errors_csv: Path
    out_md: Path


def _escape_md(value: Any) -> str:
    if value is None:
        return "-"
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _fmt_float(value: Any, *, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    try:
        if pd.isna(value):
            return "-"
    except Exception:
        pass
    try:
        number = float(value)
    except Exception:
        return _escape_md(value)
    if math.isfinite(number):
        return f"{number:.{digits}f}"
    return "-"


def _fmt_pct(value: Any, *, digits: int = 1) -> str:
    if value is None:
        return "-"
    try:
        if pd.isna(value):
            return "-"
    except Exception:
        pass
    try:
        number = float(value)
    except Exception:
        return _escape_md(value)
    if not math.isfinite(number):
        return "-"
    return f"{number * 100:.{digits}f}%"


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = [
        "| " + " | ".join(_escape_md(h) for h in headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(_escape_md(v) for v in row) + " |")
    return "\n".join(out)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _status_counter(results: Iterable[dict[str, Any]]) -> Counter:
    c: Counter = Counter()
    for r in results:
        c[r.get("status", "unknown")] += 1
    return c


def _ensure_files(run_dir: Path, *, out_md: Path | None) -> RunFiles:
    run_dir = run_dir.resolve()
    summary_json = run_dir / "summary.json"
    results_jsonl = run_dir / "results.jsonl"
    grid_compare_csv = run_dir / "grid_compare.csv"
    grid_errors_csv = run_dir / "grid_errors.csv"
    if out_md is None:
        out_md = run_dir / "grid_run_analysis.md"

    missing = [p for p in [summary_json, results_jsonl, grid_compare_csv] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files: " + ", ".join(str(p) for p in missing))

    return RunFiles(
        run_dir=run_dir,
        summary_json=summary_json,
        results_jsonl=results_jsonl,
        grid_compare_csv=grid_compare_csv,
        grid_errors_csv=grid_errors_csv,
        out_md=out_md,
    )


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in [IR_COL, ANNRET_COL, MAXDD_COL, IR_NC_COL, ANNRET_NC_COL, MAXDD_NC_COL]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if ANNRET_NC_COL in df.columns and ANNRET_COL in df.columns:
        df["cost_drag.annualized_return"] = df[ANNRET_NC_COL] - df[ANNRET_COL]
    else:
        df["cost_drag.annualized_return"] = pd.NA

    if "window.test_start" in df.columns:
        df["window.test_start"] = pd.to_datetime(df["window.test_start"], errors="coerce")
        df["test_year"] = df["window.test_start"].dt.year
    else:
        df["test_year"] = pd.NA

    return df


def _check_alignment(df: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    if "label_horizon" in df.columns and "strategy.hold_thresh" in df.columns:
        tmp = df[["label_horizon", "strategy.hold_thresh"]].dropna().drop_duplicates()
        mismatched = tmp[tmp["label_horizon"] != tmp["strategy.hold_thresh"]]
        if not mismatched.empty:
            samples = mismatched.head(5).to_dict(orient="records")
            warnings.append(f"- 发现 label_horizon 与 hold_thresh 不一致：{samples}")
    return warnings


def _group_summary(
    df: pd.DataFrame, *, group_cols: list[str], top_n: int
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    g = df.groupby(group_cols, dropna=False)

    summary = (
        g.agg(
            n=("key", "size"),
            ir_median=(IR_COL, "median"),
            annret_median=(ANNRET_COL, "median"),
            maxdd_median=(MAXDD_COL, "median"),
            ir_pos_rate=(IR_COL, lambda s: float((s > 0).mean()) if len(s) else float("nan")),
            maxdd_worst=(MAXDD_COL, "min"),
            pred_inst_median=("pred_instruments", "median"),
            cost_drag_annret_median=("cost_drag.annualized_return", "median"),
        )
        .reset_index()
        .sort_values(["ir_median", "annret_median"], ascending=[False, False], na_position="last")
    )

    idx_best = g[IR_COL].idxmax()
    idx_worst = g[IR_COL].idxmin()
    best_rows = (
        df.loc[idx_best]
        .sort_values(IR_COL, ascending=False, na_position="last")
        .head(top_n)
        .to_dict(orient="records")
    )
    worst_rows = (
        df.loc[idx_worst]
        .sort_values(IR_COL, ascending=True, na_position="first")
        .head(top_n)
        .to_dict(orient="records")
    )
    return summary, best_rows, worst_rows


def _year_trend(df: pd.DataFrame, *, group_cols: list[str]) -> pd.DataFrame:
    cols = [*group_cols, "test_year"]
    cols = [c for c in cols if c in df.columns]
    if "test_year" not in cols:
        return pd.DataFrame()

    return (
        df.groupby(cols, dropna=False)
        .agg(
            n=("key", "size"),
            ir_median=(IR_COL, "median"),
            annret_median=(ANNRET_COL, "median"),
            maxdd_median=(MAXDD_COL, "median"),
        )
        .reset_index()
        .sort_values(cols)
    )


def _pick_best_horizon_per_market(summary_mh: pd.DataFrame) -> list[tuple[str, int]]:
    if summary_mh.empty:
        return []
    if "market" not in summary_mh.columns or "label_horizon" not in summary_mh.columns:
        return []
    best: list[tuple[str, int]] = []
    for market, sub in summary_mh.groupby("market", dropna=False):
        sub = sub.sort_values(["ir_median", "annret_median"], ascending=[False, False], na_position="last")
        if sub.empty:
            continue
        row0 = sub.iloc[0]
        best.append((str(market), int(row0["label_horizon"])))
    return best


def analyze(run_dir: Path, *, out_md: Path | None, top_n: int, include_pit_appendix: bool) -> Path:
    files = _ensure_files(run_dir, out_md=out_md)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    summary = _read_json(files.summary_json)
    results = list(_iter_jsonl(files.results_jsonl))
    status = _status_counter(results)

    df = pd.read_csv(files.grid_compare_csv)
    df = _add_derived_columns(df)

    warnings = _check_alignment(df)
    df_sorted_ir = df.sort_values(IR_COL, ascending=False, na_position="last")
    df_sorted_ann = df.sort_values(ANNRET_COL, ascending=False, na_position="last")

    lines: list[str] = []
    lines.append(f"# Grid Run 分析报告")
    lines.append("")
    lines.append(f"- run_dir: `{files.run_dir}`")
    lines.append(f"- generated_at: `{generated_at}`")
    lines.append("")

    # 1. 产物结构
    lines.append("## 1. 产物结构")
    lines.append("")
    lines.append(_md_table(["文件", "说明"], [
        [files.summary_json.name, "网格与窗口配置（walk-forward）"],
        [files.results_jsonl.name, "每个 job 一行（status/recorder/log 等）"],
        [files.grid_compare_csv.name, "成功（ok）job 的指标对比表（推荐入口）"],
        [files.grid_errors_csv.name, "失败与异常汇总（若存在）"],
    ]))
    lines.append("")

    # 2. 数据概览
    lines.append("## 2. 数据概览（量化策略视角）")
    lines.append("")

    total_jobs = summary.get("num_jobs", sum(status.values()))
    ok_jobs = status.get("ok", 0)
    lines.append(f"- job: total={total_jobs}, ok={ok_jobs}, others={total_jobs - ok_jobs}")
    lines.append(f"- markets={summary.get('markets')}, label_horizons={summary.get('label_horizons')}, pit_specs={summary.get('pit_specs')}")
    if "workers" in summary:
        lines.append(f"- workers={summary.get('workers')}, lgb_num_threads={summary.get('lgb_num_threads')}")
    lines.append("")

    if warnings:
        lines.append("### 2.1 一致性检查（可能导致结果失真）")
        lines.extend(warnings)
        lines.append("")

    lines.append("### 2.2 全局分布（with cost / 超额口径）")
    lines.append("")
    if all(c in df.columns for c in [ANNRET_COL, IR_COL, MAXDD_COL]):
        desc = df[[ANNRET_COL, IR_COL, MAXDD_COL]].describe(percentiles=[0.1, 0.5, 0.9]).T
        rows = []
        for metric, row in desc.iterrows():
            rows.append([
                metric.replace("excess_return_with_cost.", ""),
                _fmt_float(row.get("min")),
                _fmt_float(row.get("10%")),
                _fmt_float(row.get("50%")),
                _fmt_float(row.get("90%")),
                _fmt_float(row.get("max")),
            ])
        lines.append(_md_table(["metric", "min", "p10", "p50", "p90", "max"], rows))
        lines.append("")
    else:
        lines.append("- 缺少关键列，无法生成全局分布表。")
        lines.append("")

    # 基准映射
    if "market" in df.columns and "benchmark" in df.columns:
        bm = df[["market", "benchmark"]].dropna().drop_duplicates().sort_values(["market", "benchmark"])
        if not bm.empty:
            rows = [[r["market"], r["benchmark"]] for r in bm.to_dict(orient="records")]
            lines.append("### 2.3 benchmark 对齐（同 universe 基准）")
            lines.append("")
            lines.append(_md_table(["market", "benchmark"], rows))
            lines.append("")

    # 覆盖率/可交易集合
    if "pred_instruments" in df.columns:
        pred_desc = df["pred_instruments"].describe(percentiles=[0.05, 0.1]).to_dict()
        lines.append("### 2.4 可交易性快检（coverage）")
        lines.append("")
        lines.append(
            "- pred_instruments: "
            + f"min={_fmt_float(pred_desc.get('min'), digits=0)}, "
            + f"p10={_fmt_float(pred_desc.get('10%'), digits=0)}, "
            + f"p50={_fmt_float(pred_desc.get('50%'), digits=0)}, "
            + f"max={_fmt_float(pred_desc.get('max'), digits=0)}"
        )
        low_cov = df.sort_values("pred_instruments", ascending=True, na_position="first").head(min(8, len(df)))
        if not low_cov.empty:
            rows = []
            for r in low_cov.to_dict(orient="records"):
                rows.append([
                    r.get("market", "-"),
                    r.get("label_horizon", "-"),
                    r.get("pit", "-"),
                    int(r.get("test_year")) if pd.notna(r.get("test_year")) else "-",
                    _fmt_float(r.get("pred_instruments"), digits=0),
                    _fmt_float(r.get(IR_COL)),
                    r.get("key", "-"),
                ])
            lines.append("")
            lines.append("- coverage 极低样本（优先排查数据/可交易集合/停牌与缺失）：")
            lines.append(_md_table(["market", "h", "pit", "test_year", "pred_inst", "IR", "key"], rows))
        lines.append("")

    # 成本拖累
    if "cost_drag.annualized_return" in df.columns and df["cost_drag.annualized_return"].notna().any():
        cost_desc = df["cost_drag.annualized_return"].describe(percentiles=[0.5, 0.9]).to_dict()
        lines.append("### 2.5 成本拖累（without_cost → with_cost）")
        lines.append("")
        lines.append(
            "- cost_drag.annualized_return: "
            + f"p50={_fmt_float(cost_desc.get('50%'))}, "
            + f"p90={_fmt_float(cost_desc.get('90%'))}, "
            + f"max={_fmt_float(cost_desc.get('max'))}"
        )
        lines.append("")

    # “哪个策略最好”（粗粒度导航）
    lines.append("### 2.6 粗粒度导航：哪个参数组整体更好、更稳")
    lines.append("")

    summary_mh, best_rows_mh, worst_rows_mh = _group_summary(
        df, group_cols=["market", "label_horizon"], top_n=top_n
    )
    if summary_mh.empty:
        lines.append("- 无法从 grid_compare.csv 生成分组汇总（列缺失或数据为空）。")
        lines.append("")
    else:
        rows = []
        for r in summary_mh.head(12).to_dict(orient="records"):
            rows.append([
                r.get("market"),
                r.get("label_horizon"),
                int(r.get("n", 0)),
                _fmt_float(r.get("annret_median")),
                _fmt_float(r.get("ir_median")),
                _fmt_pct(r.get("ir_pos_rate")),
                _fmt_float(r.get("maxdd_median")),
                _fmt_float(r.get("maxdd_worst")),
                _fmt_float(r.get("pred_inst_median"), digits=0),
            ])
        lines.append(_md_table(
            ["market", "h", "n", "AnnRet_med", "IR_med", "IR>0", "MaxDD_med", "MaxDD_worst", "pred_inst_med"],
            rows,
        ))
        lines.append("")
        lines.append("- 解释：先用 `IR_med/IR>0` 看“是否有稳定 alpha”，再用 `MaxDD_worst` 看“最差窗口能否接受”，最后再看 `AnnRet_med`。")
        lines.append("")

    lines.append("### 2.7 最佳/最差窗口样本（用于定位 regime 与异常）")
    lines.append("")
    top_rows = df_sorted_ir.head(min(top_n, len(df))).to_dict(orient="records")
    bottom_rows = df.sort_values(IR_COL, ascending=True, na_position="first").head(min(top_n, len(df))).to_dict(orient="records")
    if top_rows:
        rows = []
        for r in top_rows:
            rows.append([
                r.get("market", "-"),
                r.get("label_horizon", "-"),
                r.get("pit", "-"),
                int(r.get("test_year")) if pd.notna(r.get("test_year")) else "-",
                _fmt_float(r.get(ANNRET_COL)),
                _fmt_float(r.get(IR_COL)),
                _fmt_float(r.get(MAXDD_COL)),
                r.get("key", "-"),
            ])
        lines.append("- IR 最好窗口（单窗口）：")
        lines.append(_md_table(["market", "h", "pit", "test_year", "AnnRet", "IR", "MaxDD", "key"], rows))
        lines.append("")
    if bottom_rows:
        rows = []
        for r in bottom_rows:
            rows.append([
                r.get("market", "-"),
                r.get("label_horizon", "-"),
                r.get("pit", "-"),
                int(r.get("test_year")) if pd.notna(r.get("test_year")) else "-",
                _fmt_float(r.get(ANNRET_COL)),
                _fmt_float(r.get(IR_COL)),
                _fmt_float(r.get(MAXDD_COL)),
                r.get("key", "-"),
            ])
        lines.append("- IR 最差窗口（单窗口）：")
        lines.append(_md_table(["market", "h", "pit", "test_year", "AnnRet", "IR", "MaxDD", "key"], rows))
        lines.append("")

    # 3. 分组对比（静态汇总）
    lines.append("## 3. 分组对比（静态汇总）")
    lines.append("")
    if not summary_mh.empty:
        best_by_market = _pick_best_horizon_per_market(summary_mh)
        if best_by_market:
            lines.append("- 每个 market 按 IR_med 选出的候选（用于下一节动态趋势）：")
            for market, h in best_by_market:
                lines.append(f"  - {market}: h={h}")
            lines.append("")

    if "pit" in df.columns:
        lines.append("### 3.1 按 pit 分组（不作为主结论，仅供参考）")
        lines.append("")
        summary_mhp, _, _ = _group_summary(df, group_cols=["market", "label_horizon", "pit"], top_n=top_n)
        if not summary_mhp.empty:
            rows = []
            for r in summary_mhp.to_dict(orient="records"):
                rows.append([
                    r.get("market"),
                    r.get("label_horizon"),
                    r.get("pit"),
                    int(r.get("n", 0)),
                    _fmt_float(r.get("annret_median")),
                    _fmt_float(r.get("ir_median")),
                    _fmt_pct(r.get("ir_pos_rate")),
                    _fmt_float(r.get("maxdd_median")),
                ])
            lines.append(_md_table(["market", "h", "pit", "n", "AnnRet_med", "IR_med", "IR>0", "MaxDD_med"], rows))
            lines.append("")

    # 4. 动态趋势（按窗口滚动）
    lines.append("## 4. 动态趋势（按窗口滚动）")
    lines.append("")
    best_by_market = _pick_best_horizon_per_market(summary_mh) if not summary_mh.empty else []
    if not best_by_market:
        lines.append("- 无法选出候选组（可能是分组汇总为空）。")
        lines.append("")
    else:
        trend = _year_trend(df, group_cols=["market", "label_horizon", "pit"] if "pit" in df.columns else ["market", "label_horizon"])
        for market, h in best_by_market:
            sub = trend[(trend["market"] == market) & (trend["label_horizon"] == h)].copy()
            if sub.empty:
                continue
            lines.append(f"### 4.{len([l for l in lines if l.startswith('### 4.')]) + 1} {market} / h={h}")
            lines.append("")
            if "pit" in sub.columns:
                sub2 = sub.pivot_table(
                    index="test_year",
                    columns="pit",
                    values=["annret_median", "ir_median", "maxdd_median"],
                    aggfunc="first",
                )
                sub2 = sub2.sort_index()
                pits = [c for c in sorted(df["pit"].dropna().unique().tolist()) if c in sub2.columns.get_level_values(1)]
                headers = ["test_year"]
                for pit in pits:
                    headers.extend([f"AnnRet_med[{pit}]", f"IR_med[{pit}]", f"MaxDD_med[{pit}]"])
                rows = []
                for year, row in sub2.iterrows():
                    r = [int(year) if pd.notna(year) else "-"]
                    for pit in pits:
                        r.append(_fmt_float(row.get(("annret_median", pit))))
                        r.append(_fmt_float(row.get(("ir_median", pit))))
                        r.append(_fmt_float(row.get(("maxdd_median", pit))))
                    rows.append(r)
                lines.append(_md_table(headers, rows))
            else:
                rows = []
                for r in sub.to_dict(orient="records"):
                    rows.append([
                        int(r.get("test_year")) if pd.notna(r.get("test_year")) else "-",
                        int(r.get("n", 0)),
                        _fmt_float(r.get("annret_median")),
                        _fmt_float(r.get("ir_median")),
                        _fmt_float(r.get("maxdd_median")),
                    ])
                lines.append(_md_table(["test_year", "n", "AnnRet_med", "IR_med", "MaxDD_med"], rows))
            lines.append("")
        lines.append("- 解读建议：重点看 IR 是否“跨年持续为正”和是否存在集中回撤年份（regime）。")
        lines.append("")

    # 5. 异常与失败
    lines.append("## 5. 异常与失败")
    lines.append("")
    non_ok = [r for r in results if r.get("status") != "ok"]
    if not non_ok:
        lines.append("- 本次 run 无失败（results.jsonl 全部为 ok）。")
        lines.append("")
    else:
        lines.append(f"- 失败/异常 job 数：{len(non_ok)}")
        lines.append("")
        sample = non_ok[: min(top_n, len(non_ok))]
        rows = []
        for r in sample:
            rows.append([
                r.get("key", "-"),
                r.get("status", "-"),
                (r.get("error") or "-")[:120],
                r.get("train_log", "-"),
                r.get("report_log", "-"),
            ])
        lines.append(_md_table(["key", "status", "error(head)", "train_log", "report_log"], rows))
        lines.append("")

    # 6. PIT（可选附录）
    if include_pit_appendix and "pit" in df.columns:
        lines.append("## 6. PIT（可选附录）")
        lines.append("")
        summary_mhp, _, _ = _group_summary(df, group_cols=["market", "label_horizon", "pit"], top_n=top_n)
        if summary_mhp.empty:
            lines.append("- 无 PIT 分组数据。")
            lines.append("")
        else:
            rows = []
            for r in summary_mhp.head(18).to_dict(orient="records"):
                rows.append([
                    r.get("market"),
                    r.get("label_horizon"),
                    r.get("pit"),
                    int(r.get("n", 0)),
                    _fmt_float(r.get("annret_median")),
                    _fmt_float(r.get("ir_median")),
                    _fmt_pct(r.get("ir_pos_rate")),
                    _fmt_float(r.get("maxdd_median")),
                ])
            lines.append(_md_table(
                ["market", "h", "pit", "n", "AnnRet_med", "IR_med", "IR>0", "MaxDD_med"],
                rows,
            ))
            lines.append("")
            lines.append("- 用法：只在你明确要回答“PIT 是否有增量”时，再基于同一 `(market,h,window)` 做对照。")
            lines.append("")

    # 7. 结论与下一步
    lines.append("## 7. 结论与下一步（基于本次 run）")
    lines.append("")
    if not summary_mh.empty:
        best3 = summary_mh.head(3).to_dict(orient="records")
        lines.append("- 粗结论（静态汇总）：")
        for r in best3:
            lines.append(
                f"  - {r.get('market')}/h={r.get('label_horizon')}: "
                f"IR_med={_fmt_float(r.get('ir_median'))}, "
                f"AnnRet_med={_fmt_float(r.get('annret_median'))}, "
                f"IR>0={_fmt_pct(r.get('ir_pos_rate'))}, "
                f"MaxDD_worst={_fmt_float(r.get('maxdd_worst'))}"
            )
        lines.append("")
        lines.append("- 风险/质量提示：")
        if "pred_instruments" in df.columns:
            low_cov_cnt = int((df["pred_instruments"] <= 50).sum())
            if low_cov_cnt > 0:
                lines.append(f"  - 发现 pred_instruments≤50 的窗口：{low_cov_cnt} 条（优先排查数据/停牌/覆盖率）。")
            else:
                lines.append("  - 未发现明显极低 coverage 的窗口（按 pred_instruments≤50 判断）。")
        lines.append("  - 建议把第 4 节里 IR 明显为负的年份视为 regime 风险，优先解释原因再做结论。")
        lines.append("")
        lines.append("- 下一步建议（最省时间的验证路径）：")
        lines.append("  - 先固定 `csi300/h=10` 做更细的窗口级检查（成本拖累、coverage、最差窗口回撤）。")
        lines.append("  - 对 `csiall`：若目标是选股 alpha，优先确认 universe/基准/成本口径无误；否则考虑更换特征/模型或分层建模。")
        lines.append("  - 若要评估 PIT 增量：只做同一 `(market,h,window)` 的 no_pit vs pit_all 对照，并同步检查 coverage 是否变化。")
        lines.append("")
    else:
        lines.append("- 分组汇总为空，暂无法给出结论。")
        lines.append("")

    content = "\n".join(lines).rstrip() + "\n"
    files.out_md.write_text(content, encoding="utf-8")
    return files.out_md


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze a grid run and generate grid_run_analysis.md")
    p.add_argument("--run-dir", required=True, help="e.g. reports/grid_runs/20260122_042207")
    p.add_argument("--out", default=None, help="Output markdown path (default: <run-dir>/grid_run_analysis.md)")
    p.add_argument("--top-n", type=int, default=10, help="Top N rows/groups to display (default: 10)")
    p.add_argument("--no-pit-appendix", action="store_true", help="Do not generate PIT appendix section")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_path = Path(args.out) if args.out else None
    out_md = analyze(
        Path(args.run_dir),
        out_md=out_path,
        top_n=max(1, int(args.top_n)),
        include_pit_appendix=not args.no_pit_appendix,
    )
    print(f"[ok] wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
