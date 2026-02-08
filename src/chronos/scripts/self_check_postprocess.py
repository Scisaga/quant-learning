"""
纯数学/逻辑自检脚本（不依赖 Chronos-2 模型、不依赖 Qlib）。

用途：
- 人工验证：分位数 -> 阈值概率 -> score -> conformal 校准 -> 概率校准
- 你可以直接运行，看打印结果是否符合直觉

运行：
  python src/chronos/scripts/self_check_postprocess.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 让 `src/chronos/qchronos` 可被直接 import（不要求安装成 site-packages）
_ROOT = Path(__file__).resolve().parents[1]  # .../src/chronos
sys.path.insert(0, str(_ROOT))

from qchronos.calibration import (
    ProbabilityCalibrator,
    brier_score,
    conformal_nonconformity,
    reliability_table,
    rolling_conformal_q,
)
from qchronos.postprocess import (
    derive_interval_and_scores,
    event_prob_from_quantiles,
    log_return_down_threshold_from_simple_return,
    log_return_threshold_from_simple_return,
)


def _demo_event_prob() -> None:
    print("\n=== [1] 分位数网格 -> 阈值概率（直觉：阈值越高，上涨概率越低） ===")
    quantile_levels = np.linspace(0.01, 0.99, 99)

    # 构造一个“类似正态”的分布分位数（只是为了演示）
    # 均值 0.01，标准差 0.02 的 log-return
    mu, sigma = 0.01, 0.02
    from scipy.stats import norm

    q_values = norm.ppf(quantile_levels, loc=mu, scale=sigma)

    for a in [0.01, 0.03, 0.05]:
        u = log_return_threshold_from_simple_return(a)
        p_up = event_prob_from_quantiles(
            quantile_levels=quantile_levels,
            quantile_values=q_values,
            threshold=u,
            side="up",
        )
        print(f"a={a:.2%} => u=log(1+a)={u:.4f} => p_up≈{p_up:.3f}")


def _demo_scores() -> None:
    print("\n=== [2] q10/q50/q90 -> interval_width + score（直觉：width 越大，score2 越小） ===")
    df = pd.DataFrame(
        {
            "q10": [-0.05, -0.02, -0.01],
            "q50": [0.02, 0.02, 0.02],
            "q90": [0.10, 0.05, 0.03],
            "p_up": [0.7, 0.7, 0.7],
            "p_down": [0.1, 0.2, 0.3],
        },
        index=["A", "B", "C"],
    )
    out = derive_interval_and_scores(df, q10_col="q10", q50_col="q50", q90_col="q90", p_up_col="p_up", p_down_col="p_down")
    print(out[["interval_width", "score1_med", "score2_med_over_width", "score3_pdiff"]])


def _demo_conformal() -> None:
    print("\n=== [3] Conformal 区间校准（直觉：校准后覆盖率更接近目标） ===")
    rng = np.random.default_rng(0)
    n = 200

    # 构造真实值：均值 0，方差随时间变大（模拟漂移）
    y = rng.normal(0.0, 0.02, size=n) + np.linspace(0, 1, n) * rng.normal(0.0, 0.02, size=n)

    # 构造“偏窄”的 raw 区间：固定半宽 0.02（后期明显不够）
    lower_raw = y - 0.02
    upper_raw = y + 0.02

    # 真实覆盖：应当接近 1.0（因为 raw 用 y 构造），这里为了演示改一下：加入噪声，让 y_true 偏离区间
    y_true = y + rng.normal(0.0, 0.03, size=n)

    scores = conformal_nonconformity(y_true, lower_raw, upper_raw)
    alpha = 0.2  # 目标覆盖 80%
    q = rolling_conformal_q(scores, alpha=alpha, window=60, min_count=30)
    lower_cal = lower_raw - q
    upper_cal = upper_raw + q

    raw_covered = np.mean((y_true >= lower_raw) & (y_true <= upper_raw))
    cal_covered = np.nanmean((y_true >= lower_cal) & (y_true <= upper_cal))

    print(f"raw 覆盖率≈{raw_covered:.3f}（通常偏离目标）")
    print(f"cal 覆盖率≈{cal_covered:.3f}（应更接近 0.8；前期 NaN 不计入）")
    print("最后 5 天的 q（扩张量）示例：", np.round(q[-5:], 4))


def _demo_prob_calibration() -> None:
    print("\n=== [4] 概率校准（直觉：cal 后 Brier 更小/可靠性更好） ===")
    rng = np.random.default_rng(1)
    n = 1000

    # 构造“过度自信”的 raw 概率：靠近 0/1
    p_raw = np.clip(rng.beta(0.5, 0.5, size=n), 1e-6, 1 - 1e-6)

    # 构造真实事件：真实发生率其实更接近 0.3 * p_raw + 0.2（故 raw 不可靠）
    p_true = np.clip(0.3 * p_raw + 0.2, 0.0, 1.0)
    y = (rng.uniform(0, 1, size=n) < p_true).astype(int)

    # 切分 calib / test
    calib = slice(0, 600)
    test = slice(600, None)

    cal = ProbabilityCalibrator(method="isotonic").fit(p_raw[calib], y[calib])
    p_cal = cal.predict(p_raw[test])

    bs_raw = brier_score(p_raw[test], y[test])
    bs_cal = brier_score(p_cal, y[test])
    print(f"Brier raw={bs_raw:.4f}  cal={bs_cal:.4f}（cal 通常更小）")

    tbl_raw = reliability_table(p_raw[test], y[test], n_bins=10)
    tbl_cal = reliability_table(p_cal, y[test], n_bins=10)
    print("\nraw reliability（前 5 行）：")
    print(tbl_raw.head())
    print("\ncal reliability（前 5 行）：")
    print(tbl_cal.head())


def main() -> None:
    _demo_event_prob()
    _demo_scores()
    _demo_conformal()
    _demo_prob_calibration()


if __name__ == "__main__":
    main()
