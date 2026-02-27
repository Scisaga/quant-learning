from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class QuantileSpec:
    """
    分位数配置：用于统一“我们到底预测了哪些分位点”。

    说明（人工验证用）：
    - Chronos-2 输出通常是多分位数（quantile forecasts）
    - 你可以只用 0.1/0.5/0.9 做区间，也可以用 0.01..0.99 做阈值概率插值
    """

    levels: Sequence[float]

    def validate(self) -> None:
        if not self.levels:
            raise ValueError("QuantileSpec.levels 不能为空")
        for q in self.levels:
            if not (0.0 < float(q) < 1.0):
                raise ValueError(f"分位数水平必须在 (0,1) 内，得到: {q}")

    @property
    def levels_sorted(self) -> list[float]:
        return sorted(float(x) for x in self.levels)


@dataclass(frozen=True)
class HorizonSpec:
    """
    预测 horizon（未来 N 个交易日）。

    注意：本文/代码统一使用“交易日步长”，不直接用自然日。
    """

    horizons: Sequence[int]

    def validate(self) -> None:
        if not self.horizons:
            raise ValueError("HorizonSpec.horizons 不能为空")
        for h in self.horizons:
            if int(h) <= 0:
                raise ValueError(f"horizon 必须为正整数，得到: {h}")

    @property
    def max_horizon(self) -> int:
        return max(int(x) for x in self.horizons)


@dataclass(frozen=True)
class EventThresholdSpec:
    """
    事件阈值（a）：以“简单收益比例”表达，例如 0.03 表示 3%。
    """

    thresholds: Sequence[float]

    def validate(self) -> None:
        if not self.thresholds:
            raise ValueError("EventThresholdSpec.thresholds 不能为空")
        for a in self.thresholds:
            a = float(a)
            if a <= 0:
                raise ValueError(f"阈值 a 必须 >0，得到: {a}")
            if a >= 1:
                raise ValueError(f"阈值 a 建议 <1（<100%），得到: {a}")


def ensure_unique_sorted_int(values: Iterable[int]) -> list[int]:
    return sorted({int(x) for x in values})


def ensure_unique_sorted_float(values: Iterable[float]) -> list[float]:
    return sorted({float(x) for x in values})

