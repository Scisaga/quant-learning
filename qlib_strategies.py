from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from qlib.contrib.strategy import TopkDropoutStrategy


class VolatilityControlledTopkDropoutStrategy(TopkDropoutStrategy):
    """
    基于 `TopkDropoutStrategy` 的“市场波动率控仓位”策略。

    保持 Topk-Drop 的选股/换仓逻辑不变，仅在每个交易步动态调整仓位比例 `risk_degree`：

        risk_degree_t = clip(base_risk_degree * target_vol / market_vol_t, min_risk_degree, max_risk_degree)

    说明
    ----
    - `market_vol_t`：用市场指数（或你指定的标的）收益率的滚动标准差估计（收盘到收盘），并用 `annual_factor` 年化（默认 252）。
    - 为避免“偷看未来”，波动率默认在上一个 bar 结束时刻计算（`vol_shift=1`）。
    - `max_risk_degree` 默认等于初始化时的 `risk_degree`（不加杠杆，只减仓）。
    """

    def __init__(
        self,
        *,
        topk: int,
        n_drop: int,
        market: str = "SH000300",
        vol_window: int = 20,
        target_vol: float = 0.20,
        min_risk_degree: float = 0.10,
        max_risk_degree: Optional[float] = None,
        annual_factor: float = 252.0,
        vol_freq: str = "day",
        price_field: str = "$close",
        vol_shift: int = 1,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        topk / n_drop :
            与 `TopkDropoutStrategy` 一致：持仓数与每期换出数。
        market :
            用哪个“市场/指数”来计算波动率（例如：`SH000300`）。
        vol_window :
            波动率滚动窗口（例如 20 个交易日）。
        target_vol :
            目标年化波动率（例如 0.20 表示 20%）。波动率越高则仓位越低。
        min_risk_degree / max_risk_degree :
            风控后的仓位上下限（都是占总资产的比例）。
        annual_factor :
            年化因子，日频常用 252；如果是分钟级可自行调整。
        vol_freq / price_field :
            拉取用于计算波动率的数据频率与价格字段。
        vol_shift :
            用于计算波动率的“滞后步数”，默认 1 以避免未来函数。
        kwargs :
            透传给 `TopkDropoutStrategy`（例如 `signal`、`risk_degree`、`only_tradable` 等）。
        """
        super().__init__(topk=topk, n_drop=n_drop, **kwargs)

        self.market = market
        self.vol_window = int(vol_window)
        self.target_vol = float(target_vol)
        self.min_risk_degree = float(min_risk_degree)
        self.max_risk_degree = float(self.risk_degree if max_risk_degree is None else max_risk_degree)
        self.annual_factor = float(annual_factor)
        self.vol_freq = str(vol_freq)
        self.price_field = str(price_field)
        self.vol_shift = int(vol_shift)

        # 记录“基础仓位”，后续只做缩放（不会改变用户最初配置的 risk_degree 语义）
        self._base_risk_degree = float(self.risk_degree)
        self._market_vol: Optional[pd.Series] = None
        self._market_vol_range: Optional[tuple[pd.Timestamp, pd.Timestamp]] = None

    def _ensure_market_vol(self) -> None:
        # 延迟加载：只有在回测/执行开始后（拿到 trade_calendar 的时间范围）才计算并缓存波动率序列。
        if getattr(self, "trade_calendar", None) is None:
            return

        start_time, end_time = self.trade_calendar.get_all_time()
        if start_time is None or end_time is None:
            return

        cached_range = self._market_vol_range
        # 同一段回测区间内重复调用时直接复用缓存，避免每个交易步都去取数/计算。
        if (
            cached_range is not None
            and cached_range[0] == pd.Timestamp(start_time)
            and cached_range[1] == pd.Timestamp(end_time)
        ):
            return

        # 为了让滚动窗口在回测起点也能尽快“热身”，向前多取一段历史。
        fetch_start = pd.Timestamp(start_time) - pd.Timedelta(days=max(30, self.vol_window * 3))
        fetch_end = pd.Timestamp(end_time)

        try:
            from qlib.data import D

            df = D.features([self.market], [self.price_field], fetch_start, fetch_end, freq=self.vol_freq)
        except Exception:
            # 数据缺失/初始化问题等：退化为不控仓位（直接用 base_risk_degree）。
            self._market_vol = None
            self._market_vol_range = (pd.Timestamp(start_time), pd.Timestamp(end_time))
            return

        if df is None or len(df) == 0:
            self._market_vol = None
            self._market_vol_range = (pd.Timestamp(start_time), pd.Timestamp(end_time))
            return

        col = self.price_field if self.price_field in df.columns else df.columns[0]
        s = df[col]
        try:
            # D.features 返回 MultiIndex(datetime, instrument)，这里取出 market 对应的时间序列。
            close = s.xs(self.market, level="instrument").sort_index()
        except Exception:
            self._market_vol = None
            self._market_vol_range = (pd.Timestamp(start_time), pd.Timestamp(end_time))
            return

        # 用收盘到收盘的简单收益率估计 realized volatility，并做年化。
        ret = close.pct_change()
        vol = ret.rolling(self.vol_window, min_periods=self.vol_window).std() * np.sqrt(self.annual_factor)

        self._market_vol = vol
        self._market_vol_range = (pd.Timestamp(start_time), pd.Timestamp(end_time))

    def get_risk_degree(self, trade_step=None):
        # 返回当前交易步的“动态 risk_degree”。
        base = self._base_risk_degree
        if self.target_vol <= 0:
            return base

        try:
            if trade_step is None:
                trade_step = self.trade_calendar.get_trade_step()
            # 默认 shift=1：用“上一根 bar”的结束时刻来对齐波动率，避免使用当期尚未结束的数据。
            _, eval_time = self.trade_calendar.get_step_time(trade_step, shift=self.vol_shift)
            eval_time = pd.Timestamp(eval_time)
        except Exception:
            return base

        self._ensure_market_vol()
        if self._market_vol is None or self._market_vol.empty:
            return base

        vol_slice = self._market_vol.loc[:eval_time].dropna()
        if vol_slice.empty:
            return base

        cur_vol = float(vol_slice.iloc[-1])
        if not np.isfinite(cur_vol) or cur_vol <= 0:
            return base

        # 波动率越高仓位越低；波动率越低仓位越高（但不超过 max_risk_degree）。
        scaled = base * (self.target_vol / cur_vol)
        return float(np.clip(scaled, self.min_risk_degree, self.max_risk_degree))

    def generate_trade_decision(self, execute_result=None):
        # TopkDropoutStrategy 内部会直接用 self.risk_degree 分配买入资金；这里临时覆盖后再调用父类逻辑。
        original = float(self.risk_degree)
        self.risk_degree = self.get_risk_degree()
        try:
            return super().generate_trade_decision(execute_result=execute_result)
        finally:
            self.risk_degree = original
