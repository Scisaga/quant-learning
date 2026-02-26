from __future__ import annotations

from typing import Dict, Optional

import copy
import numpy as np
import pandas as pd

from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
from qlib.backtest.position import Position
from qlib.contrib.strategy.signal_strategy import BaseSignalStrategy


class StabilizedHoldingStrategy(BaseSignalStrategy):
    """
    SHS: Stabilized Holding Strategy

    核心思想
    --------
    在标准 Top-K 选股的框架上，引入“时间稳定性”和“换手约束”，避免因日度噪声导致的频繁调仓：

    1. **Score 时间平滑（EMA）**  
       - 对原始预测分数做指数加权滑动平均（EMA），减少单日异常值对持仓的冲击。  
       - 新分数：`s_t_ema = α * s_t_raw + (1-α) * s_{t-1}_ema`，其中 `α = ema_alpha`。

    2. **Top-K 双阈值 buffer（K_in / K_out）**  
       - `buy_buffer`（K_in）：只允许排名进入前 K_in 的新股票作为买入候选。  
       - `sell_buffer`（K_out）：只有当已持仓股票的排名跌出 K_out 之后才考虑卖出。  
       - 中间的 `K_in < rank ≤ K_out` 区间相当于“缓冲带”，可以显著降低换手率。

    3. **最小持有期约束（min_holding_days）**  
       - 每只个股从建仓开始计数，在未达到 `min_holding_days` 之前，即便排名恶化也不允许卖出。  
       - 通过时间约束进一步抑制“追涨杀跌”式高频换仓。

    4. **基于“信号恶化”的卖出，而非瞬时排名比较**  
       - 卖出只在“两条件同时满足”时触发：  
         (a) 持有天数 ≥ `min_holding_days`；(b) 当前 EMA 排名 > `sell_buffer`。  
       - 与按单期 Top-K 直接截断不同，更关注“持续的排名恶化”，从而提高持仓稳定性。

    5. **等权 + 风险度控制（risk_degree）**  
       - 在目标持仓集合确定后，对所有持仓股票按等权分配资金，总仓位由 `risk_degree` 控制。  
       - 与 Qlib 内置的信号型策略一致，`risk_degree` 表示“投资资产/总资产”的比例。

    适用场景
    --------
    - 追求 **中等频率、相对平滑** 的调仓节奏，而非每日剧烈换仓的 alpha 策略。  
    - 希望在保持 Top-K 选股能力的同时，显著降低换手率、提高因子/模型在组合层面的可用性。
    """

    def __init__(
        self,
        *,
        signal,  # 预测信号源（如 Qlib 的 ExpressionDAG），产出 alpha 分数
        topk: int = 50,  # 目标持仓数量
        buy_buffer: int = 30,  # 买入阈值 K_in：仅排名 ≤ K_in 的股票可买入
        sell_buffer: int = 70,  # 卖出阈值 K_out：排名 > K_out 且满足最小持有期后才卖出
        min_holding_days: int = 7,  # 最小持有天数，未达到前禁止卖出
        ema_alpha: float = 0.3,  # EMA 平滑系数 α，越大越依赖近期分数
        risk_degree: float = 0.95,  # 投资资产占总资产比例（0~1）
        only_tradable: bool = True,  # 是否过滤不可交易标的
        forbid_all_trade_at_limit: bool = True,  # 涨跌停时是否禁止买卖
        **kwargs,
    ):
        # 使用 BaseSignalStrategy 统一管理 signal / risk_degree / 基础回测基础设施。
        super().__init__(signal=signal, risk_degree=risk_degree, **kwargs)

        assert buy_buffer <= topk <= sell_buffer

        self.topk = int(topk)
        self.buy_buffer = int(buy_buffer)
        self.sell_buffer = int(sell_buffer)
        self.min_holding_days = int(min_holding_days)
        self.ema_alpha = float(ema_alpha)
        self.only_tradable = bool(only_tradable)
        self.forbid_all_trade_at_limit = bool(forbid_all_trade_at_limit)

        # --- 内部状态 ---
        self._ema_score: Optional[pd.Series] = None
        self._holding_days: Dict[str, int] = {}

    # ==========================================================
    # 1. Score 平滑（EMA）
    # ==========================================================
    def _smooth_score(self, score: pd.Series) -> pd.Series:
        if self._ema_score is None:
            self._ema_score = score.copy()
        else:
            self._ema_score = (
                self.ema_alpha * score
                + (1 - self.ema_alpha) * self._ema_score.reindex(score.index).fillna(0.0)
            )
        return self._ema_score

    # ==========================================================
    # 2. 核心交易逻辑
    # ==========================================================
    def generate_trade_decision(self, execute_result=None):
        # 与 TopkDropoutStrategy 一样：按“上一根 bar”的信号生成当期交易决策。
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)

        # 当前步使用的原始分数（上一 bar 的预测），并做去 NaN 处理。
        score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        if isinstance(score, pd.DataFrame):
            score = score.iloc[:, 0]
        if score is None:
            return TradeDecisionWO([], self)
        score = score.dropna()
        if score.empty:
            return TradeDecisionWO([], self)

        # 1) EMA 平滑分数
        score = self._smooth_score(score)
        # 2) 分数降序排名（越大越靠前）
        rank = score.rank(ascending=False, method="first")

        # 当前持仓（复制一份 Position，用于在本函数内模拟卖出/买入对现金的影响）
        current_temp: Position = copy.deepcopy(self.trade_position)
        cash = current_temp.get_cash()
        current_stock_list = current_temp.get_stock_list()
        current_holdings = set(current_stock_list)

        sell_list = []

        # ======================================================
        # 卖出逻辑：最小持有期 + 跌出 sell_buffer
        # ======================================================
        for code in current_stock_list:
            r = rank.get(code, np.inf)
            holding_days = self._holding_days.get(code, 0)

            # 未超过最小持有期：禁止卖出
            if holding_days < self.min_holding_days:
                continue

            # 跌破 sell_buffer：允许卖出
            if r > self.sell_buffer:
                sell_list.append(code)

        # 卖出之后的持仓数量
        remaining_after_sell = len(current_holdings) - len(sell_list)
        # 为了不超过 topk，最多还能买入多少只
        max_new = max(0, self.topk - remaining_after_sell)

        # ======================================================
        # 买入逻辑：排名进入 buy_buffer 的新股票，且不超过 max_new
        # ======================================================
        candidates = rank[rank <= self.buy_buffer].sort_values(ascending=False).index
        buy_list = []
        for code in candidates:
            if code in current_holdings:
                continue
            if len(buy_list) >= max_new:
                break
            buy_list.append(code)

        # ======================================================
        # 生成具体订单（参考 TopkDropoutStrategy）
        # ======================================================
        sell_order_list = []
        buy_order_list = []

        # 先卖出
        for code in current_stock_list:
            if code not in sell_list:
                continue

            # 与 TopkDropoutStrategy 保持一致：only_tradable 为 True 时才过滤不可交易标的。
            if self.only_tradable and not self.trade_exchange.is_stock_tradable(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=None if self.forbid_all_trade_at_limit else OrderDir.SELL,
            ):
                continue

            sell_amount = current_temp.get_stock_amount(code=code)
            sell_order = Order(
                stock_id=code,
                amount=sell_amount,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=Order.SELL,
            )
            if self.trade_exchange.check_order(sell_order):
                sell_order_list.append(sell_order)
                trade_val, trade_cost, trade_price = self.trade_exchange.deal_order(
                    sell_order, position=current_temp
                )
                cash += trade_val - trade_cost

        # 再买入：对 buy_list 等权分配可用资金 * risk_degree
        value_per_stock = cash * self.risk_degree / len(buy_list) if buy_list else 0.0

        for code in buy_list:
            if self.only_tradable and not self.trade_exchange.is_stock_tradable(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=None if self.forbid_all_trade_at_limit else OrderDir.BUY,
            ):
                continue

            buy_price = self.trade_exchange.get_deal_price(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=OrderDir.BUY,
            )
            if not np.isfinite(buy_price) or buy_price <= 0:
                continue

            buy_amount = value_per_stock / buy_price
            factor = self.trade_exchange.get_factor(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
            )
            buy_amount = self.trade_exchange.round_amount_by_trade_unit(buy_amount, factor)
            if buy_amount <= 0:
                continue

            buy_order = Order(
                stock_id=code,
                amount=buy_amount,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=Order.BUY,
            )
            buy_order_list.append(buy_order)

        # ======================================================
        # 更新 holding days：基于“理论上的最终持仓集合”
        # ======================================================
        final_holdings = (current_holdings - set(sell_list)) | set(buy_list)
        new_holding_days: Dict[str, int] = {}
        for code in final_holdings:
            if code in current_holdings:
                new_holding_days[code] = self._holding_days.get(code, 0) + 1
            else:
                new_holding_days[code] = 1
        self._holding_days = new_holding_days

        return TradeDecisionWO(sell_order_list + buy_order_list, self)
