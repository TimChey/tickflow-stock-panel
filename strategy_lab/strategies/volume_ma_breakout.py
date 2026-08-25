"""放量均线突破 - MA20 上方 + 放量 + 收阳 + 动量正向

polars_expr 策略示例: 最简单的策略模式，只需 META + filter 函数。
filter 接收 enriched DataFrame 和参数，返回 Polars 过滤表达式。

enriched 常用列:
  symbol, date, open, high, low, close, volume, amount,
  change_pct, turnover_rate, amount,
  ma5, ma10, ma20, ma60,
  macd, macd_signal, macd_dea,
  kdj_k, kdj_d, kdj_j,
  rsi_6, rsi_14, rsi_24,
  boll_upper, boll_mid, boll_lower,
  momentum_5d, momentum_20d, momentum_60d,
  vol_ratio_5d, vol_ratio_10d,
  annual_vol_20d, amplitude,
  signal_limit_up, signal_broken_limit_up,
  consecutive_limit_ups, consecutive_limit_downs,
  name, total_shares, float_shares,
  raw_close, raw_high, raw_low
"""
from __future__ import annotations

import polars as pl

META = {
    "id": "lab_volume_ma_breakout",
    "name": "放量均线突破",
    "description": "MA20上方 + 量比≥2 + 收阳 + 5日动量>0",
    "tags": ["放量", "均线", "突破"],
    "asset_types": ["stock"],
    "timeframes": ["1d"],
    "params": [
        {
            "id": "vol_ratio_min",
            "label": "最低量比",
            "type": "float",
            "default": 2.0,
            "min": 1.0,
            "max": 10.0,
            "step": 0.1,
        },
        {
            "id": "require_bullish",
            "label": "要求收阳",
            "type": "bool",
            "default": True,
        },
        {
            "id": "require_above_ma60",
            "label": "要求在MA60上方",
            "type": "bool",
            "default": True,
        },
    ],
    "scoring": {"vol_ratio_5d": 0.3, "change_pct": 0.3, "momentum_20d": 0.4},
    "order_by": "score",
    "descending": True,
    "limit": 50,
}

# 基础过滤: 排除 ST、低价股、低流动性
BASIC_FILTER = {
    "price_min": 3,
    "price_max": 300,
    "market_cap_min": 20e8,
    "amount_min": 0.5e8,
    "exclude_st": True,
    "exclude_new_days": 30,
    "boards": ["沪主板", "深主板", "创业板", "科创板"],
}

ENTRY_SIGNALS = ["signal_volume_surge"]
EXIT_SIGNALS = ["signal_ma20_breakdown"]
STOP_LOSS = -0.06
MAX_HOLD_DAYS = 15
ALERTS = []


def filter(df: pl.DataFrame, params: dict) -> pl.Expr:
    """策略过滤: 返回 Polars 表达式，选中满足条件的行。

    条件:
      1. 收盘价在 MA20 上方 (趋势向上)
      2. 量比 ≥ vol_ratio_min (放量)
      3. 收阳 (close > open) -- 可选
      4. 收盘价在 MA60 上方 (中期趋势) -- 可选
      5. 5日动量 > 0 (近期上涨)
    """
    exprs: list[pl.Expr] = []

    # 趋势: 收盘价在 MA20 上方
    if "ma20" in df.columns:
        exprs.append(pl.col("close") > pl.col("ma20"))

    # 放量: 量比达标
    vol_min = float(params.get("vol_ratio_min", 2.0))
    if "vol_ratio_5d" in df.columns:
        exprs.append(pl.col("vol_ratio_5d") >= vol_min)

    # 收阳
    if params.get("require_bullish", True):
        exprs.append(pl.col("close") > pl.col("open"))

    # MA60 上方
    if params.get("require_above_ma60", True) and "ma60" in df.columns:
        exprs.append(pl.col("close") > pl.col("ma60"))

    # 5日动量为正
    if "momentum_5d" in df.columns:
        exprs.append(pl.col("momentum_5d") > 0)

    if not exprs:
        return pl.lit(False)
    return pl.all_horizontal(exprs)
