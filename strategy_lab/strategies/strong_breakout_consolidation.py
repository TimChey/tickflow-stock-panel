"""强势突破后整理 - 放量上涨后近10日窄幅整理，大流通市值

策略逻辑:
  1. 近10个交易日，日平均成交额 > 2亿
  2. 近10个交易日，每日成交额都 > 1.5亿
  3. 近15个交易日，至少有一天涨幅 > 7%
  4. 近10个交易日，每日涨幅在 -6% 到 1.5% 之间 (整理)
  5. 流通市值 > 100亿 (在 BASIC_FILTER 中检查)

执行后端: python_history_legacy (需要历史窗口)
"""
from __future__ import annotations

import polars as pl

META = {
    "id": "lab_strong_breakout_consolidation",
    "name": "强势突破后整理",
    "description": "近15日有强势上涨 + 近10日放量窄幅整理 + 流通市值>100亿",
    "tags": ["强势", "突破", "整理", "放量", "大市值"],
    "asset_types": ["stock"],
    "timeframes": ["1d"],
    "params": [
        {
            "id": "avg_amount_min",
            "label": "10日平均成交额下限(亿)",
            "type": "float",
            "default": 2.0,
            "min": 0.5,
            "max": 20.0,
            "step": 0.5,
        },
        {
            "id": "daily_amount_min",
            "label": "10日每日成交额下限(亿)",
            "type": "float",
            "default": 1.5,
            "min": 0.3,
            "max": 10.0,
            "step": 0.1,
        },
        {
            "id": "max_change_min",
            "label": "15日最大涨幅下限(%)",
            "type": "float",
            "default": 7.0,
            "min": 3.0,
            "max": 20.0,
            "step": 0.5,
        },
        {
            "id": "consolidation_low",
            "label": "整理期涨幅下限(%)",
            "type": "float",
            "default": -6.0,
            "min": -10.0,
            "max": 0.0,
            "step": 0.5,
        },
        {
            "id": "consolidation_high",
            "label": "整理期涨幅上限(%)",
            "type": "float",
            "default": 2.0,
            "min": 0.0,
            "max": 5.0,
            "step": 0.5,
        },
    ],
    "scoring": {"_avg_amount_10d": 0.4, "_max_change_15d": 0.3, "momentum_20d": 0.3},
    "order_by": "score",
    "descending": True,
    "limit": 50,
}

LOOKBACK_DAYS = 15

BASIC_FILTER = {
    "price_min": 3,
    "price_max": 500,
    "float_cap_min": 100e8,  # 流通市值 > 100亿
    "exclude_st": True,
    "exclude_new_days": 30,
    "boards": ["沪主板", "深主板", "创业板", "科创板"],
}

ENTRY_SIGNALS = []
EXIT_SIGNALS = ["signal_ma20_breakdown"]
STOP_LOSS = -0.05
MAX_HOLD_DAYS = 15
ALERTS = []


def filter_history(df: pl.DataFrame, params: dict) -> pl.DataFrame:
    """历史窗口过滤: 返回满足条件的股票 (含所有历史行，引擎再按 as_of 裁剪)。

    条件 1-4 在此检查；条件 5 (流通市值) 由 BASIC_FILTER 在 as_of 日检查。
    """
    df = df.sort(["symbol", "date"])

    # 从参数提取阈值 (用户输入为百分比/亿元，转为小数/元)
    avg_amt_min = float(params.get("avg_amount_min", 2.0)) * 1e8
    daily_amt_min = float(params.get("daily_amount_min", 1.5)) * 1e8
    max_chg_min = float(params.get("max_change_min", 7.0)) / 100.0
    consol_low = float(params.get("consolidation_low", -6.0)) / 100.0
    consol_high = float(params.get("consolidation_high", 1.5)) / 100.0

    # 按 symbol 分组，取近 N 个交易日的统计量
    stats = (
        df.group_by("symbol", maintain_order=True)
        .agg(
            # 条件 1: 近10日平均成交额
            pl.col("amount").tail(10).mean().alias("_avg_amount_10d"),
            # 条件 2: 近10日最小成交额 (每日都达标 = 最小值达标)
            pl.col("amount").tail(10).min().alias("_min_amount_10d"),
            # 条件 3: 近15日最大涨幅 (至少一天达标 = 最大值达标)
            pl.col("change_pct").tail(15).max().alias("_max_change_15d"),
            # 条件 4: 近10日涨幅范围 (每日都在区间内 = min>=下限 且 max<=上限)
            pl.col("change_pct").tail(10).min().alias("_min_change_10d"),
            pl.col("change_pct").tail(10).max().alias("_max_change_10d"),
            # 数据完整性: 确保有足够交易日
            pl.col("date").count().alias("_day_count"),
        )
        .filter(
            (pl.col("_day_count") >= 15)
            & (pl.col("_avg_amount_10d") > avg_amt_min)
            & (pl.col("_min_amount_10d") > daily_amt_min)
            & (pl.col("_max_change_15d") > max_chg_min)
            & (pl.col("_min_change_10d") >= consol_low)
            & (pl.col("_max_change_10d") <= consol_high)
        )
    )

    # 返回通过筛选的股票的全部历史行 (引擎会按 as_of 裁剪到目标日)
    return df.join(stats, on="symbol", how="inner")
