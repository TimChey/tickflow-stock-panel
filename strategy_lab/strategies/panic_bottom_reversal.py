"""底部恐慌下跌 - 放量大盘股恐慌性下跌 + 底部特征

策略逻辑:
  1. 近10个交易日，日平均成交额 > 2亿
  2. 近10个交易日，每日成交额都 > 1.5亿
  3. 近10个交易日，至少有一天跌幅 > 5% (恐慌性下跌)
  4. 近10日最低价接近20日最低价 (底部特征，差距 ≤ 5%)
  5. 流通市值 > 100亿 (在 BASIC_FILTER 中检查)

执行后端: python_history_legacy (需要历史窗口)
"""
from __future__ import annotations

import polars as pl

META = {
    "id": "lab_panic_bottom_reversal",
    "name": "底部恐慌下跌",
    "description": "近10日放量 + 恐慌性大跌 + 股价处于底部区域 + 流通市值>100亿",
    "tags": ["恐慌", "超跌", "底部", "放量", "大市值"],
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
            "id": "panic_drop_min",
            "label": "恐慌跌幅下限(%)",
            "type": "float",
            "default": 5.0,
            "min": 3.0,
            "max": 10.0,
            "step": 0.5,
        },
        {
            "id": "bottom_threshold",
            "label": "底部偏离度上限(%)",
            "type": "float",
            "default": 5.0,
            "min": 1.0,
            "max": 15.0,
            "step": 0.5,
        },
    ],
    "scoring": {"_avg_amount_10d": 0.3, "_panic_drop_neg": 0.4, "vol_ratio_5d": 0.3},
    "order_by": "score",
    "descending": True,
    "limit": 50,
}

EXECUTION_BACKEND = "python_history_legacy"

LOOKBACK_DAYS = 20

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
MAX_HOLD_DAYS = 10
ALERTS = []


def filter_history(df: pl.DataFrame, params: dict) -> pl.DataFrame:
    """历史窗口过滤: 返回满足条件的股票 (含所有历史行，引擎再按 as_of 裁剪)。

    条件 1-4 在此检查；条件 5 (流通市值) 由 BASIC_FILTER 在 as_of 日检查。
    """
    df = df.sort(["symbol", "date"])

    # 从参数提取阈值 (用户输入为百分比/亿元，转为小数/元)
    avg_amt_min = float(params.get("avg_amount_min", 2.0)) * 1e8
    daily_amt_min = float(params.get("daily_amount_min", 1.5)) * 1e8
    panic_drop_min = float(params.get("panic_drop_min", 5.0)) / 100.0
    bottom_threshold = float(params.get("bottom_threshold", 5.0)) / 100.0

    # 按 symbol 分组，取近 N 个交易日的统计量
    stats = (
        df.group_by("symbol", maintain_order=True)
        .agg(
            # 条件 1: 近10日平均成交额
            pl.col("amount").tail(10).mean().alias("_avg_amount_10d"),
            # 条件 2: 近10日最小成交额 (每日都达标 = 最小值达标)
            pl.col("amount").tail(10).min().alias("_min_amount_10d"),
            # 条件 3: 近10日最小涨幅 (跌幅最大的一天，至少有一天恐慌性大跌)
            pl.col("change_pct").tail(10).min().alias("_min_change_10d"),
            # 条件 4: 近10日最低价 vs 近20日最低价 (底部特征)
            pl.col("low").tail(10).min().alias("_low_10d"),
            pl.col("low").tail(20).min().alias("_low_20d"),
            # 评分用: 恐慌日跌幅 (取负值，跌得越多得分越高)
            pl.col("change_pct").tail(10).min().alias("_panic_drop_neg"),
            # 数据完整性: 确保有足够交易日
            pl.col("date").count().alias("_day_count"),
        )
        .with_columns(
            # 底部偏离度 = (近10日最低价 - 近20日最低价) / 近20日最低价
            # 偏离度越小 (越接近0)，说明股价越接近20日最低点
            ((pl.col("_low_10d") - pl.col("_low_20d")) / pl.col("_low_20d")).alias("_bottom_deviation"),
        )
        .filter(
            (pl.col("_day_count") >= 20)
            & (pl.col("_avg_amount_10d") > avg_amt_min)
            & (pl.col("_min_amount_10d") > daily_amt_min)
            # 条件 3: 近10日至少有一天恐慌性大跌 (最小涨幅 < -恐慌跌幅阈值)
            & (pl.col("_min_change_10d") < -panic_drop_min)
            # 条件 4: 近10日最低价接近20日最低价 (底部区域)
            & (pl.col("_bottom_deviation") <= bottom_threshold)
        )
    )

    # 返回通过筛选的股票的全部历史行 (引擎会按 as_of 裁剪到目标日)
    return df.join(stats, on="symbol", how="inner")