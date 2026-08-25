"""连续急跌 - 放量大盘股连续5日跌幅超3%

策略逻辑:
  1. 近10个交易日，日平均成交额 > 2亿
  2. 近10个交易日，每日成交额都 > 1.5亿
  3. 连续5个交易日 (截至选股日的最近5日)，每日跌幅 < -3%
  4. 流通市值 > 100亿 (在 BASIC_FILTER 中检查)

执行后端: python_history_legacy (需要历史窗口)
"""
from __future__ import annotations

import polars as pl

META = {
    "id": "lab_consecutive_drop",
    "name": "连续急跌",
    "description": "近10日放量 + 连续5日每日跌幅超3% + 流通市值>100亿",
    "tags": ["急跌", "超跌", "放量", "大市值"],
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
            "id": "drop_min",
            "label": "单日跌幅下限(%)",
            "type": "float",
            "default": 3.0,
            "min": 2.0,
            "max": 10.0,
            "step": 0.5,
        },
        {
            "id": "consecutive_days",
            "label": "连续下跌天数",
            "type": "int",
            "default": 4,
            "min": 2,
            "max": 5,
            "step": 1,
        },
    ],
    # 累计跌幅取负值参与评分: 跌得越多, -drop 越大, 归一化后得分越高
    "scoring": {"_avg_amount_10d": 0.4, "_drop_neg": 0.4, "vol_ratio_5d": 0.2},
    "order_by": "score",
    "descending": True,
    "limit": 50,
}

EXECUTION_BACKEND = "python_history_legacy"

LOOKBACK_DAYS = 10

BASIC_FILTER = {
    "price_min": 3,
    "price_max": 500,
    "float_cap_min": 100e8,  # 流通市值 > 100亿
    "exclude_st": True,
    "exclude_new_days": 30,
    "boards": ["沪主板", "深主板", "创业板", "科创板"],
}

ENTRY_SIGNALS = []
EXIT_SIGNALS = []
STOP_LOSS = -0.05
MAX_HOLD_DAYS = 10
ALERTS = []


def filter_history(df: pl.DataFrame, params: dict) -> pl.DataFrame:
    """历史窗口过滤: 返回满足条件的股票 (含所有历史行，引擎再按 as_of 裁剪)。

    条件 1-3 在此检查；条件 4 (流通市值) 由 BASIC_FILTER 在 as_of 日检查。
    """
    df = df.sort(["symbol", "date"])

    # 从参数提取阈值 (用户输入为亿元/百分比，转为元/小数)
    avg_amt_min = float(params.get("avg_amount_min", 2.0)) * 1e8
    daily_amt_min = float(params.get("daily_amount_min", 1.5)) * 1e8
    drop_min = float(params.get("drop_min", 3.0)) / 100.0
    consec_days = int(params.get("consecutive_days", 5))

    # 按 symbol 分组，取近 N 个交易日的统计量
    stats = (
        df.group_by("symbol", maintain_order=True)
        .agg(
            # 条件 1: 近10日平均成交额
            pl.col("amount").tail(10).mean().alias("_avg_amount_10d"),
            # 条件 2: 近10日最小成交额 (每日都达标 = 最小值达标)
            pl.col("amount").tail(10).min().alias("_min_amount_10d"),
            # 条件 3: 最近连续N日每日跌幅都超阈值 (最大值达标 = 每日达标)
            pl.col("change_pct").tail(consec_days).max().alias("_max_change_nd"),
            # 评分用: 连续N日累计跌幅 (各日涨跌幅求和)
            pl.col("change_pct").tail(consec_days).sum().alias("_drop_nd"),
            # 数据完整性: 确保有足够交易日
            pl.col("date").count().alias("_day_count"),
        )
        .with_columns(
            # 评分用: 累计跌幅取负值 (跌得越多得分越高)
            (-pl.col("_drop_nd")).alias("_drop_neg"),
        )
        .filter(
            (pl.col("_day_count") >= 10)
            & (pl.col("_avg_amount_10d") > avg_amt_min)
            & (pl.col("_min_amount_10d") > daily_amt_min)
            & (pl.col("_max_change_nd") < -drop_min)
        )
    )

    # 返回通过筛选的股票的全部历史行 (引擎会按 as_of 裁剪到目标日)
    return df.join(stats, on="symbol", how="inner")
