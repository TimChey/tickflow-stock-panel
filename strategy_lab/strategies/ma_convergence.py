"""放量均线粘合 - MA5/10/20/30粘合 + 站上MA60，大流通市值

策略逻辑:
  1. 近10个交易日，日平均成交额 > 2亿
  2. 近10个交易日，每日成交额都 > 1.5亿
  3. MA5、MA10、MA20、MA30 均线粘合 (极差/最小值 ≤ 3%，可调)
  4. 股价 (收盘价) 站上 MA60
  5. 流通市值 > 100亿 (在 BASIC_FILTER 中检查)

执行后端: python_history_legacy (需要历史窗口)
"""
from __future__ import annotations

import polars as pl

META = {
    "id": "lab_ma_convergence",
    "name": "放量均线粘合",
    "description": "近10日放量 + MA5/10/20/30粘合(极差≤3%) + 收盘价站上MA60 + 流通市值>100亿",
    "tags": ["放量", "均线", "粘合", "大市值"],
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
            "id": "ma_range_max",
            "label": "均线粘合极差上限(%)",
            "type": "float",
            "default": 3.0,
            "min": 1.0,
            "max": 10.0,
            "step": 0.5,
        },
    ],
    # 粘合度取负值参与评分: 极差越小, -range 越大, 归一化后得分越高
    "scoring": {"_avg_amount_10d": 0.4, "_ma_range_neg": 0.4, "vol_ratio_5d": 0.2},
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
EXIT_SIGNALS = ["signal_ma20_breakdown"]
STOP_LOSS = -0.05
MAX_HOLD_DAYS = 20
ALERTS = []

_MA_COLS = ("ma5", "ma10", "ma20", "ma30")


def filter_history(df: pl.DataFrame, params: dict) -> pl.DataFrame:
    """历史窗口过滤: 返回满足条件的股票 (含所有历史行，引擎再按 as_of 裁剪)。

    条件 1-4 在此检查；条件 5 (流通市值) 由 BASIC_FILTER 在 as_of 日检查。
    """
    df = df.sort(["symbol", "date"])

    # 从参数提取阈值 (用户输入为亿元/百分比，转为元/小数)
    avg_amt_min = float(params.get("avg_amount_min", 2.0)) * 1e8
    daily_amt_min = float(params.get("daily_amount_min", 1.5)) * 1e8
    ma_range_max = float(params.get("ma_range_max", 3.0)) / 100.0

    # 最新一日四条均线的最大/最小值 (任一为空则粘合不成立, fail-closed)
    ma_last = [pl.col(c).last() for c in _MA_COLS]
    ma_max = pl.max_horizontal(*ma_last)
    ma_min = pl.min_horizontal(*ma_last)
    ma_valid = pl.all_horizontal([pl.col(c).last().is_not_null() for c in _MA_COLS])

    # 按 symbol 分组，取近 N 个交易日的统计量
    stats = (
        df.group_by("symbol", maintain_order=True)
        .agg(
            # 条件 1: 近10日平均成交额
            pl.col("amount").tail(10).mean().alias("_avg_amount_10d"),
            # 条件 2: 近10日最小成交额 (每日都达标 = 最小值达标)
            pl.col("amount").tail(10).min().alias("_min_amount_10d"),
            # 条件 3: 均线粘合度 = (最大-最小)/最小, 空值时置空被过滤
            pl.when(ma_valid)
            .then((ma_max - ma_min) / ma_min)
            .otherwise(None)
            .alias("_ma_range"),
            # 条件 4: 最新一日收盘价站上 MA60 (ma60 为空时比较结果为空, 自动排除)
            (pl.col("close").last() > pl.col("ma60").last()).alias("_above_ma60"),
            # 数据完整性: 确保有足够交易日
            pl.col("date").count().alias("_day_count"),
        )
        .with_columns(
            # 评分用: 粘合度取负值 (极差越小得分越高)
            (-pl.col("_ma_range")).alias("_ma_range_neg"),
        )
        .filter(
            (pl.col("_day_count") >= 10)
            & (pl.col("_avg_amount_10d") > avg_amt_min)
            & (pl.col("_min_amount_10d") > daily_amt_min)
            & (pl.col("_ma_range") <= ma_range_max)
            & pl.col("_above_ma60")
        )
    )

    # 返回通过筛选的股票的全部历史行 (引擎会按 as_of 裁剪到目标日)
    return df.join(stats, on="symbol", how="inner")
