"""强势整理加速回调 - 强势上涨后窄幅整理, 近5日加速下跌

策略逻辑:
  1. 近10个交易日, 日平均成交额 > 1亿
  2. 近5个交易日, 出现加速下跌 (近5日累计涨幅为负, 且最近一日跌幅为近5日最深)
  3. 近15个交易日, 至少有一天涨幅 > 7%
  4. 近10个交易日, 每日涨幅在 -6% 到 1.5% 之间 (整理)
  5. 基础过滤: 排除 ST、新股、低价股 (在 BASIC_FILTER 中检查)

执行后端: python_history_legacy (需要历史窗口)
"""
from __future__ import annotations

import polars as pl

META = {
    "id": "lab_surge_accel_pullback",
    "name": "强势整理加速回调",
    "description": "近15日有单日大涨(>7%) + 近10日窄幅整理 + 近5日加速下跌 + 10日日均成交额>1亿",
    "tags": ["强势", "整理", "加速下跌", "回调", "放量"],
    "asset_types": ["stock"],
    "timeframes": ["1d"],
    "params": [
        {
            "id": "avg_amount_min",
            "label": "10日平均成交额下限(亿)",
            "type": "float",
            "default": 1.0,
            "min": 0.5,
            "max": 20.0,
            "step": 0.5,
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
            "default": 1.5,
            "min": 0.0,
            "max": 5.0,
            "step": 0.5,
        },
    ],
    "scoring": {
        "_avg_amount_10d": 0.3,
        "_max_change_15d": 0.3,
        "_drop_neg": 0.2,
        "momentum_20d": 0.2,
    },
    "order_by": "score",
    "descending": True,
    "limit": 50,
}

EXECUTION_BACKEND = "python_history_legacy"

LOOKBACK_DAYS = 15

# filter_history 动态依赖的公开字段
REQUIRED_FEATURES = {"amount", "change_pct"}

BASIC_FILTER = {
    "price_min": 3,
    "price_max": 500,
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
    """历史窗口过滤: 返回满足条件的股票 (含所有历史行, 引擎再按 as_of 裁剪)。

    条件 1-4 在此检查; BASIC_FILTER (价格/ST/新股/板块) 由引擎在 as_of 日检查。
    """
    df = df.sort(["symbol", "date"])

    # 从参数提取阈值 (用户输入为百分比/亿元, 转为小数/元)
    avg_amt_min = float(params.get("avg_amount_min", 1.0)) * 1e8
    max_chg_min = float(params.get("max_change_min", 7.0)) / 100.0
    consol_low = float(params.get("consolidation_low", -6.0)) / 100.0
    consol_high = float(params.get("consolidation_high", 1.5)) / 100.0

    # 按 symbol 分组, 取近 N 个交易日的统计量
    stats = (
        df.group_by("symbol", maintain_order=True)
        .agg(
            # 条件 1: 近10日平均成交额
            pl.col("amount").tail(10).mean().alias("_avg_amount_10d"),
            # 条件 2: 近5日累计涨幅 (加速下跌要求为负)
            pl.col("change_pct").tail(5).sum().alias("_sum_change_5d"),
            # 条件 2: 最近一日涨幅 vs 近5日最小涨幅 (最近一日为近5日最深 = 逐日走弱)
            pl.col("change_pct").tail(5).min().alias("_min_change_5d"),
            pl.col("change_pct").last().alias("_last_change"),
            # 条件 3: 近15日最大涨幅 (至少一天达标 = 最大值达标)
            pl.col("change_pct").tail(15).max().alias("_max_change_15d"),
            # 条件 4: 近10日涨幅范围 (每日都在区间内 = min>=下限 且 max<=上限)
            pl.col("change_pct").tail(10).min().alias("_min_change_10d"),
            pl.col("change_pct").tail(10).max().alias("_max_change_10d"),
            # 数据完整性: 确保有足够交易日
            pl.col("date").count().alias("_day_count"),
        )
        .with_columns(
            # 评分用: 近5日累计跌幅取负值 (跌得越多得分越高)
            (-pl.col("_sum_change_5d")).alias("_drop_neg"),
        )
        .filter(
            (pl.col("_day_count") >= 15)
            & (pl.col("_avg_amount_10d") > avg_amt_min)
            # 条件 2: 近5日累计下跌, 且最近一日跌幅为近5日最深
            & (pl.col("_sum_change_5d") < 0)
            & (pl.col("_last_change") <= pl.col("_min_change_5d"))
            & (pl.col("_max_change_15d") > max_chg_min)
            & (pl.col("_min_change_10d") >= consol_low)
            & (pl.col("_max_change_10d") <= consol_high)
        )
    )

    # 返回通过筛选的股票的全部历史行 (引擎会按 as_of 裁剪到目标日)
    return df.join(stats, on="symbol", how="inner")
