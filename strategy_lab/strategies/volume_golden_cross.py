"""放量金叉 - 放量整理后MA5金叉MA20，站上MA60，大流通市值

策略逻辑:
  1. 近10个交易日，日平均成交额 > 2亿
  2. 近10个交易日，每日成交额都 > 1.5亿
  3. 近N个交易日 (默认3日) 内，MA5 上穿 MA20 (金叉)
  4. 股价 (收盘价) 站上 MA60
  5. 流通市值 > 100亿 (在 BASIC_FILTER 中检查)
  6. 剔除银行板块股票 (名称含"银行"或"商行")

执行后端: python_history_legacy (需要历史窗口)
"""
from __future__ import annotations

import polars as pl

META = {
    "id": "lab_volume_golden_cross",
    "name": "放量金叉",
    "description": "近10日放量 + 近3日MA5金叉MA20 + 收盘价站上MA60 + 流通市值>100亿 (剔除银行)",
    "tags": ["放量", "均线", "金叉", "大市值"],
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
            "id": "golden_cross_days",
            "label": "金叉回看天数",
            "type": "int",
            "default": 3,
            "min": 1,
            "max": 10,
            "step": 1,
        },
    ],
    "scoring": {"_avg_amount_10d": 0.4, "momentum_20d": 0.3, "vol_ratio_5d": 0.3},
    "order_by": "score",
    "descending": True,
    "limit": 50,
}

EXECUTION_BACKEND = "python_history_legacy"

LOOKBACK_DAYS = 10

# 声明 filter_history 依赖的信号列 (回测按此计算特征, 避免全量回退)
REQUIRED_FEATURES = {"signal_ma_golden_5_20"}

BASIC_FILTER = {
    "price_min": 3,
    "price_max": 500,
    "float_cap_min": 100e8,  # 流通市值 > 100亿
    "exclude_st": True,
    "exclude_new_days": 30,
    "boards": ["沪主板", "深主板", "创业板", "科创板"],
}

ENTRY_SIGNALS = ["signal_ma_golden_5_20"]
EXIT_SIGNALS = ["signal_ma_dead_5_20"]
STOP_LOSS = -0.06
MAX_HOLD_DAYS = 15
ALERTS = []


def filter_history(df: pl.DataFrame, params: dict) -> pl.DataFrame:
    """历史窗口过滤: 返回满足条件的股票 (含所有历史行，引擎再按 as_of 裁剪)。

    条件 1-4、6 在此检查；条件 5 (流通市值) 由 BASIC_FILTER 在 as_of 日检查。
    """
    df = df.sort(["symbol", "date"])

    # 条件 6: 剔除银行板块 (银行股简称均含"银行"或"商行", 如渝农商行; 回测面板缺 name 列时跳过)
    if "name" in df.columns:
        df = df.filter(~pl.col("name").str.contains("银行|商行"))

    # 从参数提取阈值 (用户输入为亿元，转为元)
    avg_amt_min = float(params.get("avg_amount_min", 2.0)) * 1e8
    daily_amt_min = float(params.get("daily_amount_min", 1.5)) * 1e8
    golden_days = int(params.get("golden_cross_days", 3))

    # 按 symbol 分组，取近 N 个交易日的统计量
    stats = (
        df.group_by("symbol", maintain_order=True)
        .agg(
            # 条件 1: 近10日平均成交额
            pl.col("amount").tail(10).mean().alias("_avg_amount_10d"),
            # 条件 2: 近10日最小成交额 (每日都达标 = 最小值达标)
            pl.col("amount").tail(10).min().alias("_min_amount_10d"),
            # 条件 3: 近N日内出现 MA5 上穿 MA20 (信号列为空值按 False 处理)
            pl.col("signal_ma_golden_5_20")
            .fill_null(False)
            .tail(golden_days)
            .max()
            .alias("_golden_cross"),
            # 条件 4: 最新一日收盘价站上 MA60 (ma60 为空时比较结果为空, 自动排除)
            (pl.col("close").last() > pl.col("ma60").last()).alias("_above_ma60"),
            # 数据完整性: 确保有足够交易日
            pl.col("date").count().alias("_day_count"),
        )
        .filter(
            (pl.col("_day_count") >= 10)
            & (pl.col("_avg_amount_10d") > avg_amt_min)
            & (pl.col("_min_amount_10d") > daily_amt_min)
            & pl.col("_golden_cross")
            & pl.col("_above_ma60")
        )
    )

    # 返回通过筛选的股票的全部历史行 (引擎会按 as_of 裁剪到目标日)
    return df.join(stats, on="symbol", how="inner")
