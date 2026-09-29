"""放量双峰缩量企稳低吸 - 放量冲高见量价双峰后缩量回调, 待末日止跌企稳选入

参考结构: 阿为特(920693.BJ) / 利尔达(920249.BJ) 2026-08-26~09-10 十二个交易日:
近8日内放量冲高见量价双峰 (量峰=日均2.3倍), 此后缩量阴跌、回撤约12%、收盘跌破MA20,
近期5日逐日走弱加速。本策略在其企稳当天选入 (低吸观察), 故回看窗口取15日、
峰窗取10日, 为企稳确认留出时间。

策略逻辑:
  1. 近15个交易日, 日平均成交额 > 0.15亿
  2. 近15日最大成交额与最高价均出现在近10个交易日内 (近期放量冲高见双峰)
  3. 量峰成交额 >= 15日日均成交额 x 2 (显著放量)
  4. 收盘价较15日最高价回撤 >= 8%
  5. 近5个交易日累计涨幅 < 0 (回调中)
  6. 最新成交额 <= 量峰的 50% (回落中明显缩量)
  7. 收盘价 < MA20 (破位)
  8. 企稳确认: 最近一日涨幅 > 前一日涨幅 (跌幅收窄或翻红)
  9. 基础过滤: 排除 ST、新股、低价股; 含北交所 (在 BASIC_FILTER 中检查)

执行后端: python_history_legacy (需要历史窗口)
"""
from __future__ import annotations

import polars as pl

META = {
    "id": "lab_volume_peak_pullback",
    "name": "放量双峰缩量企稳低吸",
    "description": "放量冲高见量价双峰后缩量回调(回撤>=8%,破MA20), 末日止跌企稳时低吸",
    "tags": ["放量", "冲高回落", "缩量", "企稳", "低吸", "北交所"],
    "asset_types": ["stock"],
    "timeframes": ["1d"],
    "params": [
        {
            "id": "avg_amount_min",
            "label": "15日平均成交额下限(亿)",
            "type": "float",
            "default": 0.15,
            "min": 0.05,
            "max": 5.0,
            "step": 0.05,
        },
        {
            "id": "peak_vol_ratio",
            "label": "量峰/日均成交额倍数下限",
            "type": "float",
            "default": 2.0,
            "min": 1.5,
            "max": 5.0,
            "step": 0.25,
        },
        {
            "id": "shrink_ratio",
            "label": "最新成交额/量峰上限",
            "type": "float",
            "default": 0.5,
            "min": 0.2,
            "max": 0.9,
            "step": 0.05,
        },
        {
            "id": "drawdown_min",
            "label": "距15日最高回撤下限(%)",
            "type": "float",
            "default": 8.0,
            "min": 3.0,
            "max": 25.0,
            "step": 0.5,
        },
    ],
    "scoring": {
        "_shrink_pct": 0.3,
        "_drawdown_pct": 0.3,
        "_avg_amount_15": 0.2,
        "momentum_20d": 0.2,
    },
    "order_by": "score",
    "descending": True,
    "limit": 50,
}

EXECUTION_BACKEND = "python_history_legacy"

LOOKBACK_DAYS = 15

# filter_history 动态依赖的公开字段 (high/close/amount/change_pct 为基础行情列)
REQUIRED_FEATURES = {"ma20"}

BASIC_FILTER = {
    "price_min": 3,
    "price_max": 500,
    # 小盘结构: 取消默认的总市值/流通市值/当日成交额下限, 流动性由条件1控制
    "market_cap_min": None,
    "float_cap_min": None,
    "amount_min": None,
    "exclude_st": True,
    "exclude_new_days": 30,
    "boards": ["沪主板", "深主板", "创业板", "科创板", "北交所"],
}

ENTRY_SIGNALS = []
EXIT_SIGNALS = []
STOP_LOSS = -0.05
MAX_HOLD_DAYS = 10
ALERTS = []


def filter_history(df: pl.DataFrame, params: dict) -> pl.DataFrame:
    """历史窗口过滤: 返回满足条件的股票 (含所有历史行, 引擎再按 as_of 裁剪)。

    条件 1-8 在此检查; BASIC_FILTER (价格/ST/新股/板块) 由引擎在 as_of 日检查。
    """
    df = df.sort(["symbol", "date"])

    # 从参数提取阈值 (用户输入为百分比/亿元, 转为小数/元)
    avg_amt_min = float(params.get("avg_amount_min", 0.15)) * 1e8
    peak_vol_ratio = float(params.get("peak_vol_ratio", 2.0))
    shrink_ratio = float(params.get("shrink_ratio", 0.5))
    drawdown_min = float(params.get("drawdown_min", 8.0)) / 100.0

    # 按 symbol 分组, 取近 N 个交易日的统计量
    stats = (
        df.group_by("symbol", maintain_order=True)
        .agg(
            # 条件 1: 近15日平均成交额
            pl.col("amount").tail(15).mean().alias("_avg_amount_15"),
            # 条件 2: 量峰/价峰位置 (近10日最大值 == 近15日最大值 = 双峰在近10日内)
            pl.col("amount").tail(15).max().alias("_peak_amount"),
            pl.col("amount").tail(10).max().alias("_peak_amount_10d"),
            pl.col("high").tail(15).max().alias("_peak_high"),
            pl.col("high").tail(10).max().alias("_peak_high_10d"),
            # 条件 6/7/8: 最新收盘、最新成交额、MA20
            pl.col("close").last().alias("_last_close"),
            pl.col("amount").last().alias("_last_amount"),
            pl.col("ma20").last().alias("_last_ma20"),
            # 条件 5: 近5日累计涨幅
            pl.col("change_pct").tail(5).sum().alias("_sum_change_5d"),
            # 条件 8: 企稳 = 最近一日涨幅 > 前一日涨幅 (跌幅收窄或翻红)
            pl.col("change_pct").last().alias("_last_change"),
            pl.col("change_pct").tail(2).first().alias("_prev_change"),
            # 数据完整性: 确保有足够交易日
            pl.col("date").count().alias("_day_count"),
        )
        .with_columns(
            # 评分用: 回撤深度与缩量程度 (越大越充分)
            (1 - pl.col("_last_close") / pl.col("_peak_high")).alias("_drawdown_pct"),
            (1 - pl.col("_last_amount") / pl.col("_peak_amount")).alias("_shrink_pct"),
        )
        .filter(
            (pl.col("_day_count") >= 15)
            & (pl.col("_avg_amount_15") > avg_amt_min)
            # 条件 2: 量峰与价峰均出现在近10个交易日内
            & (pl.col("_peak_amount_10d") == pl.col("_peak_amount"))
            & (pl.col("_peak_high_10d") == pl.col("_peak_high"))
            # 条件 3: 显著放量
            & (pl.col("_peak_amount") >= pl.col("_avg_amount_15") * peak_vol_ratio)
            # 条件 4: 高点回撤充分
            & (pl.col("_drawdown_pct") >= drawdown_min)
            # 条件 5: 近5日累计下跌
            & (pl.col("_sum_change_5d") < 0)
            # 条件 6: 回落中明显缩量
            & (pl.col("_last_amount") <= pl.col("_peak_amount") * shrink_ratio)
            # 条件 7: 收盘跌破 MA20
            & (pl.col("_last_close") < pl.col("_last_ma20"))
            # 条件 8: 末日止跌企稳 (跌幅收窄或翻红)
            & (pl.col("_last_change") > pl.col("_prev_change"))
        )
    )

    # 返回通过筛选的股票的全部历史行 (引擎会按 as_of 裁剪到目标日)
    return df.join(stats, on="symbol", how="inner")
