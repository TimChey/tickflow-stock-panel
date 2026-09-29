"""大盘暴跌错杀低吸 - 大盘大跌之日, 选前期强势的超跌错杀股博弈反弹

适用场景: 大盘系统性大跌当日 (以全市场日涨幅中位数度量, fail-closed,
正常交易日不产出任何结果)。选当日跌得最深、且明显弱于市场中位数,
但大跌前处于强势 (剔除末日后的近19日累计涨幅为正) 的错杀股,
博弈次日超跌反弹 (接飞刀策略, 风险较高, 仓位宜轻)。

策略逻辑:
  1. 大盘大跌: as_of 日全市场个股涨幅中位数 <= -3% (不满足则当日为空)
  2. 个股超跌: 当日涨幅 <= -6%
  3. 错杀深度: 当日涨幅 <= 全市场中位数 - 2pp (跌得显著比大盘更深)
  4. 前期强势: 剔除末日后的近19个交易日累计涨幅 >= 0 (非趋势下跌)
  5. 流动性: 近10个交易日日平均成交额 > 1亿
  6. 基础过滤: 排除 ST、新股、低价股; 含北交所 (在 BASIC_FILTER 中检查)

执行后端: python_history_legacy (需要全市场历史窗口)
"""
from __future__ import annotations

import polars as pl

META = {
    "id": "lab_crash_day_dip",
    "name": "大盘暴跌错杀低吸",
    "description": "大盘大跌日(全市场涨幅中位数<=-3%), 选当日超跌(<=-6%且深于大盘2pp)但前期强势的错杀股",
    "tags": ["大盘暴跌", "超跌", "错杀", "低吸", "反弹"],
    "asset_types": ["stock"],
    "timeframes": ["1d"],
    "params": [
        {
            "id": "crash_median",
            "label": "大盘大跌阈值(全市场涨幅中位数,%)",
            "type": "float",
            "default": -3.0,
            "min": -10.0,
            "max": 0.0,
            "step": 0.5,
        },
        {
            "id": "drop_min",
            "label": "个股当日跌幅下限(%)",
            "type": "float",
            "default": 6.0,
            "min": 2.0,
            "max": 15.0,
            "step": 0.5,
        },
        {
            "id": "rel_extra",
            "label": "跌幅需超过大盘中位数(pp)",
            "type": "float",
            "default": 2.0,
            "min": 0.0,
            "max": 8.0,
            "step": 0.5,
        },
        {
            "id": "prior_trend_min",
            "label": "前19日累计涨幅下限(%)",
            "type": "float",
            "default": 0.0,
            "min": -20.0,
            "max": 30.0,
            "step": 1.0,
        },
        {
            "id": "avg_amount_min",
            "label": "10日平均成交额下限(亿)",
            "type": "float",
            "default": 1.0,
            "min": 0.1,
            "max": 20.0,
            "step": 0.1,
        },
    ],
    "scoring": {
        "_rel_drop": 0.35,
        "_prior_trend": 0.3,
        "_drawdown_pct": 0.2,
        "_avg_amount_10d": 0.15,
    },
    "order_by": "score",
    "descending": True,
    "limit": 50,
}

EXECUTION_BACKEND = "python_history_legacy"

# 21 行窗口: 近20个交易日的 change_pct (前19日趋势 + 末日大跌)
LOOKBACK_DAYS = 21

# change_pct 为引擎计算列, 需显式声明 (symbol/date/OHLCV/amount 为基础列)
REQUIRED_FEATURES = {"change_pct"}

BASIC_FILTER = {
    "price_min": 3,
    "price_max": 500,
    # 流动性由条件 5 控制, 取消默认的市值/当日成交额下限
    "market_cap_min": None,
    "float_cap_min": None,
    "amount_min": None,
    "exclude_st": True,
    "exclude_new_days": 30,
    "boards": ["沪主板", "深主板", "创业板", "科创板", "北交所"],
}

ENTRY_SIGNALS = []
EXIT_SIGNALS = ["signal_ma20_breakdown"]
STOP_LOSS = -0.05
MAX_HOLD_DAYS = 5
ALERTS = []


def filter_history(df: pl.DataFrame, params: dict) -> pl.DataFrame:
    """历史窗口过滤: 返回满足条件的股票 (含所有历史行, 引擎再按 as_of 裁剪)。

    条件 1-5 在此检查; BASIC_FILTER (价格/ST/新股/板块) 由引擎在 as_of 日检查。
    """
    df = df.sort(["symbol", "date"])

    # 从参数提取阈值 (用户输入为百分比/亿元, 转为小数/元)
    crash_median = float(params.get("crash_median", -3.0)) / 100.0
    drop_min = float(params.get("drop_min", 6.0)) / 100.0
    rel_extra = float(params.get("rel_extra", 2.0)) / 100.0
    prior_trend_min = float(params.get("prior_trend_min", 0.0)) / 100.0
    avg_amt_min = float(params.get("avg_amount_min", 1.0)) * 1e8

    # 条件 1: as_of 日全市场涨幅中位数 (fail-closed, 非大跌日条件恒不成立)
    as_of_date = df["date"].max()
    market_median = (
        df.filter(pl.col("date") == as_of_date)
        .get_column("change_pct")
        .median()
    )
    if market_median is None or market_median > crash_median:
        return df.clear()

    # 按 symbol 分组, 取近 N 个交易日的统计量
    stats = (
        df.group_by("symbol", maintain_order=True)
        .agg(
            # 条件 2/3: 最新一日涨幅 (大盘大跌日的个股跌幅)
            pl.col("change_pct").last().alias("_last_change"),
            # 条件 4: 近20日累计涨幅 - 末日涨幅 = 前期19日累计涨幅
            pl.col("change_pct").tail(20).sum().alias("_sum_change_20d"),
            # 条件 5: 近10日平均成交额
            pl.col("amount").tail(10).mean().alias("_avg_amount_10d"),
            # 评分用: 收盘价距近20日最高价的折价
            pl.col("close").last().alias("_last_close"),
            pl.col("high").tail(20).max().alias("_high_20d"),
            # 数据完整性: 确保有足够交易日
            pl.col("date").count().alias("_day_count"),
        )
        .with_columns(
            (pl.col("_sum_change_20d") - pl.col("_last_change")).alias("_prior_trend"),
            (1 - pl.col("_last_close") / pl.col("_high_20d")).alias("_drawdown_pct"),
            # 评分用: 错杀深度 = 个股跌幅相对大盘中位数的超出幅度
            (market_median - pl.col("_last_change")).alias("_rel_drop"),
        )
        .filter(
            (pl.col("_day_count") >= 21)
            # 条件 2: 个股当日超跌
            & (pl.col("_last_change") <= -drop_min)
            # 条件 3: 跌得显著比大盘更深 (错杀)
            & (pl.col("_last_change") <= market_median - rel_extra)
            # 条件 4: 大跌前处于强势, 非趋势下跌
            & (pl.col("_prior_trend") >= prior_trend_min)
            # 条件 5: 流动性
            & (pl.col("_avg_amount_10d") > avg_amt_min)
        )
    )

    # 返回通过筛选的股票的全部历史行 (引擎会按 as_of 裁剪到目标日)
    return df.join(stats, on="symbol", how="inner")
