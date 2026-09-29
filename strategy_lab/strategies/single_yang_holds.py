"""单阳不破 - 近20个交易日内出现大阳线, 且单阳距今超过10个交易日, 期间收盘价始终不跌破该阳线开盘价

策略逻辑:
  1. 近20个交易日内出现"单阳": 当日涨幅 >= surge_change_min (默认7%)
  2. 自单阳次日起, 每日收盘价均不跌破该阳线开盘价;
     期间若再出现新的单阳, 以最新一根为准重新起算
  3. 最近一根单阳距今大于 10 个交易日 (按个股实际交易日计)
  4. 流通市值 > 100亿 (在 BASIC_FILTER 中检查)

执行后端: python_history_legacy (需要历史窗口)
"""
from __future__ import annotations

import polars as pl

META = {
    "id": "lab_single_yang_holds",
    "name": "单阳不破",
    "description": "近20日出现大阳线(涨幅≥7%)且单阳距今>10个交易日, 期间收盘价始终不跌破阳线开盘价 + 流通市值>100亿",
    "tags": ["单阳不破", "强势", "整理", "大市值"],
    "asset_types": ["stock"],
    "timeframes": ["1d"],
    "params": [
        {
            "id": "surge_change_min",
            "label": "单阳涨幅下限(%)",
            "type": "float",
            "default": 7.0,
            "min": 2.0,
            "max": 15.0,
            "step": 0.5,
        },
    ],
    # 评分: 单阳涨幅越大、单阳量比越高、20日动量越强, 得分越高
    "scoring": {"_yang_change": 0.4, "_yang_vol_ratio": 0.3, "momentum_20d": 0.3},
    "order_by": "score",
    "descending": True,
    "limit": 50,
}

EXECUTION_BACKEND = "python_history_legacy"

LOOKBACK_DAYS = 20

# 条件 3: 最近一根单阳距今须大于该值 (按个股实际交易日计)
MIN_DAYS_SINCE_YANG = 10

# filter_history 动态依赖的公开字段 (open/close/volume 为基础 OHLCV 列, 由基础过滤自动携带)
REQUIRED_FEATURES = {"change_pct"}

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
    """历史窗口过滤: 返回满足条件的行 (选股引擎再按 as_of 裁剪)。

    用 over("symbol") 逐行计算: 选股时最新一行即"截至 as_of 的近 20 个
    交易日"; 回测时每个 (symbol, date) 行按各自截至当日的窗口判断,
    不引入未来数据。单阳属性用 forward_fill 取"截至当日"最近一根单阳,
    中间值为 null (尚无单阳) 时比较结果为 null 被 filter 丢弃 (fail-closed)。

    条件 1-3 在此检查; 条件 4 (流通市值) 由 BASIC_FILTER 在 as_of 日检查。
    """
    df = df.sort(["symbol", "date"])

    # 从参数提取阈值 (用户输入为百分比, 转为小数)
    surge_min = float(params.get("surge_change_min", 7.0)) / 100.0

    # 单阳: 涨幅达标
    is_yang = pl.col("change_pct") >= surge_min

    # 当日量比 (评分用): 当日成交量 / 近20日均量 (历史不足时按实际天数)
    vol_ratio = pl.col("volume") / pl.col("volume").rolling_mean(
        LOOKBACK_DAYS, min_samples=1
    ).over("symbol")

    df = df.with_columns(
        pl.int_range(pl.len()).over("symbol").alias("_row_idx"),
        is_yang.alias("_is_yang"),
        vol_ratio.alias("_vol_ratio"),
    )

    # 最近一根单阳的行号/开盘价/涨幅/量比 (每行取"截至当日"最近一根)
    df = df.with_columns(
        pl.when(pl.col("_is_yang")).then(pl.col("_row_idx")).forward_fill().over("symbol").alias("_yang_idx"),
        pl.when(pl.col("_is_yang")).then(pl.col("open")).forward_fill().over("symbol").alias("_yang_open"),
        pl.when(pl.col("_is_yang")).then(pl.col("change_pct")).forward_fill().over("symbol").alias("_yang_change"),
        pl.when(pl.col("_is_yang")).then(pl.col("_vol_ratio")).forward_fill().over("symbol").alias("_yang_vol_ratio"),
    )

    # 破位: 单阳次日起收盘价跌破该阳开盘价 (阳线当日不计)。
    # 状态机: 单阳日重置为 0, 破位日置 1, 其余日期 forward_fill 沿用 ——
    # 即"自最近一根单阳以来是否破位", 新单阳出现后重新起算。
    broken = (
        (pl.col("_row_idx") > pl.col("_yang_idx"))
        & (pl.col("close") < pl.col("_yang_open"))
    )
    df = df.with_columns(
        pl.when(pl.col("_is_yang")).then(0)
        .when(broken).then(1)
        .forward_fill().over("symbol")
        .alias("_broken_since_yang"),
        (pl.col("_row_idx") - pl.col("_yang_idx")).alias("_days_since_yang"),
    )

    return (
        df.filter(
            pl.col("_yang_idx").is_not_null()
            & (pl.col("_days_since_yang") < LOOKBACK_DAYS)        # 单阳在近20个交易日内
            & (pl.col("_days_since_yang") > MIN_DAYS_SINCE_YANG)  # 最近一根单阳距今大于10个交易日
            & (pl.col("_broken_since_yang") == 0)                 # 自单阳以来未破位
        )
        .drop(
            "_is_yang", "_row_idx", "_vol_ratio", "_yang_idx",
            "_yang_open", "_days_since_yang", "_broken_since_yang",
        )
    )
