"""强势急涨回调 - 放量大市值股近期急涨后短线超跌

策略逻辑:
  1. 近10个交易日，日平均成交额 > 2亿
  2. 近10个交易日，每日成交额都 > 1.5亿
  3. 近13个交易日，至少有一天涨幅 > 6%
  4. 日线 BIAS1 (N日乖离率, 默认6日) < -7
  5. 流通市值 > 100亿 (在 BASIC_FILTER 中检查)
  6. 股票价格 (收盘价) 在日线布林线下轨附近 (不高于下轨上方 2%, 含跌破, 可调)

BIAS1 口径: (收盘价 - MA_N) / MA_N * 100 (百分点), N 默认 6,
与通达信/同花顺 BIAS1 (6日乖离率) 惯例一致, 周期可用 bias_period 调整。
布林下轨为 enriched 的 boll_lower 列 (MA20 - 2σ), 容差由 boll_lower_tolerance 控制。

执行后端: python_history_legacy (需要历史窗口)
"""
from __future__ import annotations

import polars as pl

META = {
    "id": "lab_surge_bias_pullback",
    "name": "强势急涨回调",
    "description": "近13日有单日大涨(>6%) + 近10日放量 + 6日乖离率BIAS1<-7 + 价格贴近布林下轨 + 流通市值>100亿",
    "tags": ["强势", "回调", "超跌", "布林", "放量", "大市值"],
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
            "id": "surge_change_min",
            "label": "13日单日涨幅下限(%)",
            "type": "float",
            "default": 6.0,
            "min": 3.0,
            "max": 20.0,
            "step": 0.5,
        },
        {
            "id": "bias1_max",
            "label": "BIAS1上限(百分点)",
            "type": "float",
            "default": -4.0,
            "min": -15.0,
            "max": 0.0,
            "step": 0.5,
        },
        {
            "id": "bias_period",
            "label": "BIAS1均线周期(日)",
            "type": "int",
            "default": 6,
            "min": 3,
            "max": 12,
            "step": 1,
        },
        {
            "id": "boll_lower_tolerance",
            "label": "布林下轨附近容差(%)",
            "type": "float",
            "default": 2.0,
            "min": 0.0,
            "max": 10.0,
            "step": 0.5,
        },
    ],
    # 乖离率取负参与评分: 超跌越深, -bias 越大, 归一化后得分越高
    "scoring": {"_max_change_13d": 0.4, "_bias1_neg": 0.3, "vol_ratio_5d": 0.3},
    "order_by": "score",
    "descending": True,
    "limit": 50,
}

EXECUTION_BACKEND = "python_history_legacy"

LOOKBACK_DAYS = 13

# filter_history 动态依赖的公开字段 (close 为基础 OHLCV 列, 由基础过滤自动携带)
REQUIRED_FEATURES = {"amount", "change_pct", "boll_lower"}

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
    """历史窗口过滤: 返回满足条件的行 (选股引擎再按 as_of 裁剪)。

    用 over("symbol") 逐行滚动窗口计算: 选股时最新一行的窗口即"截至 as_of
    的近 N 个交易日"; 回测时每个 (symbol, date) 行按各自截至当日的窗口判断,
    不引入未来数据。滚动窗口在数据不足时为 null, 比较结果为 null 被 filter
    丢弃, 即上市/停牌导致交易日不足的股票自动排除 (fail-closed)。

    条件 1-4、6 在此检查；条件 5 (流通市值) 由 BASIC_FILTER 在 as_of 日检查。
    """
    df = df.sort(["symbol", "date"])

    # 从参数提取阈值 (用户输入为亿元/百分比，转为元/小数; BIAS 以百分点计)
    avg_amt_min = float(params.get("avg_amount_min", 2.0)) * 1e8
    daily_amt_min = float(params.get("daily_amount_min", 1.5)) * 1e8
    surge_min = float(params.get("surge_change_min", 6.0)) / 100.0
    bias_max = float(params.get("bias1_max", -4.0))
    bias_period = int(params.get("bias_period", 6))
    boll_tol = float(params.get("boll_lower_tolerance", 2.0)) / 100.0

    # BIAS1 分母: N 日收盘均线 (周期由参数决定, 上限 12 < LOOKBACK_DAYS)
    ma_n = pl.col("close").rolling_mean(bias_period).over("symbol")

    return (
        df.with_columns(
            # 条件 1: 近10日平均成交额
            pl.col("amount").rolling_mean(10).over("symbol").alias("_avg_amount_10d"),
            # 条件 2: 近10日最小成交额 (每日都达标 = 最小值达标)
            pl.col("amount").rolling_min(10).over("symbol").alias("_min_amount_10d"),
            # 条件 3: 近13日最大涨幅 (至少一天达标 = 最大值达标)
            pl.col("change_pct").rolling_max(13).over("symbol").alias("_max_change_13d"),
            # 条件 4: BIAS1 = (收盘价 - MA_N) / MA_N * 100 (百分点)
            ((pl.col("close") - ma_n) / ma_n * 100).alias("_bias1"),
            # 条件 6: 收盘价相对布林下轨的偏离 (<=容差 即"下轨附近", 含跌破)
            ((pl.col("close") - pl.col("boll_lower")) / pl.col("boll_lower")).alias("_boll_dev"),
        )
        .filter(
            (pl.col("_avg_amount_10d") > avg_amt_min)
            & (pl.col("_min_amount_10d") > daily_amt_min)
            & (pl.col("_max_change_13d") > surge_min)
            & (pl.col("_bias1") < bias_max)
            & (pl.col("_boll_dev") <= boll_tol)
        )
        # 评分用: 乖离率取负 (超跌越深得分越高)
        .with_columns((-pl.col("_bias1")).alias("_bias1_neg"))
    )
