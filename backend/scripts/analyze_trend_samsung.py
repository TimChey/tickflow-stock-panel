#!/usr/bin/env python
"""查询三星电气(601567.SH)近1年K线数据并分析当前趋势。

读取本地 enriched parquet (前复权 OHLCV),调用项目自身的指标流水线
(compute_indicators + compute_signals)与关键价位模块(compute_levels),
输出客观的技术面趋势分析报告。不下任何买卖指令。

用法 (从 backend/ 目录运行):
    uv run python -m scripts.analyze_trend_samsung
    # 或直接:
    uv run python scripts/analyze_trend_samsung.py
"""
from __future__ import annotations

import logging
import math
import sys
from datetime import date, timedelta
from pathlib import Path

import polars as pl

# 确保 backend/ 在 sys.path 上 (直接 python 运行时兜底)
_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from app.config import settings
from app.indicators.pipeline import compute_indicators, compute_signals
from app.indicators.levels import compute_levels, summarize_levels
from app.parquet import scan_enriched_parquet

logger = logging.getLogger(__name__)

SYMBOL = "601567.SH"
SYMBOL_NAME = "三星电气"


# ================================================================
# 数据加载
# ================================================================
def load_kline(symbol: str, years: float = 1.3) -> pl.DataFrame:
    """读取本地 enriched parquet 中该标的近 N 年日K (前复权 OHLCV)。

    取 1.3 年是为了让 MA60/MACD 等指标在 1 年窗口的起点处已完成预热,
    避免 1 年窗口开头若干行指标为空影响分析。
    """
    end = date.today()
    start = end - timedelta(days=int(365 * years))
    glob_path = str(settings.data_dir / "kline_daily_enriched" / "**" / "*.parquet")
    df = (
        scan_enriched_parquet(glob_path)
        .filter(
            (pl.col("symbol") == symbol)
            & (pl.col("date") >= start)
            & (pl.col("date") <= end)
        )
        .collect()
    )
    if df.is_empty():
        # 退路: enriched 无数据时读 14 列原始日K
        glob_daily = str(settings.data_dir / "kline_daily" / "**" / "*.parquet")
        from app.parquet import scan_daily_parquet
        df = (
            scan_daily_parquet(glob_daily)
            .filter(
                (pl.col("symbol") == symbol)
                & (pl.col("date") >= start)
                & (pl.col("date") <= end)
            )
            .collect()
        )
    return df.sort("date")


# ================================================================
# 工具
# ================================================================
def _f(v, nd=2) -> str:
    if v is None:
        return "--"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "--"
    return "--" if not math.isfinite(f) else f"{f:.{nd}f}"


def _pct(v, nd=2) -> str:
    if v is None:
        return "--"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "--"
    return "--" if not math.isfinite(f) else f"{f * 100:.{nd}f}%"


def _last(df: pl.DataFrame, col: str):
    if df.is_empty() or col not in df.columns:
        return None
    s = df.select(pl.col(col).tail(1)).to_series()
    return s[0] if len(s) else None


def _last_n(df: pl.DataFrame, col: str, n: int) -> list:
    if df.is_empty() or col not in df.columns:
        return []
    s = df.select(pl.col(col).tail(n)).to_series()
    return s.to_list()


# ================================================================
# 分析
# ================================================================
def analyze(df: pl.DataFrame) -> None:
    if df.is_empty():
        print(f"[警告] 本地未找到 {SYMBOL}({SYMBOL_NAME}) 的日K数据,请先在数据页同步。")
        return

    df = compute_indicators(df)
    df = compute_signals(df)

    one_year_ago = date.today() - timedelta(days=365)
    df_1y = df.filter(pl.col("date") >= one_year_ago)
    if df_1y.is_empty():
        df_1y = df

    close = _last(df_1y, "close")
    trade_date = _last(df_1y, "date")

    print("=" * 64)
    print(f"  {SYMBOL_NAME} ({SYMBOL})  趋势分析报告")
    print(f"  数据区间: {df_1y['date'][0]} ~ {df_1y['date'][-1]}  ({df_1y.height} 个交易日)")
    print(f"  最新交易日: {trade_date}")
    print("=" * 64)

    # ---------- 1. 价格概览 ----------
    high_1y = float(df_1y["high"].max())
    low_1y = float(df_1y["low"].min())
    close_1y_ago = float(df_1y["close"][0])
    ann_ret = (close - close_1y_ago) / close_1y_ago if close_1y_ago else float("nan")
    chg = _last(df_1y, "change_pct")
    amp = _last(df_1y, "amplitude")
    vol = _last(df_1y, "volume")
    amt = _last(df_1y, "amount")
    turnover = _last(df_1y, "turnover_rate")

    print("\n【1. 价格概览】")
    print(f"  最新收盘: {_f(close)} 元   日涨跌幅: {_pct(chg)}   日振幅: {_pct(amp)}")
    print(f"  近1年最高: {_f(high_1y)} 元   近1年最低: {_f(low_1y)} 元")
    print(f"  近1年涨跌幅: {_pct(ann_ret)}   (区间起点收盘 {_f(close_1y_ago)})")
    print(f"  最新成交量: {_f(vol, 0)} 手   成交额: {_f(amt / 1e8, 2)} 亿   换手率: {_pct(turnover)}")
    print(f"  距1年高点: {_pct((close - high_1y) / high_1y)}   距1年低点: {_pct((close - low_1y) / low_1y)}")

    # ---------- 2. 均线系统 ----------
    ma5 = _last(df_1y, "ma5")
    ma10 = _last(df_1y, "ma10")
    ma20 = _last(df_1y, "ma20")
    ma60 = _last(df_1y, "ma60")
    print("\n【2. 均线系统 (MA)】")
    print(f"  MA5={_f(ma5)}  MA10={_f(ma10)}  MA20={_f(ma20)}  MA60={_f(ma60)}")
    if None not in (ma5, ma10, ma20, ma60):
        if ma5 > ma10 > ma20 > ma60:
            align = "多头排列 (MA5>MA10>MA20>MA60), 中短期趋势向上"
        elif ma5 < ma10 < ma20 < ma60:
            align = "空头排列 (MA5<MA10<MA20<MA60), 中短期趋势向下"
        else:
            align = "均线纠缠/非典型排列, 趋势不明确"
        print(f"  均线排列: {align}")
    print(f"  价格 vs MA5: {'上方' if close and ma5 and close > ma5 else '下方'}")
    print(f"  价格 vs MA20: {'上方' if close and ma20 and close > ma20 else '下方'}")
    print(f"  价格 vs MA60: {'上方' if close and ma60 and close > ma60 else '下方'}  (中期趋势{'向上' if close and ma60 and close > ma60 else '向下'})")

    # ---------- 3. MACD ----------
    dif = _last(df_1y, "macd_dif")
    dea = _last(df_1y, "macd_dea")
    hist = _last(df_1y, "macd_hist")
    print("\n【3. MACD】")
    print(f"  DIF={_f(dif)}  DEA={_f(dea)}  HIST={_f(hist)}")
    if dif is not None and dea is not None:
        if dif > dea:
            print(f"  DIF 在 DEA 上方, MACD 处于多头; 最新是否金叉: {bool(_last(df_1y, 'signal_macd_golden'))}")
        else:
            print(f"  DIF 在 DEA 下方, MACD 处于空头; 最新是否死叉: {bool(_last(df_1y, 'signal_macd_dead'))}")
        if dif < 0 and dea < 0:
            print("  DIF/DEA 均在零轴下方, 中期动能偏弱")
        elif dif > 0 and dea > 0:
            print("  DIF/DEA 均在零轴上方, 中期动能偏强")

    # ---------- 4. KDJ ----------
    kk = _last(df_1y, "kdj_k")
    kd = _last(df_1y, "kdj_d")
    kj = _last(df_1y, "kdj_j")
    print("\n【4. KDJ】")
    print(f"  K={_f(kk)}  D={_f(kd)}  J={_f(kj)}")
    if None not in (kk, kd, kj):
        if kj > 100:
            print("  J 值 >100, 极度超买")
        elif kk > 80:
            print("  K>80, 超买区")
        elif kj < 0:
            print("  J 值 <0, 极度超卖")
        elif kk < 20:
            print("  K<20, 超卖区")
        else:
            print("  KDJ 处于中性区间 (20~80)")
        print(f"  K{'>' if kk > kd else '<'}D, KDJ {'金叉' if kk > kd else '死叉'}方向")

    # ---------- 5. RSI ----------
    rsi6 = _last(df_1y, "rsi_6")
    rsi14 = _last(df_1y, "rsi_14")
    rsi24 = _last(df_1y, "rsi_24")
    print("\n【5. RSI】")
    print(f"  RSI6={_f(rsi6)}  RSI14={_f(rsi14)}  RSI24={_f(rsi24)}")
    if rsi14 is not None:
        if rsi14 > 70:
            print("  RSI14>70, 短期偏强/超买")
        elif rsi14 < 30:
            print("  RSI14<30, 短期偏弱/超卖")
        else:
            print("  RSI14 处于中性区间 (30~70)")

    # ---------- 6. 布林带 ----------
    bup = _last(df_1y, "boll_upper")
    blw = _last(df_1y, "boll_lower")
    bmid = ma20
    print("\n【6. 布林带 (BOLL)】")
    print(f"  上轨={_f(bup)}  中轨={_f(bmid)}  下轨={_f(blw)}")
    if None not in (close, bup, blw, bmid) and bup > blw:
        if close > bup:
            pos = "突破上轨 (强势/超买)"
        elif close < blw:
            pos = "跌破下轨 (弱势/超卖)"
        elif close > bmid:
            pos = "中轨上方, 偏强"
        else:
            pos = "中轨下方, 偏弱"
        bw = (bup - blw) / bmid * 100 if bmid else float("nan")
        print(f"  位置: {pos}   带宽: {_pct(bw / 100)} (带宽收窄意味波动降低)")

    # ---------- 7. 量价 ----------
    vol_ratio = _last(df_1y, "vol_ratio_5d")
    vol_ma5 = _last(df_1y, "vol_ma5")
    print("\n【7. 量价配合】")
    print(f"  量比(5日): {_f(vol_ratio)}   5日均量: {_f(vol_ma5, 0)} 手")
    if vol_ratio is not None:
        if vol_ratio >= 2.0:
            print("  量比>=2, 显著放量")
        elif vol_ratio >= 1.5:
            print("  量比偏高, 温和放量")
        elif vol_ratio < 0.7:
            print("  量比<0.7, 缩量")
        else:
            print("  量能接近常态")
    vols = _last_n(df_1y, "volume", 5)
    if len(vols) >= 2:
        trend = "递增" if vols[-1] > vols[0] else "递减"
        print(f"  近5日量能: {trend}  ({[round(x) for x in vols]})")

    # ---------- 8. 动量 ----------
    m5 = _last(df_1y, "momentum_5d")
    m20 = _last(df_1y, "momentum_20d")
    m60 = _last(df_1y, "momentum_60d")
    print("\n【8. 动量】")
    print(f"  5日动量: {_pct(m5)}   20日动量: {_pct(m20)}   60日动量: {_pct(m60)}")
    if None not in (m5, m20, m60):
        if m5 > 0 and m20 > 0 and m60 > 0:
            mom = "短中长动量共振向上, 趋势偏多"
        elif m5 < 0 and m20 < 0 and m60 < 0:
            mom = "短中长动量共振向下, 趋势偏空"
        elif m60 < 0:
            short = "转正(企稳迹象)" if m5 > 0 else "为负(延续调整)"
            mom = f"长期动量向下, 中期趋势仍弱; 短期动量{short}"
        else:
            mom = "动量方向不一, 处于震荡/转折期"
        print(f"  动量结构: {mom}")

    # ---------- 9. 波动率 ----------
    avol = _last(df_1y, "annual_vol_20d")
    atr = _last(df_1y, "atr_14")
    print("\n【9. 波动率 / 风险】")
    print(f"  20日年化波动率: {_pct(avol / 100) if avol is not None else '--'}   ATR(14): {_f(atr)} 元")
    if atr and close:
        print(f"  ATR/收盘 ≈ {_pct(atr / close)} (单日典型波动幅度)")

    # ---------- 10. 关键价位 ----------
    levels = compute_levels(df_1y)
    print("\n【10. 关键价位】")
    print(f"  价位摘要: {summarize_levels(levels, close)}")
    sr = levels.get("sr", [])
    resistances = sorted(
        [r for r in sr if r.get("side") == "resistance" and r.get("value", 0) > (close or 0)],
        key=lambda x: x["value"],
    )[:2]
    supports = sorted(
        [s for s in sr if s.get("side") == "support" and s.get("value", 0) < (close or 0)],
        key=lambda x: -x["value"],
    )[:2]
    if resistances:
        print("  上方压力位: " + ", ".join(f"{_f(r['value'])}({r.get('label','')})" for r in resistances))
    if supports:
        print("  下方支撑位: " + ", ".join(f"{_f(s['value'])}({s.get('label','')})" for s in supports))

    # ---------- 11. 近期信号 ----------
    print("\n【11. 近10日信号事件】")
    sig_cols = [c for c in df_1y.columns if c.startswith("signal_")]
    tail10 = df_1y.tail(10)
    events: list[str] = []
    for row in tail10.iter_rows(named=True):
        for c in sig_cols:
            if row.get(c):
                events.append(f"{row['date']} {c.replace('signal_', '')}")
    if events:
        for e in events[-15:]:
            print(f"    {e}")
    else:
        print("    近10日无显著技术信号")

    # ---------- 12. 综合研判 ----------
    print("\n【12. 综合趋势研判】")
    long_trend = "向上" if (close and ma60 and close > ma60) else "向下"
    mid_trend = "向上" if (close and ma20 and close > ma20) else "向下"
    macd_state = "多头" if (dif and dea and dif > dea) else "空头"
    print(f"  中期趋势 (价格 vs MA60): {long_trend}")
    print(f"  中短期趋势 (价格 vs MA20): {mid_trend}")
    print(f"  MACD 状态: {macd_state}")
    print(f"  近1年涨跌幅: {_pct(ann_ret)}  距1年高点: {_pct((close - high_1y) / high_1y) if high_1y else '--'}")
    print("  --")
    print("  客观陈述: 上述指标反映该股当前的技术状态, 仅供研究参考,")
    print("  不构成任何投资建议或买卖指令。交易有风险, 入市需谨慎。")


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    df = load_kline(SYMBOL)
    analyze(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
