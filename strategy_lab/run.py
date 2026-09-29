"""策略文件直接启动器。

用法:
    python strategy_lab/run.py <策略文件.py>
    python strategy_lab/run.py volume_ma_breakout.py
    python strategy_lab/run.py volume_ma_breakout.py --date 2025-01-10

策略文件本身保持纯净 (只含 META + filter)，由本脚本负责路径引导和执行。
选股结果打印后，对选出的股票查询实时行情并展示。

实时行情口径说明 (与项目主数据流隔离):
    使用腾讯公开行情接口 (qt.gtimg.cn)，复用 backend/scripts/stock_care_monitor.py
    的拉取与渲染逻辑，仅用于终端展示，不落盘、不进入 provider 标准化。
    腾讯接口涨跌幅为百分数制 (3.66 表示 3.66%)，与项目内部小数制契约不同。
"""
import argparse
import sys
from pathlib import Path

# 将项目根和 backend 加入 sys.path
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from strategy_lab.runner import run_file  # noqa: E402

# 腾讯行情接口单次请求的 symbol 数上限 (控制 URL 长度)
_QUOTE_BATCH_SIZE = 60


def _resolve_strategy_file(name: str) -> Path:
    """支持传入文件名或相对/绝对路径。"""
    p = Path(name)
    if p.exists():
        return p.resolve()

    # 尝试在 strategy_lab/strategies/ 下查找
    in_strategies = Path(__file__).resolve().parent / "strategies" / name
    if in_strategies.exists():
        return in_strategies.resolve()

    # 尝试补 .py 后缀
    if not name.endswith(".py"):
        with_ext = Path(__file__).resolve().parent / "strategies" / f"{name}.py"
        if with_ext.exists():
            return with_ext.resolve()

    raise FileNotFoundError(f"找不到策略文件: {name}")


def _to_qt_symbols(symbols: list[str]) -> list[str]:
    """项目符号 (600519.SH) -> 腾讯行情符号 (sh600519)；跳过不支持的格式。"""
    out: list[str] = []
    for symbol in symbols:
        code, _, suffix = symbol.strip().upper().partition(".")
        prefix = {"SH": "sh", "SZ": "sz", "BJ": "bj"}.get(suffix)
        if prefix and len(code) == 6 and code.isdigit():
            out.append(prefix + code)
    return out


def _print_realtime_quotes(symbols: list[str]) -> None:
    """查询并打印选中股票的实时行情 (腾讯公开接口，仅终端展示)。"""
    import httpx
    from scripts.stock_care_monitor import (
        COLUMNS,
        display_width,
        fetch_quotes,
        pad,
        pad_left,
        render_quote_line,
    )

    qt_symbols = _to_qt_symbols(symbols)
    if not qt_symbols:
        return

    quotes = []
    with httpx.Client(timeout=10.0) as client:
        for i in range(0, len(qt_symbols), _QUOTE_BATCH_SIZE):
            quotes.extend(fetch_quotes(client, qt_symbols[i : i + _QUOTE_BATCH_SIZE]))

    if not quotes:
        print("\n实时行情: 未获取到数据")
        return

    print(f"\n实时行情 ({len(quotes)} 只, 红涨绿跌):")
    header = "  ".join(
        pad_left(name, width) if right else pad(name, width)
        for name, width, right in COLUMNS
    )
    print(header)
    print("-" * display_width(header))
    for quote in quotes:
        print(render_quote_line(quote))


def main():
    parser = argparse.ArgumentParser(
        description="直接运行策略文件，选股结果打印到控制台",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python strategy_lab/run.py volume_ma_breakout.py
  python strategy_lab/run.py volume_ma_breakout.py --date 2025-01-10
        """,
    )
    parser.add_argument("file", help="策略文件名或路径 (如 volume_ma_breakout.py)")
    parser.add_argument("--date", type=str, help="目标日期 (YYYY-MM-DD)")
    parser.add_argument("--pool", type=str, help="限定股票池 (逗号分隔)")
    args = parser.parse_args()

    strategy_file = _resolve_strategy_file(args.file)
    target_date = None
    if args.date:
        from datetime import date as date_type
        target_date = date_type.fromisoformat(args.date)

    pool = [s.strip() for s in args.pool.split(",")] if args.pool else None

    result = run_file(strategy_file, as_of=target_date, pool=pool)

    # 对选出的股票查询实时行情 (与选股表展示的全部条目一致)
    shown = (result or {}).get("rows", [])
    symbols = [str(r["symbol"]) for r in shown if r.get("symbol")]
    if symbols:
        try:
            _print_realtime_quotes(symbols)
        except Exception as e:  # 实时行情为附加展示，失败不影响选股结果
            print(f"\n实时行情查询失败: {e}")


if __name__ == "__main__":
    main()
