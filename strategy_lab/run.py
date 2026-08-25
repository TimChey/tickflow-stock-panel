"""策略文件直接启动器。

用法:
    python strategy_lab/run.py <策略文件.py>
    python strategy_lab/run.py volume_ma_breakout.py
    python strategy_lab/run.py strategies/volume_ma_breakout.py --limit 50
    python strategy_lab/run.py volume_ma_breakout.py --date 2025-01-10

策略文件本身保持纯净 (只含 META + filter)，由本脚本负责路径引导和执行。
"""
import argparse
import sys
from pathlib import Path

# 将项目根和 backend 加入 sys.path
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from strategy_lab.runner import run_file  # noqa: E402


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


def main():
    parser = argparse.ArgumentParser(
        description="直接运行策略文件，选股结果打印到控制台",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python strategy_lab/run.py volume_ma_breakout.py
  python strategy_lab/run.py volume_ma_breakout.py --limit 50
  python strategy_lab/run.py volume_ma_breakout.py --date 2025-01-10
        """,
    )
    parser.add_argument("file", help="策略文件名或路径 (如 volume_ma_breakout.py)")
    parser.add_argument("--date", type=str, help="目标日期 (YYYY-MM-DD)")
    parser.add_argument("--limit", type=int, default=20, help="显示前 N 条 (默认 20)")
    parser.add_argument("--pool", type=str, help="限定股票池 (逗号分隔)")
    args = parser.parse_args()

    strategy_file = _resolve_strategy_file(args.file)
    target_date = None
    if args.date:
        from datetime import date as date_type
        target_date = date_type.fromisoformat(args.date)

    pool = [s.strip() for s in args.pool.split(",")] if args.pool else None

    run_file(strategy_file, as_of=target_date, pool=pool, limit=args.limit)


if __name__ == "__main__":
    main()
