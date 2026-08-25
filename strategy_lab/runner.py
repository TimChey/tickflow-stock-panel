"""策略选股执行器 - 复用后端数据基础设施，提供独立选股能力。

架构:
  DataStore (parquet 文件) -> KlineRepository (缓存+查询)
  -> ScreenerService (构建策略数据上下文)
  -> StrategyEngine (加载+执行+评分)
  -> 选股结果输出

数据前提: 需要先启动后端服务执行盘后管道 (POST /api/pipeline/run) 同步行情数据。
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

# 确保 backend/ 在 sys.path 上 (package __init__.py 已处理，此处兜底)
_BACKEND_DIR = str(Path(__file__).resolve().parent.parent / "backend")
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from app.config import settings  # noqa: E402
from app.tickflow.repository import DataStore, KlineRepository  # noqa: E402
from app.services.screener import ScreenerService  # noqa: E402
from app.strategy.engine import StrategyEngine  # noqa: E402
from app.strategy import config as strategy_config  # noqa: E402

logger = logging.getLogger(__name__)

# ── 单例状态 ──────────────────────────────────────────────
_store: DataStore | None = None
_repo: KlineRepository | None = None
_screener: ScreenerService | None = None
_engine: StrategyEngine | None = None


def _init(asset_type: str = "stock") -> StrategyEngine:
    """初始化数据基础设施 (惰性单例)。

    创建 DataStore -> KlineRepository -> ScreenerService -> StrategyEngine，
    加载内置策略 + 用户自定义策略 + 实验室策略。
    """
    global _store, _repo, _screener, _engine
    if _engine is not None:
        if _screener and _screener.asset_type != asset_type:
            _screener = ScreenerService(_repo, asset_type=asset_type)
        return _engine

    _store = DataStore()
    _repo = KlineRepository(_store)

    # 同步刷新缓存 (加载 parquet -> compute_indicators)
    # 后台预热适合 FastAPI lifespan，此处同步加载保证脚本执行时数据就绪
    _repo.refresh_cache(background=False)

    _screener = ScreenerService(_repo, asset_type=asset_type)

    builtin_dir = Path(_BACKEND_DIR) / "app" / "strategy" / "builtin"
    lab_strategies_dir = Path(__file__).resolve().parent / "strategies"
    strategy_dirs = [
        builtin_dir,
        _store.data_dir / "strategies" / "custom",
        _store.data_dir / "strategies" / "ai",
        _store.data_dir / "strategies" / "composite",
        lab_strategies_dir,
    ]

    _engine = StrategyEngine(
        strategy_dirs=strategy_dirs,
        override_loader=lambda sid: strategy_config.load_override(_store.data_dir, sid),
    )

    load_errors = _engine.load_errors()
    if load_errors:
        for err in load_errors:
            logger.warning("策略加载失败: %s - %s", err["file"], err["error"])

    return _engine


def list_strategies(asset_type: str = "stock") -> list[dict]:
    """列出所有可用策略。"""
    engine = _init(asset_type)
    return [
        meta for meta in engine.list_strategies()
        if asset_type in meta.get("asset_types", ["stock"])
    ]


def run_strategy(
    strategy_id: str,
    as_of: date | None = None,
    *,
    params: dict | None = None,
    pool: list[str] | None = None,
    asset_type: str = "stock",
) -> dict:
    """运行单个策略，返回选股结果。

    Args:
        strategy_id:  策略 ID (如 "ma_golden_cross")
        as_of:        目标日期，None 则自动取最新数据日
        params:       策略参数覆盖
        pool:         限定股票池 (symbol 列表)
        asset_type:   资产类型 (stock / etf)
    """
    engine = _init(asset_type)

    if not engine.has(strategy_id):
        available = [s["id"] for s in engine.list_strategies()]
        raise ValueError(f"策略 {strategy_id!r} 不存在。可用: {available}")

    if as_of is None:
        as_of = _screener.latest_date()
    if not as_of:
        raise RuntimeError(
            "无可用数据日期。请先启动后端服务执行盘后管道同步数据:\n"
            "  cd backend && uv run uvicorn app.main:app --port 3018\n"
            "  然后访问 http://localhost:3018 或 POST /api/pipeline/run"
        )

    overrides = strategy_config.load_override(_store.data_dir, strategy_id)
    resolved_params = dict(overrides.get("params") or {})
    if params:
        resolved_params.update(params)

    context = _screener.build_strategy_context(
        engine,
        as_of,
        [strategy_id],
        params_map={strategy_id: resolved_params},
        overrides_map={strategy_id: overrides},
    )

    result = engine.run(
        strategy_id,
        context,
        pool=pool,
        params=resolved_params,
        overrides=overrides or None,
    )

    return {
        "strategy_id": strategy_id,
        "strategy_name": engine.get(strategy_id).meta.get("name", strategy_id),
        "as_of": str(as_of),
        "total": result.total,
        "elapsed_ms": round(result.elapsed_ms, 1),
        "rows": result.rows,
        "scores": result.scores,
    }


def run_file(
    file_path: str | Path,
    *,
    as_of: date | None = None,
    pool: list[str] | None = None,
    limit: int = 20,
) -> None:
    """直接运行策略文件，打印选股结果到控制台。

    供策略文件 ``if __name__ == "__main__"`` 调用:
        python strategy_lab/strategies/volume_ma_breakout.py
    """
    import importlib.util

    path = Path(file_path).resolve()
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    strategy_id = mod.META["id"]
    result = run_strategy(strategy_id, as_of=as_of, pool=pool)

    print(f"\n{'=' * 90}")
    print(f"策略: {result['strategy_name']} ({result['strategy_id']})")
    print(f"日期: {result['as_of']}  |  选出: {result['total']} 只  |  耗时: {result['elapsed_ms']}ms")
    print(f"{'=' * 90}")
    rows = result["rows"][:limit]
    print(_format_table(rows))
    if result["total"] > limit:
        print(f"\n  ... 还有 {result['total'] - limit} 条未显示 (--limit 调整)")
    print()


def run_all(
    as_of: date | None = None,
    *,
    asset_type: str = "stock",
    strategy_ids: list[str] | None = None,
) -> dict[str, dict]:
    """批量运行所有 (或指定) 策略。"""
    engine = _init(asset_type)

    if as_of is None:
        as_of = _screener.latest_date()
    if not as_of:
        raise RuntimeError("无可用数据日期。请先启动后端服务执行盘后管道同步数据。")

    all_overrides = strategy_config.list_overrides(_store.data_dir)

    if strategy_ids:
        all_ids = strategy_ids
        for sid in all_ids:
            if not engine.has(sid):
                raise ValueError(f"策略 {sid!r} 不存在")
    else:
        all_ids = [
            meta["id"]
            for meta in engine.list_strategies()
            if asset_type in meta.get("asset_types", ["stock"])
        ]

    if not all_ids:
        return {}

    params_map = {
        sid: dict((all_overrides.get(sid) or {}).get("params") or {})
        for sid in all_ids
    }
    overrides_map = {sid: all_overrides.get(sid, {}) for sid in all_ids}

    context = _screener.build_strategy_context(
        engine,
        as_of,
        all_ids,
        params_map=params_map,
        overrides_map=overrides_map,
    )

    results = engine.run_all(
        context,
        params_map=params_map,
        overrides_map=overrides_map,
        strategy_ids=all_ids,
    )

    output: dict[str, dict] = {}
    for sid, result in results.items():
        output[sid] = {
            "strategy_id": sid,
            "strategy_name": engine.get(sid).meta.get("name", sid),
            "as_of": str(as_of),
            "total": result.total,
            "elapsed_ms": round(result.elapsed_ms, 1),
            "rows": result.rows,
        }
    return output


# ── 格式化输出 ────────────────────────────────────────────

_DISPLAY_COLUMNS = [
    "symbol", "name", "close", "change_pct", "score",
    "amount", "turnover_rate", "vol_ratio_5d",
]


def _fmt_val(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if v != v or abs(v) == float("inf"):
            return ""
        return f"{v:.2f}"
    return str(v)


def _format_table(rows: list[dict], columns: list[str] | None = None) -> str:
    """将结果格式化为文本表格。"""
    if not rows:
        return "  (无命中)"

    if columns is None:
        columns = _DISPLAY_COLUMNS

    available = [c for c in columns if any(c in r for r in rows)]
    if not available:
        available = list(rows[0].keys())[:6]

    widths: dict[str, int] = {}
    for col in available:
        widths[col] = max(
            len(col),
            max(len(_fmt_val(r.get(col))) for r in rows),
        )

    header = " | ".join(col.rjust(widths[col]) for col in available)
    separator = "-+-".join("-" * widths[col] for col in available)
    lines = [header, separator]
    for r in rows:
        lines.append(" | ".join(_fmt_val(r.get(col)).rjust(widths[col]) for col in available))
    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="策略选股执行器 - 基于项目服务端数据进行策略选股",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m strategy_lab.runner --list
  python -m strategy_lab.runner --run ma_golden_cross
  python -m strategy_lab.runner --run ma_golden_cross --date 2025-01-10
  python -m strategy_lab.runner --run volume_price_surge --pool 600519.SH,000858.SZ
  python -m strategy_lab.runner --run-all --limit 5
        """,
    )
    parser.add_argument("--list", action="store_true", help="列出所有可用策略")
    parser.add_argument("--run", type=str, metavar="ID", help="运行指定策略")
    parser.add_argument("--run-all", action="store_true", help="运行所有策略")
    parser.add_argument("--date", type=str, metavar="YYYY-MM-DD", help="目标日期 (默认最新)")
    parser.add_argument("--pool", type=str, metavar="SYM,SYM,...", help="限定股票池")
    parser.add_argument("--limit", type=int, default=20, help="每策略显示前 N 条 (默认 20)")
    parser.add_argument("--asset-type", type=str, default="stock", choices=["stock", "etf"])
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.list:
        strategies = list_strategies(args.asset_type)
        print(f"\n可用策略 ({len(strategies)} 个, asset_type={args.asset_type}):")
        print("-" * 90)
        for s in strategies:
            tags = ", ".join(s.get("tags", []))
            print(f"  {s['id']:<30s} {s.get('name', ''):<12s} [{tags}]")
        print()
        return

    target_date = date.fromisoformat(args.date) if args.date else None
    pool = [s.strip() for s in args.pool.split(",")] if args.pool else None

    if args.run:
        try:
            result = run_strategy(
                args.run,
                as_of=target_date,
                pool=pool,
                asset_type=args.asset_type,
            )
        except (ValueError, RuntimeError) as e:
            print(f"错误: {e}", file=sys.stderr)
            sys.exit(1)

        print(f"\n{'=' * 90}")
        print(f"策略: {result['strategy_name']} ({result['strategy_id']})")
        print(f"日期: {result['as_of']}  |  选出: {result['total']} 只  |  耗时: {result['elapsed_ms']}ms")
        print(f"{'=' * 90}")
        rows = result["rows"][:args.limit]
        print(_format_table(rows))
        if result["total"] > args.limit:
            print(f"\n  ... 还有 {result['total'] - args.limit} 条未显示 (--limit 调整)")
        print()
        return

    if args.run_all:
        try:
            results = run_all(as_of=target_date, asset_type=args.asset_type)
        except (ValueError, RuntimeError) as e:
            print(f"错误: {e}", file=sys.stderr)
            sys.exit(1)

        if not results:
            print("无策略可执行。")
            return

        as_of = next(iter(results.values()))["as_of"]
        print(f"\n{'=' * 90}")
        print(f"批量选股  日期: {as_of}  策略数: {len(results)}")
        print(f"{'=' * 90}")

        for sid, result in results.items():
            print(f"\n  [{result['strategy_name']}] 选出 {result['total']} 只  ({result['elapsed_ms']}ms)")
            rows = result["rows"][:args.limit]
            if rows:
                print(_format_table(rows))
        print()
        return

    parser.print_help()


if __name__ == "__main__":
    main()
