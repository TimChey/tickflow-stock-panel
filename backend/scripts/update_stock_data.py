#!/usr/bin/env python
"""手动执行盘后管道,更新本地股票行情数据。

无需启动 Web 服务,直接在命令行触发与 /api/pipeline/run 相同的盘后管道:
  维表同步 -> 日K同步 -> 除权因子 -> enriched 指标计算 -> 指数/ETF -> 分钟K -> 市场环境

用法 (从 backend/ 目录运行):
    # 正常同步 (自动判定日期范围)
    uv run python -m scripts.update_stock_data

    # 修正/补数据: 从指定日期开始拉取到今天
    uv run python -m scripts.update_stock_data --repair 2025-01-01

    # 指定数据目录
    uv run python -m scripts.update_stock_data --data-dir /path/to/data
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)


def _progress(stage: str, pct: int, msg: str, **kwargs) -> None:
    """控制台进度回调 - 与 daily_pipeline.run_now 的 on_progress 协议一致。"""
    skip_log = kwargs.get("skip_log", False)
    if skip_log and pct < 100:
        return
    logger.info("[%3d%%] %s: %s", pct, stage, msg)


def main() -> int:
    ap = argparse.ArgumentParser(description="手动更新股票行情数据 (盘后管道)")
    ap.add_argument(
        "--repair", type=str, metavar="START_DATE",
        help="修正模式: 从指定日期 (YYYY-MM-DD) 拉取到今天",
    )
    ap.add_argument(
        "--data-dir", type=str, metavar="PATH",
        help="数据目录 (默认使用项目配置)",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # 确保 backend/ 在 sys.path 上 (直接 python 运行时兜底)
    backend_dir = str(Path(__file__).resolve().parent.parent)
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    override_start_date: date | None = None
    if args.repair:
        try:
            override_start_date = date.fromisoformat(args.repair)
        except ValueError:
            logger.error("无效的日期格式: %s (应为 YYYY-MM-DD)", args.repair)
            return 1
        if override_start_date > date.today():
            logger.error("起始日期不能晚于今天: %s", override_start_date)
            return 1

    # 初始化数据层 (与 main.py lifespan 一致)
    from app.tickflow.repository import DataStore, KlineRepository

    store = DataStore(data_dir=Path(args.data_dir)) if args.data_dir else DataStore()
    repo = KlineRepository(store)

    # 加载自定义数据源 (与 main.py 一致, 失败不阻断)
    try:
        from app.data_providers import custom as custom_sources
        custom_sources.load_all()
        logger.info("自定义数据源已加载: %d 个", len(custom_sources.list_sources()))
    except Exception as e:
        logger.warning("自定义数据源加载失败 (不影响 TickFlow 基准路径): %s", e)

    # 能力探测
    from app.tickflow.policy import detect_capabilities
    capset = detect_capabilities()
    logger.info("能力探测完成: %d 个能力激活", len(capset.all()))

    # 执行盘后管道
    from app.jobs import daily_pipeline

    mode = f"修正模式 [{override_start_date} ~ 今天]" if override_start_date else "正常同步"
    logger.info("==== 开始盘后管道 (%s) ====", mode)

    try:
        result = daily_pipeline.run_now(
            repo, capset,
            on_progress=_progress,
            override_start_date=override_start_date,
        )
    except daily_pipeline.PipelineStageError as e:
        logger.warning("管道部分阶段失败: %s", "; ".join(e.errors))
        result = getattr(e, "_result", None) or {}
        _print_summary(result)
        repo.refresh_cache()
        logger.info("==== 管道完成 (部分失败), 已刷新缓存 ====")
        return 1
    except Exception as e:
        logger.exception("管道执行失败: %s", e)
        return 1

    _print_summary(result)

    # 刷新 Polars 内存缓存 (与 API 端点 /api/pipeline/run 一致)
    repo.refresh_cache()
    logger.info("==== 管道完成, 已刷新缓存 ====")
    return 0


def _print_summary(result: dict) -> None:
    """打印管道结果摘要。"""
    if not result:
        return
    lines = [
        f"  标的池: {result.get('universe_size', '?')} 只",
        f"  日K覆盖: {result.get('daily_days', '?')} 天",
        f"  除权因子: {result.get('adj_factor_symbols', '?')} 只个股",
        f"  enriched: {result.get('enriched_days', '?')} 天",
        f"  指数: {result.get('index_count', '?')} 只 / {result.get('index_daily_rows', '?')} 行",
        f"  ETF: {result.get('etf_count', '?')} 只 / {result.get('etf_daily_rows', '?')} 行",
        f"  分钟K: {result.get('minute_rows', '?')} 行",
        f"  市场环境: {result.get('regime_days', '?')} 天",
    ]
    skipped = result.get("skipped_stages", [])
    if skipped:
        lines.append(f"  跳过: {', '.join(skipped)}")
    lagging = result.get("lagging_symbols", 0)
    if lagging:
        lines.append(f"  落后标的: {lagging} 只 (>3 日未更新)")
    stage_errors = result.get("stage_errors", [])
    if stage_errors:
        lines.append(f"  阶段错误: {'; '.join(stage_errors)}")
    logger.info("管道结果:\n%s", "\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
