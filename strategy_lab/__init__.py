"""策略实验室 - 独立策略开发与选股执行环境。

复用后端数据基础设施 (DataStore / KlineRepository / ScreenerService / StrategyEngine)，
提供独立的策略编写、调试和批量选股执行能力。

使用方式:
    # 列出所有可用策略
    python -m strategy_lab.runner --list

    # 运行单个策略
    python -m strategy_lab.runner --run ma_golden_cross

    # 运行单个策略并指定日期
    python -m strategy_lab.runner --run ma_golden_cross --date 2025-01-10

    # 运行所有策略
    python -m strategy_lab.runner --run-all

    # 指定股票池
    python -m strategy_lab.runner --run volume_price_surge --pool 600519.SH,000858.SZ
"""
from __future__ import annotations

import sys
from pathlib import Path

# 将 backend/ 加入 sys.path，使 from app.xxx import yyy 可用
_BACKEND_DIR = str(Path(__file__).resolve().parent.parent / "backend")
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)
