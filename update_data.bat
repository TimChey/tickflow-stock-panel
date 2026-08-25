cd backend

@REM # 正常同步 (自动判定日期范围)
uv run python -m scripts.update_stock_data

@REM # 修正/补数据: 从指定日期拉取到今天
@REM uv run python -m scripts.update_stock_data --repair 2025-01-01

@REM # 指定数据目录
@REM uv run python -m scripts.update_stock_data --data-dir /path/to/data