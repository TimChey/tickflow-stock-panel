"""策略实验室 - 用户自定义策略目录。

放入此目录的 .py 策略文件会被 StrategyEngine 自动发现并加载。
策略文件需遵循项目策略契约 (polars_expr / matrix_native / python_history_legacy)。

最简单的 polars_expr 策略只需:
  1. META 字典 (含 id / name / asset_types 等)
  2. filter(df, params) -> pl.Expr  (返回过滤表达式)

参考内置策略: backend/app/strategy/builtin/
完整开发规范: backend/app/strategy/prompts/strategy-guide.md
"""
