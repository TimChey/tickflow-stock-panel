#!/usr/bin/env python3
"""stock_care 清单实时行情监控脚本。

读取项目根目录 stock_care 文件中的标的代码, 通过腾讯公开行情接口
(qt.gtimg.cn) 以 HTTP 长连接(keep-alive 复用 TCP 连接)按固定间隔轮询,
在终端刷新展示实时行情。

数据口径说明(与项目主数据流隔离):
- 本脚本为独立终端工具, 结果仅用于展示, 不落盘、不进入 provider 标准化。
- 腾讯接口涨跌幅为百分数制(3.66 表示 3.66%), 与项目内部小数制契约
  不同, 不得把本脚本的数值直接搬到主数据流。

代码解释规则:
- 文件中可写带市场前缀的代码(如 sh000001 / sz000001)明确指定市场。
- 纯 6 位代码自动探测沪深两市: 0 开头优先沪市(指数侧, 如 000001=上证
  指数、000688=科创50), 其余按代码段固定市场; 仅一侧有效时自动采用
  有效侧(如 688825=科创板个股)。

用法:
    cd backend
    uv run python -m scripts.stock_care_monitor
    uv run python -m scripts.stock_care_monitor --interval 2
    uv run python -m scripts.stock_care_monitor --once
    uv run python -m scripts.stock_care_monitor --file ../stock_care
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

QT_BASE_URL = "https://qt.gtimg.cn/q="
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (stock_care_monitor)"}

# 腾讯行情字段索引(~ 分隔): 1 名称 2 代码 3 现价 4 昨收 5 今开 6 成交量(手)
# 30 时间 31 涨跌额 32 涨跌幅(百分数制) 33 最高 34 最低 37 成交额(万元)
_F_NAME, _F_CODE, _F_PRICE = 1, 2, 3
_F_PREV_CLOSE, _F_OPEN, _F_VOLUME = 4, 5, 6
_F_TIME, _F_CHANGE, _F_CHANGE_PCT = 30, 31, 32
_F_HIGH, _F_LOW, _F_AMOUNT = 33, 34, 37
_MIN_FIELDS = 35  # 最高/最低之后才算完整行情; 无效代码只有 1 个字段

ANSI_RESET = "\033[0m"
ANSI_RED = "\033[31m"  # A 股习惯: 红涨
ANSI_GREEN = "\033[32m"  # 绿跌
ANSI_DIM = "\033[2m"
ANSI_BOLD = "\033[1m"


@dataclass
class Quote:
    """单只标的的实时快照(仅本脚本展示用)。"""

    symbol: str
    code: str
    name: str
    price: float | None
    change: float | None
    change_pct: float | None  # 百分数制: 3.66 表示 3.66%
    open: float | None
    high: float | None
    low: float | None
    volume: float | None  # 手
    amount: float | None  # 万元
    quote_time: str  # HH:MM:SS


def parse_quote_payload(symbol: str, payload: str) -> Quote | None:
    """解析腾讯接口单条 `v_xxx="..."` 载荷; 无效代码返回 None。"""
    fields = payload.split("~")
    if len(fields) < _MIN_FIELDS or not fields[_F_NAME]:
        return None

    def to_float(idx: int) -> float | None:
        try:
            return float(fields[idx])
        except (IndexError, ValueError):
            return None

    return Quote(
        symbol=symbol,
        code=fields[_F_CODE] or symbol[2:],
        name=fields[_F_NAME].strip(),
        price=to_float(_F_PRICE),
        change=to_float(_F_CHANGE),
        change_pct=to_float(_F_CHANGE_PCT),
        open=to_float(_F_OPEN),
        high=to_float(_F_HIGH),
        low=to_float(_F_LOW),
        volume=to_float(_F_VOLUME),
        amount=to_float(_F_AMOUNT),
        quote_time=format_quote_time(fields[_F_TIME]),
    )


def format_quote_time(raw: str) -> str:
    """把腾讯时间字段(如 20260825150003)格式化为 HH:MM:SS。"""
    digits = re.sub(r"\D", "", raw)
    if len(digits) >= 6:
        hhmmss = digits[-6:]
        return f"{hhmmss[:2]}:{hhmmss[2:4]}:{hhmmss[4:6]}"
    return "--:--:--"


def parse_watch_entries(text: str) -> list[str]:
    """解析 stock_care 文本: 每行一个代码, 忽略空行与 # 注释。"""
    entries: list[str] = []
    for raw in text.splitlines():
        entry = raw.strip().upper()
        if not entry or entry.startswith("#"):
            continue
        entries.append(entry)
    return entries


def entry_candidates(entry: str) -> list[str]:
    """单个条目 -> 按优先级排列的候选 symbol 列表。

    0 开头代码沪深两市均有标的(沪侧为指数、深侧为个股), 默认指数优先,
    需要个股时在 stock_care 中写 sz 前缀。
    """
    entry = entry.strip().lower()
    if len(entry) == 8 and entry[:2] in ("sh", "sz") and entry[2:].isdigit():
        return [entry]
    if len(entry) != 6 or not entry.isdigit():
        return []
    if entry.startswith(("60", "68", "51", "58")):
        return ["sh" + entry]  # 沪市个股 / ETF
    if entry.startswith(("399", "30", "15", "16", "12", "1")):
        return ["sz" + entry]  # 深市指数 / 创业板 / ETF / 可转债
    if entry.startswith("0"):
        return ["sh" + entry, "sz" + entry]  # 沪侧指数优先, 深侧个股兜底
    return ["sh" + entry, "sz" + entry]


def fetch_raw(client: httpx.Client, symbols: list[str]) -> dict[str, str]:
    """一次请求拉取全部 symbol 的原始载荷(GBK 编码)。"""
    resp = client.get(QT_BASE_URL + ",".join(symbols), headers=REQUEST_HEADERS)
    resp.raise_for_status()
    text = resp.content.decode("gbk", errors="replace")
    return dict(re.findall(r'v_([A-Za-z0-9_]+)="([^"]*)"', text))


def select_symbols(
    candidate_map: dict[str, list[str]], raw: dict[str, str]
) -> tuple[list[str], list[str]]:
    """从原始载荷中为每个条目选定第一个有效的 symbol。"""
    resolved: list[str] = []
    skipped: list[str] = []
    for entry, symbols in candidate_map.items():
        for symbol in symbols:
            if parse_quote_payload(symbol, raw.get(symbol, "")) is not None:
                resolved.append(symbol)
                break
        else:
            skipped.append(entry)
    return resolved, skipped


def resolve_symbols(
    client: httpx.Client, entries: list[str]
) -> tuple[list[str], list[str], dict[str, Quote]]:
    """探测每个条目的有效 symbol 并返回首个快照(用于启动确认)。"""
    candidate_map = {entry: entry_candidates(entry) for entry in entries}
    all_symbols = [s for symbols in candidate_map.values() for s in symbols]
    raw = fetch_raw(client, all_symbols)
    resolved, skipped = select_symbols(candidate_map, raw)
    previews: dict[str, Quote] = {}
    for symbol in resolved:
        quote = parse_quote_payload(symbol, raw.get(symbol, ""))
        if quote is not None:
            previews[symbol] = quote
    return resolved, skipped, previews


def fetch_quotes(client: httpx.Client, symbols: list[str]) -> list[Quote]:
    """拉取并解析一轮实时行情。"""
    raw = fetch_raw(client, symbols)
    quotes: list[Quote] = []
    for symbol in symbols:
        quote = parse_quote_payload(symbol, raw.get(symbol, ""))
        if quote is not None:
            quotes.append(quote)
    return quotes


def display_width(text: str) -> int:
    """按东亚宽度计算终端占位(中文占 2 列)。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def pad(text: str, width: int) -> str:
    """左侧对齐补空格到指定显示宽度; 超长截断。"""
    if display_width(text) > width:
        out = ""
        for ch in text:
            if display_width(out + ch) > width - 1:
                break
            out += ch
        return out + "…"
    return text + " " * (width - display_width(text))


def pad_left(text: str, width: int) -> str:
    """右侧对齐补空格到指定显示宽度。"""
    return " " * max(0, width - display_width(text)) + text


def fmt_num(value: float | None, decimals: int = 2, sign: bool = False) -> str:
    if value is None:
        return "-"
    return f"{value:+.{decimals}f}" if sign else f"{value:.{decimals}f}"


def colorize(text: str, change_pct: float | None) -> str:
    """按涨跌幅着色: 红涨绿跌(仅展示层约定)。"""
    if change_pct is None or change_pct == 0:
        return text
    color = ANSI_RED if change_pct > 0 else ANSI_GREEN
    return f"{color}{text}{ANSI_RESET}"


# 列定义: (标题, 宽度, 是否右对齐)
COLUMNS = (
    ("代码", 6, False),
    ("名称", 9, False),
    ("现价", 9, True),
    ("涨跌", 8, True),
    ("涨跌幅", 8, True),
    ("今开", 9, True),
    ("最高", 9, True),
    ("最低", 9, True),
    ("成交额(亿)", 10, True),
    ("时间", 8, False),
)
_TABLE_WIDTH = sum(w for _, w, _ in COLUMNS) + 2 * (len(COLUMNS) - 1)


def render_header_lines(interval: float) -> list[str]:
    now = datetime.now().strftime("%H:%M:%S")
    title = (
        f"{ANSI_BOLD}stock_care 实时行情{ANSI_RESET}  "
        f"刷新 {interval:g}s  长连接 keep-alive  {now}"
    )
    cells = []
    for name, width, right in COLUMNS:
        cell = pad_left(name, width) if right else pad(name, width)
        cells.append(ANSI_DIM + cell + ANSI_RESET)
    return [title, "  ".join(cells), ANSI_DIM + "-" * _TABLE_WIDTH + ANSI_RESET]


def render_quote_line(quote: Quote) -> str:
    amount_yi = f"{quote.amount / 10000:.1f}" if quote.amount is not None else "-"
    # 涨跌幅缺失时不拼接 % 后缀, 避免出现 "-%"
    change_pct = (
        f"{quote.change_pct:+.2f}%" if quote.change_pct is not None else "-"
    )
    values = (
        quote.code,
        quote.name,
        fmt_num(quote.price),
        fmt_num(quote.change, sign=True),
        change_pct,
        fmt_num(quote.open),
        fmt_num(quote.high),
        fmt_num(quote.low),
        amount_yi,
        quote.quote_time,
    )
    # 现价/涨跌/涨跌幅随涨跌着色, 其余列保持中性
    colored_idx = {2, 3, 4}
    cells = []
    for idx, ((_, width, right), value) in enumerate(zip(COLUMNS, values, strict=True)):
        # 先对齐再着色: ANSI 转义序列不能被计入显示宽度, 否则列会错位
        cell = pad_left(value, width) if right else pad(value, width)
        if idx in colored_idx:
            cell = colorize(cell, quote.change_pct)
        cells.append(cell)
    return "  ".join(cells)


def render_screen(
    quotes: list[Quote], interval: float, skipped: list[str], status: str
) -> None:
    """整屏重绘(光标归位 + 逐行覆盖, 避免清屏闪烁)。"""
    lines = render_header_lines(interval)
    if quotes:
        lines.extend(render_quote_line(q) for q in quotes)
    else:
        lines.append(ANSI_DIM + "(暂无数据)" + ANSI_RESET)
    lines.append(ANSI_DIM + "-" * _TABLE_WIDTH + ANSI_RESET)
    if skipped:
        lines.append(f"{ANSI_DIM}未识别: {', '.join(skipped)}{ANSI_RESET}")
    lines.append(status)
    sys.stdout.write("\033[H" + "".join(line + "\033[K\n" for line in lines) + "\033[J")
    sys.stdout.flush()


def truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def run(args: argparse.Namespace) -> int:
    watch_path: Path = args.file
    if not watch_path.is_file():
        logger.error("清单文件不存在: %s", watch_path)
        return 2
    entries = parse_watch_entries(watch_path.read_text(encoding="utf-8"))
    if not entries:
        logger.error("清单为空: %s", watch_path)
        return 2
    if args.interval <= 0:
        logger.error("--interval 必须大于 0")
        return 2

    if sys.platform == "win32":
        # Windows 传统控制台需触发一次终端初始化才启用 ANSI 转义支持
        os.system("")

    timeout = httpx.Timeout(5.0, connect=3.0)
    with httpx.Client(timeout=timeout) as client:  # 连接池复用 keep-alive 长连接
        try:
            resolved, skipped, previews = resolve_symbols(client, entries)
        except httpx.HTTPError as exc:
            logger.error("行情接口探测失败: %s", exc)
            return 1
        if not resolved:
            logger.error("清单中没有任何有效代码: %s", ", ".join(entries))
            return 1
        for symbol in resolved:
            preview = previews.get(symbol)
            name = preview.name if preview else "?"
            logger.info("%s -> %s %s", symbol[2:], symbol, name)
        if skipped:
            logger.warning("未识别代码(两市均无行情): %s", ", ".join(skipped))

        symbols = resolved
        fail_count = 0
        last_error = ""
        last_quotes: list[Quote] = []
        if args.once:
            try:
                quotes = fetch_quotes(client, symbols)
            except httpx.HTTPError as exc:
                logger.error("行情接口请求失败: %s", exc)
                return 1
            render_screen(quotes, args.interval, skipped, "单次查询完成")
            return 0

        sys.stdout.write("\033[?25l")  # 隐藏光标, 防止重绘闪烁
        try:
            while True:
                cycle_start = time.monotonic()
                try:
                    quotes = fetch_quotes(client, symbols)
                    # 失败期间保留上一帧有效数据, 避免表格闪空
                    last_quotes = quotes or last_quotes
                    fail_count = 0
                    last_error = ""
                except httpx.HTTPError as exc:
                    fail_count += 1
                    last_error = truncate(str(exc), 80)
                now = datetime.now().strftime("%H:%M:%S")
                if fail_count:
                    status = f"{now}  连续失败 {fail_count} 次: {last_error}"
                else:
                    status = f"{now}  Ctrl+C 退出"
                render_screen(last_quotes, args.interval, skipped, status)
                elapsed = time.monotonic() - cycle_start
                if (delay := args.interval - elapsed) > 0:
                    time.sleep(delay)
        except KeyboardInterrupt:
            pass
        finally:
            sys.stdout.write("\033[?25h\n")  # 恢复光标
            sys.stdout.flush()
    return 0


def main() -> int:
    default_file = Path(__file__).resolve().parents[2] / "stock_care"
    parser = argparse.ArgumentParser(
        description="实时监控 stock_care 清单行情(腾讯接口, HTTP 长连接)"
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=default_file,
        help=f"清单文件路径 (默认: {default_file})",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="刷新间隔秒数 (默认: 2)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="只查询一次并退出 (不做循环刷新)",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
