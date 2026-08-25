"""stock_care_monitor 脚本解析与代码探测逻辑测试。"""
from __future__ import annotations

from scripts.stock_care_monitor import (
    entry_candidates,
    format_quote_time,
    parse_quote_payload,
    parse_watch_entries,
    select_symbols,
)


def _payload(
    *,
    name: str = "上证指数",
    code: str = "000001",
    price: str = "3342.28",
    change: str = "6.13",
    change_pct: str = "0.18",
) -> str:
    """构造腾讯接口风格的 ~ 分隔载荷。"""
    fields = [""] * 50
    fields[1], fields[2], fields[3] = name, code, price
    fields[4], fields[5], fields[6] = "3336.15", "3340.49", "386752196"
    fields[30] = "20260825150003"
    fields[31], fields[32], fields[33], fields[34] = change, change_pct, "3350.00", "3330.00"
    fields[37] = "48968925.0"
    return "~".join(fields)


def test_parse_quote_payload_basic() -> None:
    quote = parse_quote_payload("sh000001", _payload())
    assert quote is not None
    assert quote.symbol == "sh000001"
    assert quote.code == "000001"
    assert quote.name == "上证指数"
    assert quote.price == 3342.28
    assert quote.change == 6.13
    # 腾讯接口涨跌幅为百分数制: 0.18 表示 0.18%
    assert quote.change_pct == 0.18
    assert quote.open == 3340.49
    assert quote.high == 3350.00
    assert quote.low == 3330.00
    assert quote.amount == 48968925.0
    assert quote.quote_time == "15:00:03"


def test_parse_quote_payload_invalid_code() -> None:
    # 腾讯对无效代码返回 v_pv_none="1" 之类的单字段载荷
    assert parse_quote_payload("sh000510", "1") is None
    assert parse_quote_payload("sh000510", "") is None
    # 名称缺失同样视为无效
    empty_name = [""] * 50
    assert parse_quote_payload("sh000510", "~".join(empty_name)) is None


def test_parse_quote_payload_missing_optional_fields() -> None:
    # 部分字段缺失(如停牌无成交额)不阻断解析
    fields = _payload().split("~")
    fields[37] = ""
    quote = parse_quote_payload("sz399001", "~".join(fields))
    assert quote is not None
    assert quote.amount is None


def test_format_quote_time() -> None:
    assert format_quote_time("20260825093105") == "09:31:05"
    assert format_quote_time("20260825 09:31:05") == "09:31:05"
    assert format_quote_time("") == "--:--:--"


def test_parse_watch_entries() -> None:
    text = "\n000001\n  # 注释\n399001\n\nsz000510\n"
    assert parse_watch_entries(text) == ["000001", "399001", "SZ000510"]


def test_entry_candidates_prefixed() -> None:
    assert entry_candidates("SH000001") == ["sh000001"]
    assert entry_candidates("sz000001") == ["sz000001"]
    assert entry_candidates("sh688825") == ["sh688825"]


def test_entry_candidates_plain_rules() -> None:
    # 沪市个股/ETF 单市场
    assert entry_candidates("600000") == ["sh600000"]
    assert entry_candidates("688825") == ["sh688825"]
    # 深市指数/创业板 单市场
    assert entry_candidates("399001") == ["sz399001"]
    assert entry_candidates("300750") == ["sz300750"]
    # 0 开头沪深两市均有标的: 沪侧指数优先, 深侧个股兜底
    assert entry_candidates("000001") == ["sh000001", "sz000001"]
    # 非法输入无候选
    assert entry_candidates("abc") == []
    assert entry_candidates("12345") == []


def test_select_symbols_prefers_index_side() -> None:
    raw = {
        "sh000001": _payload(name="上证指数"),
        "sz000001": _payload(name="平安银行", price="10.55"),
    }
    resolved, skipped = select_symbols({"000001": ["sh000001", "sz000001"]}, raw)
    assert resolved == ["sh000001"]  # 双侧有效时优先指数侧
    assert skipped == []


def test_select_symbols_falls_back_to_valid_side() -> None:
    raw = {"sz000510": _payload(name="新金路", code="000510", price="5.12")}
    resolved, skipped = select_symbols({"000510": ["sh000510", "sz000510"]}, raw)
    assert resolved == ["sz000510"]
    assert skipped == []


def test_select_symbols_reports_unresolvable() -> None:
    resolved, skipped = select_symbols({"000510": ["sh000510", "sz000510"]}, {})
    assert resolved == []
    assert skipped == ["000510"]
