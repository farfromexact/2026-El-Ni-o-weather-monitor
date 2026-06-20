from __future__ import annotations

import datetime as dt
import json
import math
import os

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


APP_TZ = ZoneInfo("Asia/Shanghai")
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
NOAA_ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
IFIND_ACCESS_TOKEN_URL = "https://quantapi.51ifind.com/api/v1/get_access_token"
IFIND_HISTORY_URL = "https://quantapi.51ifind.com/api/v1/cmd_history_quotation"
IFIND_DEFAULT_HTTP_TIMEOUT = 25
IFIND_DEFAULT_REFRESH_TOKEN = ""
IFIND_DEFAULT_USERNAME = ""
IFIND_DEFAULT_PASSWORD = ""
SIGNAL_LOG_DIR = "logs"
SIGNAL_LOG_PATH = os.path.join(SIGNAL_LOG_DIR, "signal_run_log.csv")
POSITION_STATE_PATH = os.path.join(SIGNAL_LOG_DIR, "position_state.csv")
POST_TRADE_NOTES_PATH = os.path.join(SIGNAL_LOG_DIR, "post_trade_notes.csv")
LAST_GOOD_CACHE_DIR = os.path.join(SIGNAL_LOG_DIR, "cache")
PUBLIC_DATA_DIR = "public_data"
PUBLIC_PRICE_DIR = os.path.join(PUBLIC_DATA_DIR, "price")
PRICE_SOURCE_IFIND = "iFinD"
PRICE_SOURCE_AKSHARE = "AKShare"
PRICE_SOURCE_STATIC = "Static Snapshot"
DAILY_WEATHER_FIELDS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "et0_fao_evapotranspiration",
]


@dataclass(frozen=True)
class RuleSettings:
    forecast_days: int
    baseline_years: int
    dry_ratio_trigger: float
    wet_ratio_trigger: float
    heat_trigger_c: float
    weather_trigger: int
    price_trigger: int
    build_trigger: int


COMMODITIES: dict[str, dict[str, Any]] = {
    "白糖 SR": {
        "symbol": "SR0",
        "ifind_symbol": "SRZL.CZC",
        "unit": "元/吨",
        "tick": 1,
        "contract_multiplier": 10,
        "default_margin_rate": 0.12,
        "active_contract_months": [1, 5, 9],
        "weather_weight": 0.55,
        "price_weight": 0.45,
        "thesis": "海外主产区干旱或收榨期降雨扰动抬升供应风险，内盘价格突破或回踩确认后才触发建仓。",
        "case_tags": ["2010 国际糖短缺", "2015/16 强厄尔尼诺", "2019/20 泰国干旱"],
    },
    "棕榈油 P": {
        "symbol": "P0",
        "ifind_symbol": "PZL.DCE",
        "unit": "元/吨",
        "tick": 2,
        "contract_multiplier": 10,
        "default_margin_rate": 0.12,
        "active_contract_months": [1, 5, 9],
        "weather_weight": 0.60,
        "price_weight": 0.40,
        "thesis": "印尼和马来西亚偏干会通过 3-9 个月滞后影响单产；过量降雨更偏短期收割和物流扰动。",
        "case_tags": ["2015/16 强厄尔尼诺", "2019 印尼干旱/烟霾", "2021/22 供需错配"],
    },
}


WEATHER_REGIONS: list[dict[str, Any]] = [
    {
        "commodity": "白糖 SR",
        "region": "巴西中南部 Ribeirao Preto",
        "country": "Brazil",
        "lat": -21.17,
        "lon": -47.81,
        "weight": 0.28,
        "dry": 1.0,
        "wet": 0.65,
        "note": "巴西中南部决定全球出口糖边际，干旱伤单产，收榨期过雨影响压榨节奏。",
    },
    {
        "commodity": "白糖 SR",
        "region": "印度 Maharashtra/Pune",
        "country": "India",
        "lat": 18.52,
        "lon": 73.86,
        "weight": 0.18,
        "dry": 1.0,
        "wet": 0.35,
        "note": "印度西部甘蔗区对季风分布敏感，连续偏干会提高减产和出口政策风险。",
    },
    {
        "commodity": "白糖 SR",
        "region": "印度 Uttar Pradesh/Lucknow",
        "country": "India",
        "lat": 26.85,
        "lon": 80.95,
        "weight": 0.14,
        "dry": 0.85,
        "wet": 0.30,
        "note": "印度北部产区，偏干叠加高温会削弱甘蔗恢复。",
    },
    {
        "commodity": "白糖 SR",
        "region": "泰国 Nakhon Sawan",
        "country": "Thailand",
        "lat": 15.70,
        "lon": 100.13,
        "weight": 0.18,
        "dry": 1.0,
        "wet": 0.45,
        "note": "泰国是主要出口国，降雨不足常对应下一榨季供应收缩预期。",
    },
    {
        "commodity": "白糖 SR",
        "region": "广西 南宁",
        "country": "China",
        "lat": 22.82,
        "lon": 108.32,
        "weight": 0.14,
        "dry": 0.85,
        "wet": 0.45,
        "note": "国内甘蔗主产区，异常天气会影响内盘基差和产业情绪。",
    },
    {
        "commodity": "白糖 SR",
        "region": "云南 普洱",
        "country": "China",
        "lat": 22.79,
        "lon": 100.97,
        "weight": 0.08,
        "dry": 0.80,
        "wet": 0.35,
        "note": "云南甘蔗区，偏干和高温会影响单产恢复。",
    },
    {
        "commodity": "棕榈油 P",
        "region": "印尼 Riau/Pekanbaru",
        "country": "Indonesia",
        "lat": 0.51,
        "lon": 101.45,
        "weight": 0.28,
        "dry": 1.0,
        "wet": 0.25,
        "note": "苏门答腊核心油棕区，偏干对未来果串产量有滞后影响。",
    },
    {
        "commodity": "棕榈油 P",
        "region": "印尼 Central Kalimantan",
        "country": "Indonesia",
        "lat": -2.21,
        "lon": 113.92,
        "weight": 0.20,
        "dry": 1.0,
        "wet": 0.25,
        "note": "加里曼丹油棕扩张区，连续偏干会强化产量担忧。",
    },
    {
        "commodity": "棕榈油 P",
        "region": "印尼 West Kalimantan",
        "country": "Indonesia",
        "lat": -0.03,
        "lon": 109.34,
        "weight": 0.14,
        "dry": 0.95,
        "wet": 0.30,
        "note": "西加里曼丹降雨异常会影响收果和物流。",
    },
    {
        "commodity": "棕榈油 P",
        "region": "马来西亚 Sabah/Sandakan",
        "country": "Malaysia",
        "lat": 5.84,
        "lon": 118.12,
        "weight": 0.16,
        "dry": 0.95,
        "wet": 0.35,
        "note": "沙巴是马来西亚关键产区，异常天气容易引发行情跟踪。",
    },
    {
        "commodity": "棕榈油 P",
        "region": "马来西亚 Sarawak/Miri",
        "country": "Malaysia",
        "lat": 4.40,
        "lon": 113.99,
        "weight": 0.12,
        "dry": 0.90,
        "wet": 0.35,
        "note": "砂拉越产区，过雨偏短期扰动，偏干偏中期产量风险。",
    },
    {
        "commodity": "棕榈油 P",
        "region": "马来西亚 Johor",
        "country": "Malaysia",
        "lat": 1.49,
        "lon": 103.76,
        "weight": 0.10,
        "dry": 0.80,
        "wet": 0.30,
        "note": "马来半岛南部产区，用于补充全马天气观察。",
    },
]


HISTORICAL_CASES = [
    {
        "commodity": "白糖 SR",
        "case": "2010 国际糖短缺",
        "weather": "印度和巴西供应偏紧叠加全球库存低位，价格对主产区天气和出口节奏非常敏感。",
        "entry_rule": "天气风险先抬升，价格随后站上 60 日均线并突破 20 日高点，回踩 20 日均线不破是更稳的二次入场。",
        "failure_rule": "如果天气风险降温且价格跌回 60 日均线下方，供应主题通常已经被市场削弱。",
    },
    {
        "commodity": "白糖 SR",
        "case": "2015/16 强厄尔尼诺",
        "weather": "巴西、印度、泰国天气扰动强化全球糖减产预期，趋势行情一般不是第一天消息就结束。",
        "entry_rule": "天气分数高于 60 后，等待收盘价突破 20 日高点；若突破后偏离 20 日均线超过 1.5 倍 ATR，优先等回踩。",
        "failure_rule": "突破后 3-5 个交易日内无法维持在突破位上方，且持仓没有增加，应降低仓位。",
    },
    {
        "commodity": "白糖 SR",
        "case": "2019/20 泰国干旱",
        "weather": "泰国干旱使下一榨季产量预期下修，交易机会来自天气主题和价格趋势共振。",
        "entry_rule": "主产区干旱分数连续维持高位，价格回踩 20 日均线企稳可分批试仓。",
        "failure_rule": "降雨修复或价格跌破 60 日均线，说明供应溢价在退潮。",
    },
    {
        "commodity": "棕榈油 P",
        "case": "2015/16 强厄尔尼诺",
        "weather": "东南亚偏干对油棕单产有滞后影响，价格常在产量数据恶化前开始计入风险。",
        "entry_rule": "印尼/马来天气分数高于 65，价格站上 60 日均线后，突破 20 日高点是第一触发，回踩 20 日均线是第二触发。",
        "failure_rule": "天气分数跌破 45 且价格跌回 60 日均线下，说明滞后减产预期不足以支撑趋势。",
    },
    {
        "commodity": "棕榈油 P",
        "case": "2019 印尼干旱/烟霾",
        "weather": "苏门答腊和加里曼丹偏干容易强化未来产量下降预期，同时也会影响市场风险偏好。",
        "entry_rule": "天气主题先行，价格确认必须跟上：收盘价高于 20/60 日均线，成交或持仓至少有一个放大。",
        "failure_rule": "只涨消息、不放量不增仓，或者价格回落至 20 日均线下方，不宜加仓。",
    },
    {
        "commodity": "棕榈油 P",
        "case": "2021/22 供需错配",
        "weather": "天气、劳工、政策和植物油联动共同驱动，单一因子容易误判，需要价格确认过滤。",
        "entry_rule": "综合分高于 75 时只给试仓，不一次打满；突破后用 ATR 止损，回踩不破再加。",
        "failure_rule": "政策利空或外盘植物油走弱导致内盘跌破 60 日均线，要优先控制风险。",
    },
]


CASE_REPLAY_CONFIG: dict[tuple[str, str], dict[str, str]] = {
    ("白糖 SR", "2010 国际糖短缺"): {
        "start": "2009-01-01",
        "narrative_trigger": "2009-05-01",
        "end": "2011-02-28",
        "note": "印度减产和全球库存低位开始被交易，观察价格何时给出趋势确认。",
    },
    ("白糖 SR", "2015/16 强厄尔尼诺"): {
        "start": "2015-01-01",
        "narrative_trigger": "2015-07-01",
        "end": "2016-12-31",
        "note": "ENSO 与主产区天气扰动进入交易窗口，回放价格确认是否滞后。",
    },
    ("白糖 SR", "2019/20 泰国干旱"): {
        "start": "2019-01-01",
        "narrative_trigger": "2019-09-01",
        "end": "2020-06-30",
        "note": "泰国干旱减产预期发酵，观察回踩和突破哪一个先出现。",
    },
    ("棕榈油 P", "2015/16 强厄尔尼诺"): {
        "start": "2015-01-01",
        "narrative_trigger": "2015-07-01",
        "end": "2017-06-30",
        "note": "强厄尔尼诺对油棕单产有滞后影响，价格通常先于产量数据反应。",
    },
    ("棕榈油 P", "2019 印尼干旱/烟霾"): {
        "start": "2019-01-01",
        "narrative_trigger": "2019-08-01",
        "end": "2020-06-30",
        "note": "印尼偏干和烟霾强化中期产量担忧，回放价格是否同步确认。",
    },
    ("棕榈油 P", "2021/22 供需错配"): {
        "start": "2021-01-01",
        "narrative_trigger": "2021-09-01",
        "end": "2022-06-30",
        "note": "政策、劳工和植物油联动驱动，验证单纯价格确认的有效性。",
    },
}

SEASONAL_WINDOWS: dict[str, list[dict[str, Any]]] = {
    "白糖 SR": [
        {
            "name": "巴西中南部压榨期",
            "months": {4, 5, 6, 7, 8, 9, 10, 11},
            "importance": "高",
            "bias": "收榨期过雨会拖慢压榨和出口节奏；持续偏干则更偏下一季单产风险。",
        },
        {
            "name": "印度季风与甘蔗恢复期",
            "months": {6, 7, 8, 9},
            "importance": "高",
            "bias": "季风降雨分布决定甘蔗恢复和政府出口政策弹性，连续偏干更容易形成供应溢价。",
        },
        {
            "name": "泰国/中国甘蔗生长期",
            "months": {5, 6, 7, 8, 9, 10},
            "importance": "中",
            "bias": "偏干和高温影响下榨季单产，国内产区异常天气更容易影响内盘情绪和基差。",
        },
    ],
    "棕榈油 P": [
        {
            "name": "东南亚偏干累积窗口",
            "months": {5, 6, 7, 8, 9, 10},
            "importance": "高",
            "bias": "印尼/马来偏干对油棕单产通常有 3-9 个月滞后，适合监控中期减产预期。",
        },
        {
            "name": "马来/印尼雨季收割物流窗口",
            "months": {11, 12, 1, 2},
            "importance": "中",
            "bias": "过量降雨更偏短期收果、运输和库存扰动，和干旱减产逻辑要分开看。",
        },
        {
            "name": "季节性产量高峰观察期",
            "months": {8, 9, 10, 11},
            "importance": "中",
            "bias": "若高峰期产量不强，市场更容易放大此前天气或政策因素。",
        },
    ],
}

st.set_page_config(
    page_title="天气驱动的白糖/棕榈油多头监控",
    page_icon="🌦️",
    layout="wide",
)


def inject_style() -> None:
    st.markdown(
        """
        <style>
        .main .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
        .signal-box {
            border: 1px solid #d9dee8;
            border-radius: 8px;
            padding: 14px 16px;
            background: #ffffff;
            min-height: 126px;
        }
        .signal-title {font-size: 0.9rem; color: #4b5563; margin-bottom: 8px;}
        .signal-action {font-size: 1.18rem; font-weight: 700; color: #111827;}
        .signal-note {font-size: 0.86rem; color: #4b5563; margin-top: 8px; line-height: 1.45;}
        .small-muted {font-size: 0.84rem; color: #6b7280;}
        .risk-callout {
            border-left: 4px solid #b45309;
            padding: 10px 12px;
            background: #fff7ed;
            color: #7c2d12;
            border-radius: 4px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def today_china() -> dt.date:
    return dt.datetime.now(APP_TZ).date()
def now_china() -> dt.datetime:
    return dt.datetime.now(APP_TZ)


def latest_price_date_from_frame(frame: pd.DataFrame) -> dt.date | None:
    if frame is None or frame.empty or "date" not in frame.columns:
        return None
    latest_value = pd.to_datetime(frame["date"], errors="coerce").max()
    if pd.isna(latest_value):
        return None
    return latest_value.date()


def classify_market_session(
    anchor: dt.date,
    price_frame: pd.DataFrame,
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    current = now or now_china()
    latest_date = latest_price_date_from_frame(price_frame)
    stale_days = None if latest_date is None else max(0, (anchor - latest_date).days)

    def result(status: str, label: str, action_allowed: bool, close_confirmed: bool, reason: str) -> dict[str, Any]:
        return {
            "status": status,
            "label": label,
            "action_allowed": action_allowed,
            "close_confirmed": close_confirmed,
            "latest_price_date": latest_date,
            "stale_days": stale_days,
            "now": current,
            "reason": reason,
        }

    if latest_date is None:
        return result("stale-session", "行情日期缺失", False, False, "无法识别最新行情日，禁止行动语言。")

    if anchor.weekday() >= 5:
        return result("weekend/holiday", "周末/节假日", False, False, "当前锚定日不是常规交易日，只允许复盘和预案。")

    if latest_date < anchor:
        return result("stale-session", "行情滞后", False, False, f"最新行情日 {latest_date.isoformat()} 早于信号日 {anchor.isoformat()}。")

    if current.date() != anchor:
        return result("close-confirmed", "收盘已确认", True, True, f"信号日 {anchor.isoformat()} 已有当日或更新行情。")

    current_time = current.time()
    if current_time < dt.time(9, 0):
        return result("pre-open", "盘前计划", False, False, "尚未开盘，先生成预案，不允许建仓行动语言。")
    if current_time < dt.time(15, 0):
        return result("intraday", "盘中观察", False, False, "盘中价格未收盘确认，等待日线确认。")
    if current_time < dt.time(15, 15):
        return result("closing-window", "收盘等待确认", False, False, "收盘后数据可能尚未落库，等待确认窗口结束。")
    return result("close-confirmed", "收盘已确认", True, True, "当日收盘数据已进入确认窗口之后。")


def apply_market_session_guard(action: str, note: str, session: dict[str, Any]) -> tuple[str, str]:
    if action in {"信号不可用", "数据不足"}:
        return action, note
    if session.get("action_allowed", False):
        return action, note
    actionable = action in {"开始试仓", "待人工复核", "限额试仓", "限额加仓", "组合限仓"}
    if not actionable:
        return action, f"{note} 交易时段提示：{session.get('label', 'n/a')}；{session.get('reason', '')}"

    status = session.get("status", "")
    if status == "pre-open":
        guarded_action = "盘前计划"
    elif status in {"intraday", "closing-window"}:
        guarded_action = "等待收盘确认"
    elif status == "weekend/holiday":
        guarded_action = "非交易日复核"
    elif status == "stale-session":
        guarded_action = "行情陈旧复核"
    else:
        guarded_action = "等待时段确认"
    return guarded_action, f"{note} 交易时段闸门：{session.get('label', 'n/a')}，{session.get('reason', '')}"


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    if value is None or math.isnan(float(value)):
        return float("nan")
    return max(low, min(high, float(value)))


def pct_text(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.0%}"


def number_text(value: float | None, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:,.{digits}f}"


def same_month_day(base: dt.date, year: int) -> dt.date:
    while True:
        try:
            return dt.date(year, base.month, base.day)
        except ValueError:
            base = base - dt.timedelta(days=1)


def same_month_day(base: dt.date, year: int) -> dt.date:
    while True:
        try:
            return dt.date(year, base.month, base.day)
        except ValueError:
            base = base - dt.timedelta(days=1)


def cache_key_slug(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff._-]+", "_", text)
    text = text.strip("_")
    return text or "default"


def cache_paths(namespace: str, key: str) -> tuple[str, str]:
    slug = cache_key_slug(key)
    base = os.path.join(LAST_GOOD_CACHE_DIR, cache_key_slug(namespace))
    return os.path.join(base, f"{slug}.csv"), os.path.join(base, f"{slug}.json")


def save_last_good_frame(namespace: str, key: str, frame: pd.DataFrame, metadata: dict[str, Any] | None = None) -> None:
    if frame is None or frame.empty:
        return
    csv_path, meta_path = cache_paths(namespace, key)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    meta = dict(metadata or {})
    meta.update(
        {
            "namespace": namespace,
            "key": key,
            "saved_at": dt.datetime.now(APP_TZ).isoformat(timespec="seconds"),
            "rows": int(len(frame)),
        }
    )
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2, default=str)


def load_last_good_frame(namespace: str, key: str) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    csv_path, meta_path = cache_paths(namespace, key)
    if not os.path.exists(csv_path):
        return pd.DataFrame(), None
    try:
        frame = pd.read_csv(csv_path, keep_default_na=False)
        for column in ("date", "period"):
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column], errors="coerce")
        meta: dict[str, Any] = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
        saved_at = parse_cache_saved_at(meta.get("saved_at"))
        stale_days = (today_china() - saved_at.date()).days if saved_at else None
        meta["stale_days"] = stale_days
        frame.attrs["last_good_cache"] = True
        frame.attrs["cache_namespace"] = namespace
        frame.attrs["cache_key"] = key
        frame.attrs["cache_saved_at"] = meta.get("saved_at", "")
        frame.attrs["cache_stale_days"] = stale_days
        return frame, meta
    except Exception:
        return pd.DataFrame(), None


def parse_cache_saved_at(value: Any) -> dt.datetime | None:
    if not value:
        return None
    try:
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return None
        if parsed.tzinfo is None:
            return parsed.to_pydatetime().replace(tzinfo=APP_TZ)
        return parsed.to_pydatetime().astimezone(APP_TZ)
    except Exception:
        return None


def cached_data_message(label: str, meta: dict[str, Any] | None, original_error: Any = None) -> str:
    meta = meta or {}
    saved_at = meta.get("saved_at", "未知时间")
    stale_days = meta.get("stale_days")
    stale_text = "n/a" if stale_days is None else f"{stale_days} 天"
    suffix = f"；原始错误：{original_error}" if original_error else ""
    return f"{label} 使用 last-good 缓存（保存于 {saved_at}，陈旧 {stale_text}）{suffix}"


def mark_frame_cache_source(frame: pd.DataFrame, source: str, message: str = "") -> pd.DataFrame:
    frame.attrs["data_source_status"] = source
    if message:
        frame.attrs["data_source_message"] = message
    return frame

def lookback_days_for_months(months: int) -> int:
    return max(30, int(float(months) * 30.4375))

def request_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    response = requests.get(url, params=params, timeout=25)
    response.raise_for_status()
    return response.json()


def request_json_post(url: str, json_body: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
    response = requests.post(url, json=json_body, headers=headers, timeout=IFIND_DEFAULT_HTTP_TIMEOUT)
    response.raise_for_status()
    return response.json()


def secret_env_value(*keys: str) -> str:
    for key in keys:
        try:
            value = st.secrets.get(key, "")
            if value:
                return str(value).strip()
        except Exception:
            pass
    for key in keys:
        value = os.getenv(key)
        if value:
            return value.strip()
    return ""

def has_ifind_credentials(refresh_token: str, username: str, password: str) -> bool:
    return bool(refresh_token or (username and password))


def is_streamlit_cloud_runtime() -> bool:
    markers = (
        "STREAMLIT_SHARING_MODE",
        "STREAMLIT_CLOUD",
        "STREAMLIT_RUNTIME_ENV",
    )
    return any(bool(os.getenv(key)) for key in markers)


def public_price_paths(commodity_name: str) -> tuple[str, str]:
    slug = cache_key_slug(commodity_name)
    return os.path.join(PUBLIC_PRICE_DIR, f"{slug}.csv"), os.path.join(PUBLIC_PRICE_DIR, f"{slug}.json")


def public_price_snapshot_exists() -> bool:
    return all(os.path.exists(public_price_paths(name)[0]) for name in COMMODITIES)


def resolve_price_source(refresh_token: str, username: str, password: str) -> str:
    requested = secret_env_value("APP_PRICE_SOURCE", "PRICE_SOURCE", "MARKET_DATA_MODE").strip().lower()
    if requested in {"static", "snapshot", "public", "public_snapshot"}:
        return PRICE_SOURCE_STATIC
    if requested in {"akshare", "sina"}:
        return PRICE_SOURCE_AKSHARE
    if requested in {"ifind", "live"} and has_ifind_credentials(refresh_token, username, password):
        return PRICE_SOURCE_IFIND
    if is_streamlit_cloud_runtime() and public_price_snapshot_exists():
        return PRICE_SOURCE_STATIC
    if has_ifind_credentials(refresh_token, username, password):
        return PRICE_SOURCE_IFIND
    if public_price_snapshot_exists():
        return PRICE_SOURCE_STATIC
    return PRICE_SOURCE_AKSHARE


def parse_open_meteo_daily(payload: dict[str, Any]) -> pd.DataFrame:
    daily = payload.get("daily", {})
    if not daily or "time" not in daily:
        return pd.DataFrame()
    frame = pd.DataFrame(daily)
    frame["date"] = pd.to_datetime(frame["time"])
    return frame.drop(columns=["time"])


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_forecast_weather(
    lat: float,
    lon: float,
    forecast_days: int,
) -> pd.DataFrame:
    payload = request_json(
        OPEN_METEO_FORECAST_URL,
        {
            "latitude": lat,
            "longitude": lon,
            "daily": ",".join(DAILY_WEATHER_FIELDS),
            "forecast_days": forecast_days,
            "timezone": "auto",
            "temperature_unit": "celsius",
            "precipitation_unit": "mm",
            "wind_speed_unit": "kmh",
        },
    )
    return parse_open_meteo_daily(payload)


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def fetch_archive_weather(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    payload = request_json(
        OPEN_METEO_ARCHIVE_URL,
        {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": ",".join(DAILY_WEATHER_FIELDS),
            "timezone": "auto",
            "temperature_unit": "celsius",
            "precipitation_unit": "mm",
            "wind_speed_unit": "kmh",
        },
    )
    return parse_open_meteo_daily(payload)


def aggregate_weather(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {
            "precip_mm": np.nan,
            "et0_mm": np.nan,
            "tmax_c": np.nan,
            "tmin_c": np.nan,
        }
    return {
        "precip_mm": float(frame["precipitation_sum"].sum(skipna=True)),
        "et0_mm": float(frame["et0_fao_evapotranspiration"].sum(skipna=True)),
        "tmax_c": float(frame["temperature_2m_max"].mean(skipna=True)),
        "tmin_c": float(frame["temperature_2m_min"].mean(skipna=True)),
    }


def climatology_for_window(
    lat: float,
    lon: float,
    anchor: dt.date,
    forecast_days: int,
    baseline_years: int,
) -> dict[str, float]:
    rows: list[dict[str, float]] = []
    for year in range(anchor.year - baseline_years, anchor.year):
        start = same_month_day(anchor, year)
        end = start + dt.timedelta(days=forecast_days - 1)
        try:
            archive = fetch_archive_weather(
                lat,
                lon,
                start.isoformat(),
                end.isoformat(),
            )
        except Exception:
            continue
        if not archive.empty:
            rows.append(aggregate_weather(archive))

    if not rows:
        return {
            "normal_precip_mm": np.nan,
            "normal_et0_mm": np.nan,
            "normal_tmax_c": np.nan,
            "normal_tmin_c": np.nan,
        }

    hist = pd.DataFrame(rows)
    return {
        "normal_precip_mm": float(hist["precip_mm"].median(skipna=True)),
        "normal_et0_mm": float(hist["et0_mm"].median(skipna=True)),
        "normal_tmax_c": float(hist["tmax_c"].median(skipna=True)),
        "normal_tmin_c": float(hist["tmin_c"].median(skipna=True)),
    }



def percentile_rank(value: float, observations: pd.Series | list[float]) -> float:
    if value is None or pd.isna(value):
        return np.nan
    series = pd.to_numeric(pd.Series(observations), errors="coerce").dropna()
    if series.empty:
        return np.nan
    return float((series <= float(value)).mean())


def climatology_samples_for_window(
    lat: float,
    lon: float,
    anchor: dt.date,
    window_days: int,
    baseline_years: int,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for year in range(anchor.year - baseline_years, anchor.year):
        start = same_month_day(anchor, year)
        end = start + dt.timedelta(days=window_days - 1)
        try:
            archive = fetch_archive_weather(lat, lon, start.isoformat(), end.isoformat())
        except Exception:
            continue
        if not archive.empty:
            rows.append(aggregate_weather(archive))
    return pd.DataFrame(rows)


def summarize_climatology_samples(samples: pd.DataFrame) -> dict[str, float]:
    if samples.empty:
        return {
            "normal_precip_mm": np.nan,
            "normal_et0_mm": np.nan,
            "normal_tmax_c": np.nan,
            "normal_tmin_c": np.nan,
        }
    return {
        "normal_precip_mm": float(samples["precip_mm"].median(skipna=True)),
        "normal_et0_mm": float(samples["et0_mm"].median(skipna=True)),
        "normal_tmax_c": float(samples["tmax_c"].median(skipna=True)),
        "normal_tmin_c": float(samples["tmin_c"].median(skipna=True)),
    }


def recent_actual_weather(lat: float, lon: float, anchor: dt.date, window_days: int) -> dict[str, float]:
    end = anchor - dt.timedelta(days=1)
    start = end - dt.timedelta(days=window_days - 1)
    try:
        archive = fetch_archive_weather(lat, lon, start.isoformat(), end.isoformat())
    except Exception:
        archive = pd.DataFrame()
    return aggregate_weather(archive)


def score_weather_components(
    region: dict[str, Any],
    settings: RuleSettings,
    precip_ratio: float,
    tmax_anom: float,
    water_balance: float,
    normal_precip: float,
) -> dict[str, Any]:
    dry_points = 0.0
    if not pd.isna(precip_ratio) and precip_ratio < settings.dry_ratio_trigger:
        dry_points += (
            (settings.dry_ratio_trigger - precip_ratio)
            / max(settings.dry_ratio_trigger, 0.01)
            * 72
        )
    if not pd.isna(tmax_anom) and tmax_anom > settings.heat_trigger_c:
        dry_points += min(24, (tmax_anom - settings.heat_trigger_c) * 8)
    if not pd.isna(water_balance) and water_balance < 0:
        deficit_base = max(normal_precip, 10.0) if not pd.isna(normal_precip) else 10.0
        dry_points += min(18, abs(water_balance) / deficit_base * 18)

    wet_points = 0.0
    if not pd.isna(precip_ratio) and precip_ratio > settings.wet_ratio_trigger:
        wet_points += (
            (precip_ratio - settings.wet_ratio_trigger)
            / max(2.0 - settings.wet_ratio_trigger, 0.25)
            * 80
        )
    if not pd.isna(water_balance) and water_balance > 50:
        wet_points += min(18, water_balance / 120 * 18)

    dry_score = clamp(dry_points * float(region["dry"]))
    wet_score = clamp(wet_points * float(region["wet"]))
    stress_score = clamp(max(dry_score, wet_score * 0.85))
    driver = "偏干/高温" if dry_score >= wet_score else "过雨/收割物流扰动"
    if pd.isna(stress_score) or stress_score < 25:
        driver = "天气中性"
    return {
        "dry_score": dry_score,
        "wet_score": wet_score,
        "stress_score": stress_score,
        "driver": driver,
    }


def weather_direction(driver: str, stress_score: float) -> str:
    if pd.isna(stress_score) or stress_score < 25 or driver == "天气中性":
        return "neutral"
    if "偏干" in driver:
        return "dry"
    if "过雨" in driver:
        return "wet"
    return "neutral"



IMPACT_TIMING_MULTIPLIERS = {
    "immediate": 1.00,
    "1-3m": 0.70,
    "3-9m": 0.40,
}
IMPACT_TIMING_LABELS = {
    "immediate": "即时",
    "1-3m": "1-3个月",
    "3-9m": "3-9个月",
}


def commodity_code(commodity_name: str) -> str:
    if "SR" in str(commodity_name):
        return "SR"
    if "P" in str(commodity_name):
        return "P"
    return ""


def classify_weather_impact_timing(
    region: dict[str, Any],
    driver: str,
    stress_score: float,
    anchor: dt.date,
) -> dict[str, Any]:
    direction = weather_direction(driver, stress_score)
    commodity = commodity_code(str(region.get("commodity", "")))
    country = str(region.get("country", ""))
    month = anchor.month
    bucket = "1-3m"
    reason = "weather stress is material but not directly tied to current harvest/logistics"

    if direction == "neutral":
        bucket = "1-3m"
        reason = "weather stress is below the material threshold"
    elif commodity == "SR":
        if direction == "wet":
            if country == "Brazil" and month in {4, 5, 6, 7, 8, 9, 10, 11}:
                bucket = "immediate"
                reason = "Brazil crush/export rainfall can affect current supply flow"
            elif country in {"China", "Thailand"} and month in {11, 12, 1, 2, 3, 4}:
                bucket = "immediate"
                reason = "harvest/crush rainfall can affect near-term domestic or export flow"
            else:
                bucket = "1-3m"
                reason = "rainfall affects cane recovery and supply expectations over the next quarter"
        elif direction == "dry":
            if country in {"India", "Thailand", "China"} and month in {5, 6, 7, 8, 9, 10}:
                bucket = "1-3m"
                reason = "monsoon/growing-season drought shifts crop and policy expectations over 1-3 months"
            elif country == "Brazil" and month in {4, 5, 6, 7, 8, 9}:
                bucket = "1-3m"
                reason = "Brazil dry heat affects current crop quality and next-quarter supply expectations"
            else:
                bucket = "3-9m"
                reason = "sugarcane drought mainly affects next crop or later supply"
    elif commodity == "P":
        if direction == "dry":
            bucket = "3-9m"
            reason = "oil-palm drought usually affects yields with a 3-9 month lag"
        elif direction == "wet":
            if month in {11, 12, 1, 2}:
                bucket = "immediate"
                reason = "rainy-season excess rainfall can disrupt harvest, transport, and exports now"
            else:
                bucket = "1-3m"
                reason = "excess rainfall affects harvesting and fruit collection over the next quarter"

    multiplier = IMPACT_TIMING_MULTIPLIERS[bucket]
    return {
        "impact_timing": bucket,
        "impact_label": IMPACT_TIMING_LABELS[bucket],
        "impact_multiplier": multiplier,
        "impact_reason": reason,
        "entry_ready_score": clamp(stress_score * multiplier),
    }
def persistence_diagnostics(current_driver: str, current_score: float, recent_driver: str, recent_score: float) -> dict[str, Any]:
    current_direction = weather_direction(current_driver, current_score)
    recent_direction = weather_direction(recent_driver, recent_score)
    multiplier = 1.0
    if current_direction == "neutral":
        label = "当前中性"
    elif recent_direction == current_direction:
        label = "持续偏干" if current_direction == "dry" else "持续过雨"
        multiplier = 1.12 if recent_score >= 50 else 1.05
    elif recent_direction == "neutral":
        label = "预报新发偏干" if current_direction == "dry" else "预报新发过雨"
        multiplier = 0.82
    else:
        label = "方向切换/信号混合"
        multiplier = 0.75
    return {
        "persistence_label": label,
        "persistence_multiplier": multiplier,
        "persistence_score": clamp(recent_score),
        "current_direction": current_direction,
        "recent_direction": recent_direction,
        "adjusted_stress_score": clamp(current_score * multiplier),
    }
def score_region_weather(
    region: dict[str, Any],
    settings: RuleSettings,
    anchor: dt.date,
) -> dict[str, Any]:
    lat = float(region["lat"])
    lon = float(region["lon"])
    forecast = fetch_forecast_weather(lat, lon, settings.forecast_days)
    current = aggregate_weather(forecast)

    current_samples = climatology_samples_for_window(
        lat,
        lon,
        anchor,
        settings.forecast_days,
        settings.baseline_years,
    )
    normal = summarize_climatology_samples(current_samples)

    normal_precip = normal["normal_precip_mm"]
    precip_ratio = (
        current["precip_mm"] / normal_precip
        if normal_precip and not pd.isna(normal_precip) and normal_precip > 0
        else np.nan
    )
    tmax_anom = (
        current["tmax_c"] - normal["normal_tmax_c"]
        if not pd.isna(normal["normal_tmax_c"])
        else np.nan
    )
    water_balance = current["precip_mm"] - current["et0_mm"]
    components = score_weather_components(
        region,
        settings,
        precip_ratio,
        tmax_anom,
        water_balance,
        normal_precip,
    )

    recent = recent_actual_weather(lat, lon, anchor, settings.forecast_days)
    recent_anchor = anchor - dt.timedelta(days=settings.forecast_days)
    recent_samples = climatology_samples_for_window(
        lat,
        lon,
        recent_anchor,
        settings.forecast_days,
        settings.baseline_years,
    )
    recent_normal = summarize_climatology_samples(recent_samples)
    recent_normal_precip = recent_normal["normal_precip_mm"]
    recent_precip_ratio = (
        recent["precip_mm"] / recent_normal_precip
        if recent_normal_precip and not pd.isna(recent_normal_precip) and recent_normal_precip > 0
        else np.nan
    )
    recent_tmax_anom = (
        recent["tmax_c"] - recent_normal["normal_tmax_c"]
        if not pd.isna(recent_normal["normal_tmax_c"])
        else np.nan
    )
    recent_water_balance = recent["precip_mm"] - recent["et0_mm"]
    recent_components = score_weather_components(
        region,
        settings,
        recent_precip_ratio,
        recent_tmax_anom,
        recent_water_balance,
        recent_normal_precip,
    )
    persistence = persistence_diagnostics(
        components["driver"],
        components["stress_score"],
        recent_components["driver"],
        recent_components["stress_score"],
    )

    raw_stress_score = components["stress_score"]
    stress_score = persistence["adjusted_stress_score"]
    impact = classify_weather_impact_timing(region, components["driver"], stress_score, anchor)
    precip_percentile = percentile_rank(current["precip_mm"], current_samples.get("precip_mm", []))
    tmax_percentile = percentile_rank(current["tmax_c"], current_samples.get("tmax_c", []))
    water_balance_percentile = percentile_rank(
        water_balance,
        current_samples.get("precip_mm", pd.Series(dtype=float)) - current_samples.get("et0_mm", pd.Series(dtype=float))
        if not current_samples.empty else [],
    )
    recent_precip_percentile = percentile_rank(recent["precip_mm"], recent_samples.get("precip_mm", []))

    return {
        **region,
        **current,
        **normal,
        "precip_ratio": precip_ratio,
        "tmax_anom_c": tmax_anom,
        "water_balance_mm": water_balance,
        "precip_percentile": precip_percentile,
        "tmax_percentile": tmax_percentile,
        "water_balance_percentile": water_balance_percentile,
        "dry_score": components["dry_score"],
        "wet_score": components["wet_score"],
        "raw_stress_score": raw_stress_score,
        "stress_score": stress_score,
        "impact_timing": impact["impact_timing"],
        "impact_label": impact["impact_label"],
        "impact_multiplier": impact["impact_multiplier"],
        "impact_reason": impact["impact_reason"],
        "entry_ready_score": impact["entry_ready_score"],
        "entry_weighted_score": impact["entry_ready_score"] * float(region["weight"]),
        "weighted_score": stress_score * float(region["weight"]),
        "driver": components["driver"],
        "persistence_label": persistence["persistence_label"],
        "persistence_score": persistence["persistence_score"],
        "persistence_multiplier": persistence["persistence_multiplier"],
        "current_direction": persistence["current_direction"],
        "recent_direction": persistence["recent_direction"],
        "recent_precip_mm": recent["precip_mm"],
        "recent_normal_precip_mm": recent_normal_precip,
        "recent_precip_ratio": recent_precip_ratio,
        "recent_tmax_anom_c": recent_tmax_anom,
        "recent_water_balance_mm": recent_water_balance,
        "recent_stress_score": recent_components["stress_score"],
        "recent_driver": recent_components["driver"],
        "recent_precip_percentile": recent_precip_percentile,
        "climatology_sample_count": len(current_samples),
    }

@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def fetch_enso_oni() -> tuple[pd.DataFrame, str | None]:
    try:
        response = requests.get(NOAA_ONI_URL, timeout=25)
        response.raise_for_status()
        rows: list[dict[str, Any]] = []
        season_mid_month = {
            "DJF": 1,
            "JFM": 2,
            "FMA": 3,
            "MAM": 4,
            "AMJ": 5,
            "MJJ": 6,
            "JJA": 7,
            "JAS": 8,
            "ASO": 9,
            "SON": 10,
            "OND": 11,
            "NDJ": 12,
        }
        for line in response.text.splitlines():
            parts = line.split()
            if len(parts) < 4 or parts[0] not in season_mid_month:
                continue
            try:
                rows.append(
                    {
                        "season": parts[0],
                        "year": int(parts[1]),
                        "total": float(parts[2]),
                        "anom": float(parts[3]),
                        "mid_month": season_mid_month[parts[0]],
                    }
                )
            except ValueError:
                continue
        frame = pd.DataFrame(rows)
        if frame.empty:
            raise ValueError("NOAA ONI returned no parseable rows")
        frame["period"] = pd.to_datetime(
            frame["year"].astype(str) + "-" + frame["mid_month"].astype(str) + "-15"
        )
        frame["label"] = frame["season"] + " " + frame["year"].astype(str)
        frame = frame.sort_values("period").reset_index(drop=True)
        mark_frame_cache_source(frame, "live")
        save_last_good_frame("enso", "oni", frame, {"source": "NOAA CPC ONI"})
        return frame, None
    except Exception as exc:
        cached, meta = load_last_good_frame("enso", "oni")
        if not cached.empty:
            message = cached_data_message("ENSO/ONI", meta, exc)
            mark_frame_cache_source(cached, "cache", message)
            return cached, message
        return pd.DataFrame(), str(exc)

def classify_enso(anom: float, persistence: int) -> tuple[str, str, str]:
    abs_anom = abs(anom)
    if abs_anom >= 2.0:
        magnitude = "超强"
    elif abs_anom >= 1.5:
        magnitude = "强"
    elif abs_anom >= 1.0:
        magnitude = "中等"
    else:
        magnitude = "弱"

    if anom >= 0.5:
        phase = "厄尔尼诺"
    elif anom <= -0.5:
        phase = "拉尼娜"
    else:
        return "中性", "无", "未达事件阈值"

    if persistence >= 5:
        confidence = "已确认"
    elif persistence >= 2:
        confidence = "形成观察"
    else:
        confidence = "单期触及"
    return phase, magnitude, confidence


def summarize_enso_for_commodity(
    commodity_name: str,
    enso_frame: pd.DataFrame,
    enso_error: str | None,
) -> dict[str, Any]:
    if enso_frame.empty:
        return {
            "available": False,
            "error": enso_error or "ENSO 数据不可用",
            "phase": "未知",
            "magnitude": "未知",
            "confidence": "未知",
            "bias": "不计入短期天气分；等待 ENSO 数据恢复。",
            "latest_label": "n/a",
            "latest_anom": np.nan,
            "trend": np.nan,
            "persistence": 0,
        }

    latest = enso_frame.iloc[-1]
    previous = enso_frame.iloc[-2] if len(enso_frame) >= 2 else latest
    latest_anom = float(latest["anom"])
    trend = latest_anom - float(previous["anom"])
    sign = 1 if latest_anom >= 0.5 else (-1 if latest_anom <= -0.5 else 0)
    persistence = 0
    if sign:
        for value in reversed(enso_frame["anom"].tolist()):
            if (sign == 1 and value >= 0.5) or (sign == -1 and value <= -0.5):
                persistence += 1
            else:
                break

    phase, magnitude, confidence = classify_enso(latest_anom, persistence)
    if commodity_name == "棕榈油 P":
        if phase == "厄尔尼诺":
            bias = "中期偏多：东南亚偏干风险通常滞后影响油棕单产，需继续用产区降雨和价格确认过滤。"
        elif phase == "拉尼娜":
            bias = "中期偏空或中性：拉尼娜常带来东南亚降雨改善，干旱减产叙事需要更强现场天气验证。"
        else:
            bias = "中性：棕榈油更依赖当前印尼/马来降雨、库存和政策驱动。"
    else:
        if phase in ("厄尔尼诺", "拉尼娜"):
            bias = "波动放大：ENSO 异常会提高印度、巴西、泰国天气扰动概率，但不能替代产区天气和价格确认。"
        else:
            bias = "中性：白糖供应主题主要看印度季风、巴西压榨天气和出口政策。"

    return {
        "available": True,
        "error": enso_error,
        "phase": phase,
        "magnitude": magnitude,
        "confidence": confidence,
        "bias": bias,
        "latest_label": latest["label"],
        "latest_anom": latest_anom,
        "trend": trend,
        "persistence": persistence,
    }


def month_distance(start_month: int, target_month: int) -> int:
    return (target_month - start_month) % 12


def seasonal_context_for(commodity_name: str, anchor: dt.date) -> dict[str, Any]:
    windows = SEASONAL_WINDOWS.get(commodity_name, [])
    active: list[dict[str, Any]] = []
    upcoming: list[dict[str, Any]] = []
    for window in windows:
        months = set(window["months"])
        if anchor.month in months:
            active.append(window)
            continue
        distance = min(month_distance(anchor.month, month) for month in months)
        if 0 < distance <= 2:
            item = dict(window)
            item["starts_in_months"] = distance
            upcoming.append(item)
    return {
        "month": anchor.month,
        "active": active,
        "upcoming": sorted(upcoming, key=lambda item: item["starts_in_months"]),
    }


def build_regime_context(
    commodity_name: str,
    enso_frame: pd.DataFrame,
    enso_error: str | None,
    anchor: dt.date,
) -> dict[str, Any]:
    enso = summarize_enso_for_commodity(commodity_name, enso_frame, enso_error)
    seasonal = seasonal_context_for(commodity_name, anchor)
    active_high = [item for item in seasonal["active"] if item.get("importance") == "高"]
    if active_high and enso["phase"] in ("厄尔尼诺", "拉尼娜"):
        regime_label = "中期背景活跃"
    elif active_high:
        regime_label = "季节窗口活跃"
    elif enso["phase"] in ("厄尔尼诺", "拉尼娜"):
        regime_label = "ENSO 背景活跃"
    else:
        regime_label = "背景中性"
    return {
        "commodity": commodity_name,
        "enso": enso,
        "enso_frame": enso_frame,
        "seasonal": seasonal,
        "label": regime_label,
    }


def render_regime_context(context: dict[str, Any]) -> None:
    st.subheader("Regime 背景（不计入短期天气分）")
    enso = context["enso"]
    seasonal = context["seasonal"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("背景状态", context["label"])
    m2.metric("ENSO", enso["phase"], enso["confidence"])
    latest_anom = enso.get("latest_anom", np.nan)
    trend = enso.get("trend", np.nan)
    m3.metric(
        "最新 ONI",
        "n/a" if pd.isna(latest_anom) else f"{latest_anom:+.2f}°C",
        "n/a" if pd.isna(trend) else f"{trend:+.2f}",
    )
    m4.metric("当前月份", f"{seasonal['month']} 月", f"活跃窗口 {len(seasonal['active'])}")

    st.write(enso["bias"])
    if enso.get("error") and not enso.get("available"):
        st.warning(f"ENSO 数据不可用：{enso['error']}")

    active_rows = [
        {
            "窗口": item["name"],
            "重要性": item["importance"],
            "交易含义": item["bias"],
        }
        for item in seasonal["active"]
    ]
    if active_rows:
        st.write("**当前活跃生产窗口**")
        st.dataframe(pd.DataFrame(active_rows), hide_index=True, width="stretch")
    else:
        st.info("当前没有高权重生产季窗口处于活跃月份。")

    upcoming_rows = [
        {
            "窗口": item["name"],
            "还有几个月": item["starts_in_months"],
            "交易含义": item["bias"],
        }
        for item in seasonal["upcoming"]
    ]
    if upcoming_rows:
        st.write("**未来两个月将进入的窗口**")
        st.dataframe(pd.DataFrame(upcoming_rows), hide_index=True, width="stretch")

    enso_frame = context.get("enso_frame", pd.DataFrame())
    if not enso_frame.empty:
        view = enso_frame.tail(36)
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=view["label"],
                y=view["anom"],
                marker_color=np.where(view["anom"] >= 0, "#dc2626", "#2563eb"),
                name="ONI",
            )
        )
        fig.add_hline(y=0.5, line_dash="dash", line_color="#dc2626", annotation_text="El Nino +0.5")
        fig.add_hline(y=-0.5, line_dash="dash", line_color="#2563eb", annotation_text="La Nina -0.5")
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10), yaxis_title="ONI °C")
        st.plotly_chart(fig, width="stretch")

def build_weather_table(
    settings: RuleSettings,
    anchor: dt.date,
    commodity: str,
) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    regions = [region for region in WEATHER_REGIONS if region["commodity"] == commodity]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(score_region_weather, region, settings, anchor): region
            for region in regions
        }
        for future in as_completed(futures):
            region = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                errors.append(f"{region['region']}: {exc}")
    return pd.DataFrame(rows), errors



def build_all_weather_table(settings: RuleSettings, anchor: dt.date) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    cache_key = f"weather_{anchor.isoformat()}_{settings.forecast_days}_{settings.baseline_years}_{settings.dry_ratio_trigger:.2f}_{settings.wet_ratio_trigger:.2f}_{settings.heat_trigger_c:.1f}"
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {
            pool.submit(score_region_weather, region, settings, anchor): region
            for region in WEATHER_REGIONS
        }
        for future in as_completed(futures):
            region = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                errors.append(f"{region['commodity']} {region['region']}: {exc}")
    frame = pd.DataFrame(rows)
    if not frame.empty:
        mark_frame_cache_source(frame, "live")
        save_last_good_frame("weather", cache_key, frame, {"anchor": anchor.isoformat(), "forecast_days": settings.forecast_days})
        if errors:
            cached, meta = load_last_good_frame("weather", cache_key)
            if not cached.empty and "commodity" in cached.columns and "region" in cached.columns and "commodity" in frame.columns and "region" in frame.columns:
                existing = set(zip(frame["commodity"].astype(str), frame["region"].astype(str)))
                fallback = cached.loc[~cached.apply(lambda row: (str(row.get("commodity", "")), str(row.get("region", ""))) in existing, axis=1)].copy()
                if not fallback.empty:
                    fallback["cache_fallback"] = True
                    frame = pd.concat([frame, fallback], ignore_index=True)
                    mark_frame_cache_source(frame, "partial-cache", cached_data_message("天气缺失产区", meta))
                    errors.append(cached_data_message("天气缺失产区", meta))
        return frame, errors

    cached, meta = load_last_good_frame("weather", cache_key)
    if not cached.empty:
        message = cached_data_message("天气全表", meta, "全部产区实时取数失败")
        mark_frame_cache_source(cached, "cache", message)
        errors.append(message)
        return cached, errors
    return frame, errors
@st.cache_data(ttl=45 * 60, show_spinner=False)
def fetch_price_history(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    import akshare as ak

    data = ak.futures_main_sina(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
    )
    return data


def format_ifind_functionpara(functionpara: dict[str, Any] | None = None) -> str:
    if functionpara is None:
        functionpara = {"Fill": "Blank", "Period": "D"}
    return ";".join(f"{k}:{v}" for k, v in functionpara.items())


def parse_ifind_history_payload(payload: dict[str, Any]) -> pd.DataFrame:
    tables = payload.get("tables")
    if not tables:
        fallback = payload.get("data")
        if isinstance(fallback, list):
            tables = fallback
        else:
            return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for table in tables:
        times = table.get("time", [])
        values = table.get("table", {})
        if not isinstance(times, list):
            continue
        if not isinstance(values, dict):
            continue
        if not values:
            continue
        for idx, moment in enumerate(times):
            row: dict[str, Any] = {
                "time": moment,
            }
            thscode = table.get("thscode")
            if thscode is not None:
                row["thscode"] = thscode
            for col, vals in values.items():
                if isinstance(vals, list):
                    row[col] = vals[idx] if idx < len(vals) else None
                else:
                    row[col] = vals
            rows.append(row)
    return pd.DataFrame(rows)


def fetch_ifind_access_token(refresh_token: str) -> str:
    if not refresh_token:
        return ""
    payload = {"refresh_token": refresh_token}
    response = request_json_post(
        IFIND_ACCESS_TOKEN_URL,
        payload,
        headers={"Content-Type": "application/json"},
    )
    data = response.get("data") or {}
    token = data.get("access_token", "")
    if token:
        return str(token)
    raise RuntimeError(f"iFinD access_token 获取失败: {response}")


@st.cache_data(ttl=30 * 60, show_spinner=False)
def fetch_price_history_ifind_http(
    ifind_symbol: str,
    start_date: str,
    end_date: str,
    access_token: str,
    indicators: str = "open,high,low,close,volume,oi",
) -> pd.DataFrame:
    headers = {
        "Content-Type": "application/json",
        "access_token": access_token,
    }
    payload = {
        "codes": ifind_symbol,
        "indicators": indicators,
        "startdate": start_date,
        "enddate": end_date,
        "functionpara": format_ifind_functionpara(
            {
                "Fill": "Blank",
                "Period": "D",
            },
        ),
    }
    payload = dict(payload)
    raw = request_json_post(IFIND_HISTORY_URL, payload, headers=headers)
    if raw.get("errorcode") not in (None, 0):
        raise RuntimeError(f"iFinD 历史行情返回错误: {raw}")
    return parse_ifind_history_payload(raw)


def fetch_price_history_ifind_sdk(
    ifind_symbol: str,
    start_date: str,
    end_date: str,
    username: str,
    password: str,
) -> pd.DataFrame:
    try:
        from iFinDPy import THS_GetErrorInfo, THS_HistoryQuotes, THS_iFinDLogin
    except Exception as exc:
        raise RuntimeError("当前环境未检测到 iFinDPy，无法使用 iFinD SDK 取数。请在 requirements 添加 iFinDPy 后重试。") from exc

    login_result = THS_iFinDLogin(username, password)
    if login_result not in (0, -201):
        try:
            detail = THS_GetErrorInfo(login_result, True)
        except Exception:
            detail = login_result
        raise RuntimeError(f"iFinD 登录失败: {detail}")

    params = "period:D,pricetype:1,rptcategory:0,fqdate:1900-01-01,hb:YSHB"
    raw = THS_HistoryQuotes(
        ifind_symbol,
        "open;high;low;close;volume;openInt",
        params,
        start_date,
        end_date,
        False,
    )
    if not isinstance(raw, dict):
        raw = dict(raw.__dict__)  # type: ignore[assignment]
    if raw.get("errorcode", 0) != 0:
        raise RuntimeError(f"iFinD 历史行情返回错误: {raw.get('errorcode')} {raw.get('errmsg','')}")
    return parse_ifind_history_payload(raw)


def normalize_price_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    rename_map = {
        "日期": "date",
        "time": "date",
        "开盘价": "open",
        "最高价": "high",
        "最低价": "low",
        "收盘价": "close",
        "成交量": "volume",
        "持仓量": "open_interest",
        "动态结算价": "settle",
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "OpenInterest": "open_interest",
        "openInt": "open_interest",
        "OI": "open_interest",
        "oi": "open_interest",
        "open_interest": "open_interest",
        "latest": "close",
        "vol": "volume",
        "open_volume": "volume",
    }
    data = frame.rename(columns=rename_map).copy()
    required = ["date", "open", "high", "low", "close"]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(f"行情 CSV/接口缺少字段: {', '.join(missing)}")

    data["date"] = pd.to_datetime(data["date"])
    for col in ["open", "high", "low", "close", "volume", "open_interest", "settle"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    if "volume" not in data.columns:
        data["volume"] = np.nan
    if "open_interest" not in data.columns:
        data["open_interest"] = np.nan

    return data.sort_values("date").dropna(subset=["close"]).reset_index(drop=True)


def add_price_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["ma20"] = data["close"].rolling(20).mean()
    data["ma60"] = data["close"].rolling(60).mean()
    data["ma120"] = data["close"].rolling(120).mean()
    prev_close = data["close"].shift(1)
    tr = pd.concat(
        [
            data["high"] - data["low"],
            (data["high"] - prev_close).abs(),
            (data["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    data["atr14"] = tr.rolling(14).mean()
    data["high20_prev"] = data["high"].rolling(20).max().shift(1)
    data["high60_prev"] = data["high"].rolling(60).max().shift(1)
    data["low20_prev"] = data["low"].rolling(20).min().shift(1)
    data["vol20"] = data["volume"].rolling(20).mean()
    data["oi20"] = data["open_interest"].rolling(20).mean()
    return data


def score_price_signal(frame: pd.DataFrame, tick: float) -> dict[str, Any]:
    if frame.empty or len(frame) < 60:
        return {
            "score": 0,
            "latest": None,
            "conditions": ["行情数据不足，至少需要 60 个交易日。"],
            "entry": None,
            "pullback_low": None,
            "pullback_high": None,
            "stop": None,
            "add_level": None,
        }

    data = add_price_indicators(frame)
    latest = data.iloc[-1]
    score = 0
    conditions: list[str] = []

    def yes(condition: bool, points: int, text: str) -> None:
        nonlocal score
        if condition:
            score += points
            conditions.append(f"+{points} {text}")
        else:
            conditions.append(f"未达标 {text}")

    close = float(latest["close"])
    ma20 = float(latest["ma20"]) if not pd.isna(latest["ma20"]) else np.nan
    ma60 = float(latest["ma60"]) if not pd.isna(latest["ma60"]) else np.nan
    ma120 = float(latest["ma120"]) if not pd.isna(latest["ma120"]) else np.nan
    atr = float(latest["atr14"]) if not pd.isna(latest["atr14"]) else max(close * 0.015, tick)
    high20 = float(latest["high20_prev"]) if not pd.isna(latest["high20_prev"]) else close
    high60 = float(latest["high60_prev"]) if not pd.isna(latest["high60_prev"]) else close
    low20 = float(latest["low20_prev"]) if not pd.isna(latest["low20_prev"]) else close
    volume = float(latest["volume"]) if not pd.isna(latest["volume"]) else np.nan
    vol20 = float(latest["vol20"]) if not pd.isna(latest["vol20"]) else np.nan
    oi = float(latest["open_interest"]) if not pd.isna(latest["open_interest"]) else np.nan
    oi20 = float(latest["oi20"]) if not pd.isna(latest["oi20"]) else np.nan

    yes(not pd.isna(ma20) and close > ma20, 15, "收盘价在 20 日均线之上")
    yes(not pd.isna(ma60) and close > ma60, 15, "收盘价在 60 日均线之上")
    yes(not pd.isna(ma20) and not pd.isna(ma60) and ma20 > ma60, 15, "20 日均线高于 60 日均线")
    yes(close > high20, 25, "收盘突破前 20 日高点")
    yes(not pd.isna(ma120) and close > ma120, 10, "收盘价在 120 日均线之上")
    yes(not pd.isna(volume) and not pd.isna(vol20) and volume > vol20 * 1.2, 10, "成交量高于 20 日均量 20%")
    yes(not pd.isna(oi) and not pd.isna(oi20) and oi > oi20 * 1.05, 10, "持仓量高于 20 日均持仓 5%")

    score = int(clamp(score))
    breakout_entry = math.ceil((high20 + tick) / tick) * tick
    pullback_low = ma20 - 0.35 * atr if not pd.isna(ma20) else np.nan
    pullback_high = ma20 + 0.35 * atr if not pd.isna(ma20) else np.nan
    trend_stop = min(ma20 - 1.15 * atr, low20 - tick) if not pd.isna(ma20) else low20 - tick
    add_level = math.ceil((high60 + tick) / tick) * tick

    return {
        "score": score,
        "latest": latest,
        "conditions": conditions,
        "entry": breakout_entry,
        "pullback_low": pullback_low,
        "pullback_high": pullback_high,
        "stop": trend_stop,
        "add_level": add_level,
        "data": data,
    }



def normalize_contract_symbol(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip().upper()


def contract_prefix(commodity: dict[str, Any]) -> str:
    base = normalize_contract_symbol(commodity.get("symbol", ""))
    match = re.match(r"[A-Z]+", base)
    return match.group(0) if match else base.rstrip("0")


def is_main_continuous_symbol(symbol: str) -> bool:
    head = normalize_contract_symbol(symbol).split(".")[0]
    return head.endswith("ZL") or head.endswith("0")


def symbol_matches_commodity(symbol: str, commodity: dict[str, Any]) -> bool:
    head = normalize_contract_symbol(symbol).split(".")[0]
    prefix = contract_prefix(commodity)
    if not prefix or not head.startswith(prefix):
        return False
    suffix = head[len(prefix):]
    if suffix == "":
        return True
    return suffix[0].isdigit() or suffix.startswith("ZL") or suffix == "0"


def infer_contract_month(symbol: str, anchor: dt.date) -> dt.date | None:
    head = normalize_contract_symbol(symbol).split(".")[0]
    if is_main_continuous_symbol(head):
        return None
    match = re.search(r"(\d{3,4})", head)
    if not match:
        return None
    digits = match.group(1)
    if len(digits) == 4:
        year = 2000 + int(digits[:2])
        month = int(digits[2:])
    else:
        base_decade = (anchor.year // 10) * 10
        year = base_decade + int(digits[0])
        if year < anchor.year - 1:
            year += 10
        month = int(digits[1:])
    if month < 1 or month > 12:
        return None
    try:
        return dt.date(year, month, 1)
    except ValueError:
        return None



def exchange_suffix(symbol: str) -> str:
    normalized = normalize_contract_symbol(symbol)
    return normalized.split(".", 1)[1] if "." in normalized else ""


def format_delivery_contract_symbol(commodity: dict[str, Any], base_symbol: str, contract_month: dt.date) -> str:
    prefix = contract_prefix(commodity)
    suffix = exchange_suffix(base_symbol or commodity.get("ifind_symbol", ""))
    if suffix == "CZC":
        digits = f"{contract_month.year % 10}{contract_month.month:02d}"
    else:
        digits = f"{contract_month.year % 100:02d}{contract_month.month:02d}"
    return f"{prefix}{digits}.{suffix}" if suffix else f"{prefix}{digits}"


def candidate_contract_months(commodity: dict[str, Any], anchor: dt.date, count: int = 5) -> list[dt.date]:
    active_months = sorted(int(month) for month in commodity.get("active_contract_months", [1, 5, 9]))
    months: list[dt.date] = []
    for year in range(anchor.year, anchor.year + 3):
        for month in active_months:
            candidate = dt.date(year, month, 1)
            if (candidate - anchor).days >= 25:
                months.append(candidate)
            if len(months) >= count:
                return months
    return months


def candidate_delivery_symbols(
    commodity: dict[str, Any],
    base_symbol: str,
    anchor: dt.date,
    count: int = 5,
) -> list[str]:
    symbols: list[str] = []
    for month in candidate_contract_months(commodity, anchor, count=count):
        symbol = format_delivery_contract_symbol(commodity, base_symbol, month)
        if symbol not in symbols:
            symbols.append(symbol)
    return symbols


def contract_has_open_interest_gap(check: dict[str, Any]) -> bool:
    latest_oi = numeric_value(check.get("latest_open_interest", np.nan))
    oi_coverage = numeric_value(check.get("oi_coverage", np.nan))
    return pd.isna(latest_oi) or latest_oi <= 0 or pd.isna(oi_coverage) or oi_coverage < 0.80


def should_attempt_active_contract_fallback(check: dict[str, Any], symbol: str) -> bool:
    return (
        check.get("status") == "block"
        and is_main_continuous_symbol(symbol)
        and contract_has_open_interest_gap(check)
    )



def is_open_interest_blocker(message: Any) -> bool:
    text = str(message).lower()
    return "持仓" in str(message) or "open interest" in text or "oi" in text


def non_open_interest_blockers(check: dict[str, Any]) -> list[Any]:
    return [item for item in check.get("blockers", []) if not is_open_interest_blocker(item)]
def contract_candidate_rank(check: dict[str, Any]) -> float:
    hard_blockers = non_open_interest_blockers(check)
    if check.get("status") == "block" and hard_blockers:
        return float("-inf")
    latest_volume = numeric_value(check.get("latest_volume", 0.0))
    latest_oi = numeric_value(check.get("latest_open_interest", 0.0))
    volume_coverage = numeric_value(check.get("volume_coverage", 0.0))
    oi_coverage = numeric_value(check.get("oi_coverage", 0.0))
    latest_volume = 0.0 if pd.isna(latest_volume) else max(0.0, latest_volume)
    latest_oi = 0.0 if pd.isna(latest_oi) else max(0.0, latest_oi)
    volume_coverage = 0.0 if pd.isna(volume_coverage) else max(0.0, volume_coverage)
    oi_coverage = 0.0 if pd.isna(oi_coverage) else max(0.0, oi_coverage)
    if latest_volume <= 0 or volume_coverage < 0.80:
        return float("-inf")
    days_to_month = check.get("days_to_contract_month")
    roll_penalty = 0.0
    if days_to_month is not None and days_to_month < 60:
        roll_penalty = 8.0
    oi_missing_penalty = 18.0 if check.get("status") == "block" and not hard_blockers else 0.0
    return (
        math.log1p(latest_oi) * 6.0
        + math.log1p(latest_volume) * 4.0
        + volume_coverage * 20.0
        + oi_coverage * 30.0
        - roll_penalty
        - oi_missing_penalty
    )


def summarize_contract_candidate(symbol: str, check: dict[str, Any] | None = None, error: str | None = None) -> dict[str, Any]:
    if error:
        return {
            "symbol": symbol,
            "status": "error",
            "score": "",
            "volume": "",
            "open_interest": "",
            "roll": "",
            "note": error,
        }
    if check is None:
        return {"symbol": symbol, "status": "n/a", "score": "", "volume": "", "open_interest": "", "roll": "", "note": ""}
    notes = check.get("blockers") or check.get("warnings") or []
    return {
        "symbol": symbol,
        "status": check.get("label", check.get("status", "n/a")),
        "score": check.get("score", ""),
        "volume": number_text(check.get("latest_volume"), 0),
        "open_interest": number_text(check.get("latest_open_interest"), 0),
        "roll": check.get("roll_summary", "n/a"),
        "note": " | ".join(str(item) for item in notes[:2]),
    }


def attach_price_metadata(
    frame: pd.DataFrame,
    price_symbol: str,
    selector: dict[str, Any] | None = None,
) -> pd.DataFrame:
    frame.attrs["price_symbol"] = normalize_contract_symbol(price_symbol)
    if selector:
        frame.attrs["contract_selector"] = selector
    return frame


def select_active_contract_frame(
    commodity_name: str,
    commodity: dict[str, Any],
    base_frame: pd.DataFrame,
    base_symbol: str,
    anchor: dt.date,
    fetch_symbol_frame: Any,
) -> pd.DataFrame:
    normalized_base = normalize_contract_symbol(base_symbol)
    base_check = evaluate_contract_liquidity(commodity_name, commodity, base_frame, normalized_base, anchor)
    selector: dict[str, Any] = {
        "mode": "continuous",
        "base_symbol": normalized_base,
        "selected_symbol": normalized_base,
        "trigger": "",
        "candidates": [],
        "selected_reason": "continuous contract passed liquidity checks",
    }
    if not should_attempt_active_contract_fallback(base_check, normalized_base):
        return attach_price_metadata(base_frame, normalized_base, selector)

    selector.update(
        {
            "mode": "active_contract_fallback",
            "trigger": "continuous contract open-interest unavailable",
            "selected_reason": "no tradable delivery-month candidate selected",
        }
    )
    best_symbol = ""
    best_frame = pd.DataFrame()
    best_check: dict[str, Any] | None = None
    best_rank = float("-inf")

    for candidate_symbol in candidate_delivery_symbols(commodity, normalized_base, anchor):
        try:
            candidate_frame = fetch_symbol_frame(candidate_symbol)
            if candidate_frame.empty:
                selector["candidates"].append(summarize_contract_candidate(candidate_symbol, error="empty frame"))
                continue
            candidate_check = evaluate_contract_liquidity(
                commodity_name,
                commodity,
                candidate_frame,
                candidate_symbol,
                anchor,
            )
            selector["candidates"].append(summarize_contract_candidate(candidate_symbol, candidate_check))
            rank = contract_candidate_rank(candidate_check)
            if rank > best_rank:
                best_rank = rank
                best_symbol = candidate_symbol
                best_frame = candidate_frame
                best_check = candidate_check
        except Exception as exc:
            selector["candidates"].append(summarize_contract_candidate(candidate_symbol, error=str(exc)))

    if best_symbol and best_check is not None and best_rank > float("-inf"):
        selection_basis = "full_oi" if best_check.get("status") != "block" else "volume_only"
        selected_reason = (
            f"selected by OI {number_text(best_check.get('latest_open_interest'), 0)} "
            f"and volume {number_text(best_check.get('latest_volume'), 0)}"
            if selection_basis == "full_oi"
            else f"selected by volume {number_text(best_check.get('latest_volume'), 0)}; OI unavailable"
        )
        selector.update(
            {
                "selected_symbol": best_symbol,
                "selected_reason": selected_reason,
                "selection_basis": selection_basis,
            }
        )
        return attach_price_metadata(best_frame, best_symbol, selector)

    return attach_price_metadata(base_frame, normalized_base, selector)
def evaluate_contract_liquidity(
    commodity_name: str,
    commodity: dict[str, Any],
    price_frame: pd.DataFrame,
    price_symbol: str,
    anchor: dt.date,
) -> dict[str, Any]:
    expected_symbol = normalize_contract_symbol(commodity.get("ifind_symbol", commodity.get("symbol", "")))
    configured_symbol = normalize_contract_symbol(price_symbol or expected_symbol)
    blockers: list[str] = []
    warnings: list[str] = []
    selector = price_frame.attrs.get("contract_selector", {}) if hasattr(price_frame, "attrs") else {}

    if price_frame.empty:
        blockers.append("\u884c\u60c5\u4e3a\u7a7a\uff0c\u65e0\u6cd5\u9a8c\u8bc1\u5408\u7ea6\u548c\u6d41\u52a8\u6027")
        status = "block"
        return {
            "status": status,
            "label": "\u4e0d\u53ef\u7528",
            "score": 0,
            "symbol": configured_symbol or expected_symbol or "n/a",
            "expected_symbol": expected_symbol,
            "observed_symbol": "",
            "latest_date": None,
            "stale_days": None,
            "latest_volume": np.nan,
            "latest_open_interest": np.nan,
            "volume_coverage": 0.0,
            "oi_coverage": 0.0,
            "max_calendar_gap_days": None,
            "max_abs_gap_pct": np.nan,
            "contract_month": None,
            "days_to_contract_month": None,
            "liquidity_summary": "n/a",
            "continuity_summary": "n/a",
            "roll_summary": "n/a",
            "date_summary": "n/a",
            "selector_summary": "",
            "selector_candidates": [],
            "blockers": blockers,
            "warnings": warnings,
        }

    data = add_price_indicators(price_frame) if len(price_frame) else price_frame.copy()
    latest = data.iloc[-1]
    observed_symbol = ""
    observed_symbols: list[str] = []
    if "thscode" in data.columns:
        observed_series = data["thscode"].dropna().astype(str).map(normalize_contract_symbol)
        observed_symbols = [item for item in observed_series.tail(60).unique().tolist() if item]
        if observed_symbols:
            observed_symbol = observed_symbols[-1]
    symbol = observed_symbol or configured_symbol or expected_symbol

    if not symbol:
        blockers.append("\u5408\u7ea6\u4ee3\u7801\u7f3a\u5931")
    elif not symbol_matches_commodity(symbol, commodity):
        blockers.append(f"\u5408\u7ea6\u4ee3\u7801\u4e0e\u54c1\u79cd\u4e0d\u5339\u914d: {symbol}")
    elif configured_symbol and not is_main_continuous_symbol(configured_symbol):
        warnings.append("\u4e0d\u662f\u9ed8\u8ba4\u4e3b\u529b\u8fde\u7eed\u4ee3\u7801\uff0c\u9700\u4eba\u5de5\u786e\u8ba4\u662f\u5426\u4e3a\u5f53\u524d\u4e3b\u529b\u5408\u7ea6")

    latest_date: dt.date | None = None
    stale_days: int | None = None
    if "date" in data.columns:
        latest_value = pd.to_datetime(data["date"]).max()
        if pd.notna(latest_value):
            latest_date = latest_value.date()
            stale_days = max(0, (anchor - latest_date).days)
    if stale_days is None:
        blockers.append("\u6700\u65b0\u884c\u60c5\u65e5\u671f\u4e0d\u53ef\u8bc6\u522b")
    elif stale_days > 3:
        blockers.append(f"\u6700\u65b0\u884c\u60c5\u6ede\u540e {stale_days} \u5929")

    latest_volume = numeric_value(latest.get("volume", np.nan))
    latest_oi = numeric_value(latest.get("open_interest", np.nan))
    vol20 = numeric_value(latest.get("vol20", np.nan))
    oi20 = numeric_value(latest.get("oi20", np.nan))
    tail60 = data.tail(60).copy()
    volume_coverage = float(tail60["volume"].notna().mean()) if "volume" in tail60.columns and len(tail60) else 0.0
    oi_coverage = float(tail60["open_interest"].notna().mean()) if "open_interest" in tail60.columns and len(tail60) else 0.0
    zero_volume_days = int((pd.to_numeric(tail60.get("volume", pd.Series(dtype=float)), errors="coerce").fillna(0) <= 0).sum()) if len(tail60) else 0

    if pd.isna(latest_volume) or latest_volume <= 0:
        blockers.append("\u6700\u65b0\u6210\u4ea4\u91cf\u7f3a\u5931\u6216\u4e3a 0")
    if pd.isna(latest_oi) or latest_oi <= 0:
        blockers.append("\u6700\u65b0\u6301\u4ed3\u91cf\u7f3a\u5931\u6216\u4e3a 0")
    if volume_coverage < 0.80:
        blockers.append("\u8fd1 60 \u65e5\u6210\u4ea4\u91cf\u8986\u76d6\u4e0d\u8db3")
    if oi_coverage < 0.80:
        blockers.append("\u8fd1 60 \u65e5\u6301\u4ed3\u91cf\u8986\u76d6\u4e0d\u8db3")
    if not pd.isna(latest_volume) and not pd.isna(vol20) and vol20 > 0 and latest_volume < vol20 * 0.40:
        warnings.append("\u6210\u4ea4\u91cf\u4f4e\u4e8e 20 \u65e5\u5747\u91cf 40%\uff0c\u6d41\u52a8\u6027\u8f6c\u5f31")
    if not pd.isna(latest_oi) and not pd.isna(oi20) and oi20 > 0 and latest_oi < oi20 * 0.40:
        warnings.append("\u6301\u4ed3\u91cf\u4f4e\u4e8e 20 \u65e5\u5747\u6301\u4ed3 40%\uff0c\u53ef\u80fd\u4e34\u8fd1\u79fb\u4ed3")
    if zero_volume_days > 0:
        warnings.append(f"\u8fd1 60 \u65e5\u5b58\u5728 {zero_volume_days} \u4e2a\u96f6\u6210\u4ea4\u65e5")

    max_calendar_gap_days: int | None = None
    if "date" in data.columns and len(data) > 1:
        dates = pd.to_datetime(data["date"]).dropna().sort_values()
        gaps = dates.diff().dt.days.dropna()
        if not gaps.empty:
            max_calendar_gap_days = int(gaps.max())
            if max_calendar_gap_days > 14:
                blockers.append(f"\u6700\u8fd1\u4ea4\u6613\u65e5\u65ad\u6863 {max_calendar_gap_days} \u5929")
            elif max_calendar_gap_days > 8:
                warnings.append(f"\u6700\u8fd1\u4ea4\u6613\u65e5\u95f4\u9694 {max_calendar_gap_days} \u5929\uff0c\u786e\u8ba4\u8282\u5047\u65e5/\u6570\u636e\u8fde\u7eed\u6027")

    returns = pd.to_numeric(data["close"], errors="coerce").pct_change().abs().replace([np.inf, -np.inf], np.nan).dropna()
    max_abs_gap_pct = float(returns.tail(60).max()) if not returns.empty else np.nan
    if not pd.isna(max_abs_gap_pct):
        if max_abs_gap_pct > 0.18:
            blockers.append(f"\u8fd1 60 \u65e5\u4ef7\u683c\u8df3\u53d8 {max_abs_gap_pct:.1%}\uff0c\u9700\u786e\u8ba4\u4e3b\u529b\u8fde\u7eed/\u590d\u6743")
        elif max_abs_gap_pct > 0.10:
            warnings.append(f"\u8fd1 60 \u65e5\u4ef7\u683c\u8df3\u53d8 {max_abs_gap_pct:.1%}\uff0c\u9700\u786e\u8ba4\u6362\u6708\u8fde\u7eed\u6027")
    if len(observed_symbols) > 1:
        warnings.append("\u6700\u8fd1 60 \u65e5\u5408\u7ea6\u6807\u8bc6\u53d1\u751f\u53d8\u5316\uff0c\u786e\u8ba4\u4e3b\u529b\u6362\u6708\u62fc\u63a5")

    contract_month = infer_contract_month(symbol, anchor)
    days_to_contract_month: int | None = None
    if contract_month is not None:
        days_to_contract_month = (contract_month - anchor).days
        if days_to_contract_month < 20:
            blockers.append("\u5408\u7ea6\u5df2\u8fdb\u5165\u6216\u63a5\u8fd1\u4ea4\u5272\u6708\uff0c\u79fb\u4ed3\u98ce\u9669\u9ad8")
        elif days_to_contract_month < 45:
            warnings.append(f"\u8ddd\u79bb\u5408\u7ea6\u6708\u4efd {days_to_contract_month} \u5929\uff0c\u5173\u6ce8\u79fb\u4ed3\u8282\u594f")

    selector_summary = ""
    if selector:
        selector_mode = selector.get("mode", "")
        selector_selected = selector.get("selected_symbol", "")
        selector_base = selector.get("base_symbol", "")
        if selector_mode == "active_contract_fallback":
            selector_summary = f"候选合约 {selector_base} -> {selector_selected}: {selector.get('selected_reason', '')}"
            if selector_selected == symbol and selector_selected != selector_base:
                warnings.append(f"已用活跃交割月候选 {selector_selected} 替代连续合约 {selector_base}")
                if selector.get("selection_basis") == "volume_only":
                    oi_blockers = [item for item in blockers if is_open_interest_blocker(item)]
                    if oi_blockers:
                        blockers = [item for item in blockers if not is_open_interest_blocker(item)]
                        warnings.extend(f"OI unavailable on selected candidate; {item}" for item in oi_blockers)
            elif selector_selected == selector_base:
                warnings.append("连续合约缺少 OI，候选交割月未选出可替代合约")

    status = "block" if blockers else "caution" if warnings else "pass"
    label_map = {"pass": "\u53ef\u7528", "caution": "\u9700\u590d\u6838", "block": "\u4e0d\u53ef\u7528"}
    score = int(clamp(100 - len(blockers) * 35 - len(warnings) * 10))
    if status == "block":
        score = min(score, 55)
    elif status == "caution":
        score = min(score, 82)

    continuity_jump = "n/a" if pd.isna(max_abs_gap_pct) else f"jump {max_abs_gap_pct:.1%}"
    continuity_gap = "gap n/a" if max_calendar_gap_days is None else f"gap {max_calendar_gap_days}d"
    if contract_month is None:
        roll_summary = "main continuous"
    else:
        roll_summary = f"{contract_month:%Y-%m} / {days_to_contract_month}d"

    return {
        "status": status,
        "label": label_map[status],
        "score": score,
        "symbol": symbol or "n/a",
        "expected_symbol": expected_symbol,
        "configured_symbol": configured_symbol,
        "observed_symbol": observed_symbol,
        "latest_date": latest_date,
        "stale_days": stale_days,
        "latest_volume": latest_volume,
        "vol20": vol20,
        "latest_open_interest": latest_oi,
        "oi20": oi20,
        "volume_coverage": volume_coverage,
        "oi_coverage": oi_coverage,
        "zero_volume_days": zero_volume_days,
        "max_calendar_gap_days": max_calendar_gap_days,
        "max_abs_gap_pct": max_abs_gap_pct,
        "contract_month": contract_month,
        "days_to_contract_month": days_to_contract_month,
        "liquidity_summary": f"Vol {number_text(latest_volume, 0)} / OI {number_text(latest_oi, 0)}",
        "continuity_summary": f"{continuity_gap} / {continuity_jump}",
        "roll_summary": roll_summary,
        "date_summary": latest_date.isoformat() if latest_date else "n/a",
        "selector_summary": selector_summary,
        "selector_candidates": selector.get("candidates", []) if selector else [],
        "blockers": blockers,
        "warnings": warnings,
    }
def weather_pressure_score_for(commodity: str, weather: pd.DataFrame) -> float:
    if weather.empty or "commodity" not in weather.columns:
        return np.nan
    subset = weather.loc[weather["commodity"] == commodity].copy()
    if subset.empty:
        return np.nan
    total_weight = subset["weight"].sum()
    if total_weight <= 0:
        return np.nan
    return float(subset["weighted_score"].sum() / total_weight)


def weather_score_for(commodity: str, weather: pd.DataFrame) -> float:
    if weather.empty or "commodity" not in weather.columns:
        return np.nan
    subset = weather.loc[weather["commodity"] == commodity].copy()
    if subset.empty:
        return np.nan
    total_weight = subset["weight"].sum()
    if total_weight <= 0:
        return np.nan
    score_column = "entry_weighted_score" if "entry_weighted_score" in subset.columns else "weighted_score"
    return float(subset[score_column].sum() / total_weight)


def classify_signal(
    weather_score: float,
    price_score: int,
    settings: RuleSettings,
    commodity: dict[str, Any],
) -> tuple[str, str, float]:
    if pd.isna(weather_score):
        combined = price_score * commodity["price_weight"]
        return "数据不足", "天气数据未取到，暂不生成建仓提示。", combined

    combined = (
        weather_score * float(commodity["weather_weight"])
        + price_score * float(commodity["price_weight"])
    )
    if (
        combined >= settings.build_trigger
        and weather_score >= settings.weather_trigger
        and price_score >= settings.price_trigger
    ):
        return "开始试仓", "天气主题和价格确认同时达标，可考虑 20%-30% 计划仓位试仓。", combined

    if weather_score >= settings.weather_trigger and price_score < settings.price_trigger:
        return "等待价格触发", "天气主题已经抬升，但价格确认不足，等待突破位或 20 日线回踩企稳。", combined

    if weather_score < settings.weather_trigger and price_score >= settings.price_trigger:
        return "技术偏强但不追", "价格结构偏强，但天气溢价不足，避免把普通反弹当作天气趋势。", combined

    if combined >= 55:
        return "观察偏多", "条件接近但未共振，适合加入观察清单，等待下一次确认。", combined

    return "继续观察", "天气和价格尚未形成多头共振。", combined


def evaluate_signal_health(
    selected: str,
    weather: pd.DataFrame,
    price_frame: pd.DataFrame,
    weather_errors: list[str],
    price_errors: list[str],
    anchor: dt.date,
    contract_check: dict[str, Any] | None = None,
) -> dict[str, Any]:
    contract_check = contract_check or {}
    expected_regions = [region for region in WEATHER_REGIONS if region["commodity"] == selected]
    expected_count = len(expected_regions)
    if weather.empty or "commodity" not in weather.columns:
        weather_rows = 0
    else:
        weather_rows = int((weather["commodity"] == selected).sum())
    weather_coverage = weather_rows / expected_count if expected_count else 0.0

    price_bars = len(price_frame)
    latest_date: dt.date | None = None
    stale_days: int | None = None
    if not price_frame.empty and "date" in price_frame.columns:
        latest_value = pd.to_datetime(price_frame["date"]).max()
        if pd.notna(latest_value):
            latest_date = latest_value.date()
            stale_days = max(0, (anchor - latest_date).days)

    if "open_interest" in price_frame.columns and price_bars:
        oi_coverage = float(price_frame["open_interest"].notna().mean())
    else:
        oi_coverage = 0.0

    weather_health = clamp(weather_coverage * 100)
    sample_health = clamp((price_bars / 120) * 100)
    freshness_health = 100.0 if stale_days is not None and stale_days <= 3 else 0.0
    field_health = 100.0 if oi_coverage >= 0.8 else 70.0
    contract_health = float(contract_check.get("score", 100.0)) if contract_check else 100.0
    health_score = int(round(
        weather_health * 0.30
        + sample_health * 0.25
        + freshness_health * 0.15
        + field_health * 0.10
        + contract_health * 0.20
    ))

    blockers: list[str] = []
    warnings: list[str] = []
    if weather_coverage < 0.80:
        blockers.append(f"天气覆盖不足：{weather_rows}/{expected_count} 个产区")
    if price_bars < 60:
        blockers.append(f"行情样本不足：{price_bars}/60 个交易日")
    if stale_days is None:
        blockers.append("行情最新日期不可识别")
    elif stale_days > 3:
        blockers.append(f"行情滞后 {stale_days} 天")
    if price_errors:
        blockers.append("行情接口存在错误")


    if contract_check:
        blockers.extend(contract_check.get("blockers", []))
        warnings.extend(contract_check.get("warnings", []))

    if weather_errors:
        warnings.append(f"天气接口有 {len(weather_errors)} 条错误，需复核缺失区域")
    if weather.attrs.get("last_good_cache") or weather.attrs.get("data_source_status") in {"cache", "partial-cache"}:
        warnings.append(weather.attrs.get("data_source_message") or "天气使用 last-good 缓存，需复核陈旧程度")
    if price_frame.attrs.get("last_good_cache") or price_frame.attrs.get("data_source_status") == "cache":
        warnings.append(price_frame.attrs.get("data_source_message") or "行情使用 last-good 缓存，需复核陈旧程度")
    if 60 <= price_bars < 120:
        warnings.append("行情样本少于 120 个交易日，长期均线确认会偏弱")
    if oi_coverage < 0.80:
        warnings.append("持仓量覆盖不足，价格确认不含有效持仓扩张信号")

    if blockers:
        gate = "block"
        label = "不可用"
    elif health_score < 75 or warnings:
        gate = "caution"
        label = "谨慎可用"
    else:
        gate = "pass"
        label = "可用"

    return {
        "score": health_score,
        "gate": gate,
        "label": label,
        "weather_rows": weather_rows,
        "expected_regions": expected_count,
        "weather_coverage": weather_coverage,
        "price_bars": price_bars,
        "latest_date": latest_date,
        "stale_days": stale_days,
        "oi_coverage": oi_coverage,
        "contract_check": contract_check,
        "blockers": blockers,
        "warnings": warnings,
    }


def apply_signal_health_gate(action: str, note: str, health: dict[str, Any]) -> tuple[str, str]:
    blockers = health.get("blockers", [])
    warnings = health.get("warnings", [])
    if health.get("gate") == "block":
        reason = "；".join(blockers[:2]) if blockers else "数据质量未达标"
        return "信号不可用", f"{reason}。先修复数据，再讨论建仓。"
    if health.get("gate") == "caution":
        caution = "；".join(warnings[:2]) if warnings else "数据质量为谨慎等级"
        if action == "开始试仓":
            return "待人工复核", f"交易条件达标，但{caution}。先复核数据源和主力合约，再轻仓处理。"
        return action, f"{note} 数据质量提示：{caution}。"
    return action, note



def build_entry_playbook(
    selected: str,
    commodity: dict[str, Any],
    weather_score: float,
    combined: float,
    price_signal: dict[str, Any],
    settings: RuleSettings,
    signal_health: dict[str, Any],
    position_plan: dict[str, Any],
) -> dict[str, Any]:
    latest = price_signal.get("latest")
    price_score = int(price_signal.get("score", 0))
    gate = signal_health.get("gate", "pass")
    unit = commodity.get("unit", "")
    entry = numeric_value(price_signal.get("entry"))
    stop = numeric_value(price_signal.get("stop"))
    add_level = numeric_value(price_signal.get("add_level"))
    pullback_low = numeric_value(price_signal.get("pullback_low"))
    pullback_high = numeric_value(price_signal.get("pullback_high"))
    latest_close = np.nan
    if latest is not None and "close" in latest and not pd.isna(latest["close"]):
        latest_close = float(latest["close"])

    weather_ready = not pd.isna(weather_score) and weather_score >= settings.weather_trigger
    weather_watch = not pd.isna(weather_score) and weather_score >= settings.weather_trigger * 0.8
    price_ready = price_score >= settings.price_trigger
    combined_ready = not pd.isna(combined) and combined >= settings.build_trigger
    entry_touched = not pd.isna(latest_close) and not pd.isna(entry) and latest_close >= entry
    add_touched = not pd.isna(latest_close) and not pd.isna(add_level) and latest_close >= add_level
    stop_broken = not pd.isna(latest_close) and not pd.isna(stop) and latest_close <= stop
    weather_failed = pd.isna(weather_score) or weather_score < settings.weather_trigger * 0.7
    gate_blocked = gate == "block"
    gate_caution = gate == "caution"

    pullback_text = (
        "n/a"
        if pd.isna(pullback_low) or pd.isna(pullback_high)
        else f"{number_text(pullback_low, 0)} - {number_text(pullback_high, 0)} {unit}"
    )
    trial_lots = position_plan.get("trial_lots", "n/a") if position_plan.get("ok") else "n/a"
    review_suffix = "；合约/数据需人工复核" if gate_caution else ""

    rows = [
        {
            "阶段": "观察",
            "状态": "已触发" if weather_watch and not gate_blocked else ("禁用" if gate_blocked else "待触发"),
            "触发条件": f"入场天气分 >= {settings.weather_trigger * 0.8:.0f}，且合约闸门不阻断。",
            "执行动作": "加入盘前跟踪清单；只记录天气主因和价格位置，不下单。",
            "风控/失效": f"若入场天气分 < {settings.weather_trigger * 0.7:.0f}，从天气交易清单移除。",
        },
        {
            "阶段": "价格触发",
            "状态": "已触发" if weather_ready and price_ready and not gate_blocked else ("禁用" if gate_blocked else "待触发"),
            "触发条件": f"入场天气分 >= {settings.weather_trigger}，价格确认分 >= {settings.price_trigger}；突破试仓价 {number_text(entry, 0)} {unit} 或回踩区 {pullback_text} 后重新收强。",
            "执行动作": "准备试仓单，确认主力合约、成交/持仓和移仓风险。",
            "风控/失效": f"若收盘跌回止损位 {number_text(stop, 0)} {unit} 下方或天气分回落，取消触发。",
        },
        {
            "阶段": "试仓",
            "状态": "已触发" if weather_ready and price_ready and combined_ready and entry_touched and not gate_blocked else ("禁用" if gate_blocked else "待触发"),
            "触发条件": f"综合分 >= {settings.build_trigger} 且收盘 >= {number_text(entry, 0)} {unit}{review_suffix}。",
            "执行动作": f"执行计划试仓 {trial_lots} 手；不超过预设试仓风险比例。",
            "风控/失效": f"防守止损 {number_text(stop, 0)} {unit}；若次日无法站回触发价，停止加仓。",
        },
        {
            "阶段": "加仓",
            "状态": "已触发" if weather_ready and price_ready and add_touched and not gate_blocked else ("禁用" if gate_blocked else "待触发"),
            "触发条件": f"持有试仓后，收盘突破加仓确认价 {number_text(add_level, 0)} {unit}，且天气分仍 >= {settings.weather_trigger}。",
            "执行动作": "只在浮盈试仓基础上加仓；加仓后整体风险仍受账户风险和保证金上限约束。",
            "风控/失效": "若突破后放量失败或收盘跌回加仓价下方，不追加新仓。",
        },
        {
            "阶段": "失效",
            "状态": "已触发" if gate_blocked or stop_broken or weather_failed else "未触发",
            "触发条件": f"合约/数据闸门阻断，或收盘 <= {number_text(stop, 0)} {unit}，或入场天气分 < {settings.weather_trigger * 0.7:.0f}。",
            "执行动作": "取消天气多头建仓计划；已有试仓按止损纪律处理。",
            "风控/失效": "失效后至少等待一次新的天气-价格共振，不用旧信号补仓。",
        },
    ]

    if gate_blocked:
        current_stage = "失效"
    elif stop_broken or weather_failed:
        current_stage = "失效"
    elif weather_ready and price_ready and add_touched:
        current_stage = "加仓"
    elif weather_ready and price_ready and combined_ready and entry_touched:
        current_stage = "试仓"
    elif weather_ready and price_ready:
        current_stage = "价格触发"
    elif weather_watch:
        current_stage = "观察"
    else:
        current_stage = "等待"

    next_row = next((row for row in rows if row["状态"] == "待触发"), None)
    invalidate_row = rows[-1]
    return {
        "commodity": selected,
        "current_stage": current_stage,
        "gate": gate,
        "weather_ready": weather_ready,
        "price_ready": price_ready,
        "combined_ready": combined_ready,
        "latest_close": latest_close,
        "next_trigger": next_row["触发条件"] if next_row else "当前没有新的待触发条件；重点检查失效条件。",
        "invalidate": invalidate_row["触发条件"],
        "rows": rows,
    }

def post_entry_management_profile(selected: str) -> dict[str, float]:
    name = str(selected)
    if "白糖" in name or "SR" in name:
        return {
            "time_stop_days": 8,
            "trail_start_r": 1.0,
            "partial_exit_r": 1.8,
            "trail_atr_mult": 0.60,
            "de_risk_weather_mult": 0.80,
            "no_add_weather_mult": 0.95,
        }
    return {
        "time_stop_days": 12,
        "trail_start_r": 1.0,
        "partial_exit_r": 2.0,
        "trail_atr_mult": 0.80,
        "de_risk_weather_mult": 0.75,
        "no_add_weather_mult": 0.95,
    }


def build_post_entry_management_playbook(
    selected: str,
    commodity: dict[str, Any],
    weather_score: float,
    price_signal: dict[str, Any],
    settings: RuleSettings,
    signal_health: dict[str, Any],
    position_plan: dict[str, Any],
    entry_playbook: dict[str, Any] | None = None,
    position_state: dict[str, Any] | None = None,
    anchor: dt.date | None = None,
) -> dict[str, Any]:
    profile = post_entry_management_profile(selected)
    latest = price_signal.get("latest")
    gate = signal_health.get("gate", "pass")
    unit = commodity.get("unit", "")
    anchor = anchor or today_china()
    position_state = normalize_position_state(position_state or {}, selected)
    actual_position = position_state_is_active(position_state)
    actual_lots = numeric_value(position_state.get("lots", 0))
    actual_entry = numeric_value(position_state.get("avg_entry_price", np.nan))
    actual_entry_date = parse_position_date(position_state.get("entry_date", ""))
    holding_days = holding_days_for_position(position_state, price_signal, anchor) if actual_position else None
    price_score = int(price_signal.get("score", 0))
    planned_entry = numeric_value(price_signal.get("entry"))
    entry = actual_entry if actual_position and not pd.isna(actual_entry) else planned_entry
    stop = numeric_value(price_signal.get("stop"))
    add_level = numeric_value(price_signal.get("add_level"))
    latest_close = np.nan
    ma20 = np.nan
    ma60 = np.nan
    atr = np.nan
    if latest is not None:
        latest_close = numeric_value(latest["close"] if "close" in latest else np.nan)
        ma20 = numeric_value(latest["ma20"] if "ma20" in latest else np.nan)
        ma60 = numeric_value(latest["ma60"] if "ma60" in latest else np.nan)
        atr = numeric_value(latest["atr14"] if "atr14" in latest else np.nan)

    risk_per_unit = entry - stop if not pd.isna(entry) and not pd.isna(stop) else np.nan
    r_multiple = (
        (latest_close - entry) / risk_per_unit
        if not pd.isna(latest_close) and not pd.isna(entry) and not pd.isna(risk_per_unit) and risk_per_unit > 0
        else np.nan
    )
    raw_trailing_stop = ma20 - profile["trail_atr_mult"] * atr if not pd.isna(ma20) and not pd.isna(atr) else np.nan
    trailing_stop = max(stop, raw_trailing_stop) if not pd.isna(stop) and not pd.isna(raw_trailing_stop) else stop
    partial_exit_level = entry + profile["partial_exit_r"] * risk_per_unit if not pd.isna(entry) and not pd.isna(risk_per_unit) else np.nan
    weather_de_risk_level = settings.weather_trigger * profile["de_risk_weather_mult"]
    weather_no_add_level = settings.weather_trigger * profile["no_add_weather_mult"]

    entry_stage = (entry_playbook or {}).get("current_stage", "")
    post_entry_active = actual_position or entry_stage in {"试仓", "加仓"}
    gate_blocked = gate == "block"
    gate_caution = gate == "caution"
    close_below_stop = not pd.isna(latest_close) and not pd.isna(stop) and latest_close <= stop
    close_below_ma20 = not pd.isna(latest_close) and not pd.isna(ma20) and latest_close < ma20
    close_below_ma60 = not pd.isna(latest_close) and not pd.isna(ma60) and latest_close < ma60
    weather_soft_fail = pd.isna(weather_score) or weather_score < weather_de_risk_level
    weather_no_add = pd.isna(weather_score) or weather_score < weather_no_add_level
    price_weak = price_score < settings.price_trigger * 0.8
    partial_triggered = not pd.isna(r_multiple) and r_multiple >= profile["partial_exit_r"]
    trail_active = not pd.isna(r_multiple) and r_multiple >= profile["trail_start_r"]
    time_stop_triggered = (
        actual_position
        and holding_days is not None
        and holding_days >= int(profile["time_stop_days"])
        and (pd.isna(r_multiple) or r_multiple < 0.5 or (not pd.isna(latest_close) and latest_close < entry))
    )
    add_blocked = (
        gate != "pass"
        or weather_no_add
        or price_score < settings.price_trigger
        or close_below_ma20
        or (not pd.isna(latest_close) and not pd.isna(add_level) and latest_close < add_level)
    )
    de_risk_triggered = gate_blocked or gate_caution or close_below_stop or close_below_ma60 or weather_soft_fail or price_weak

    rows = [
        {
            "管理项": "移动止损",
            "状态": "已启用" if post_entry_active and trail_active else ("待启用" if post_entry_active else "预案"),
            "触发条件": f"试仓后浮盈 >= {profile['trail_start_r']:.1f}R，且收盘维持在 20 日均线上方。",
            "执行动作": f"把防守止损从 {number_text(stop, 0)} {unit} 上移到 {number_text(trailing_stop, 0)} {unit}；只上移不下移。",
            "风控备注": f"跟踪参考为 MA20 - {profile['trail_atr_mult']:.2f} ATR；跌破后按收盘纪律降风险。",
        },
        {
            "管理项": "时间止损",
            "状态": "已触发" if time_stop_triggered else ("计时中" if actual_position else ("规则就绪" if post_entry_active else "预案")),
            "触发条件": f"试仓后 {int(profile['time_stop_days'])} 个交易日内仍未达到 +0.5R 或未重新站上触发价。",
            "执行动作": "不再等待天气叙事发酵，减半或退出试仓，释放风险预算。",
            "风控备注": f"当前记录持仓 {holding_days if holding_days is not None else 'n/a'} 天；实际试仓日 {actual_entry_date.isoformat() if actual_entry_date else 'n/a'}。",
        },
        {
            "管理项": "部分止盈",
            "状态": "已触发" if post_entry_active and partial_triggered else ("待触发" if post_entry_active else "预案"),
            "触发条件": f"收盘达到 +{profile['partial_exit_r']:.1f}R，参考价 {number_text(partial_exit_level, 0)} {unit}。",
            "执行动作": "减掉 1/3 到 1/2 试仓或把止损抬至成本上方，保留尾部仓位。",
            "风控备注": "部分止盈后不因单日回落重新加回，除非再次出现天气和价格共振。",
        },
        {
            "管理项": "不加仓",
            "状态": "已触发" if post_entry_active and add_blocked else ("未触发" if post_entry_active else "预案"),
            "触发条件": f"未站上加仓确认价 {number_text(add_level, 0)} {unit}，或天气分 < {weather_no_add_level:.0f}，或价格/合约闸门不足。",
            "执行动作": "只保留试仓观察，不把试仓自动扩成趋势仓。",
            "风控备注": "若合约闸门为谨慎或阻断，任何加仓都需要人工复核。",
        },
        {
            "管理项": "降风险",
            "状态": "已触发" if post_entry_active and de_risk_triggered else ("未触发" if post_entry_active else "预案"),
            "触发条件": f"收盘跌破止损/60 日线，入场天气分 < {weather_de_risk_level:.0f}，价格确认降温，或合约/数据闸门恶化。",
            "执行动作": "先减风险再解释原因；优先减试仓、取消加仓计划，并重新评估天气驱动是否仍成立。",
            "风控备注": "降风险触发后，下一次加仓必须重新通过入场 playbook。",
        },
    ]

    if not post_entry_active:
        current_management = "未进入持仓管理"
    elif close_below_stop or gate_blocked:
        current_management = "退出/失效"
    elif time_stop_triggered:
        current_management = "时间止损"
    elif de_risk_triggered:
        current_management = "降风险"
    elif partial_triggered:
        current_management = "部分止盈"
    elif add_blocked:
        current_management = "不加仓"
    elif trail_active:
        current_management = "移动止损"
    else:
        current_management = "持有试仓"

    return {
        "commodity": selected,
        "current_management": current_management,
        "post_entry_active": post_entry_active,
        "actual_position": actual_position,
        "position_entry_date": actual_entry_date.isoformat() if actual_entry_date else "",
        "position_avg_entry": "" if pd.isna(actual_entry) else actual_entry,
        "position_lots": 0 if pd.isna(actual_lots) else int(actual_lots),
        "holding_days": holding_days,
        "time_stop_triggered": time_stop_triggered,
        "planned_entry": planned_entry,
        "r_multiple": r_multiple,
        "trailing_stop": trailing_stop,
        "partial_exit_level": partial_exit_level,
        "time_stop_days": int(profile["time_stop_days"]),
        "de_risk_level": weather_de_risk_level,
        "no_add_level": weather_no_add_level,
        "rows": rows,
    }
def render_signal_health(health: dict[str, Any]) -> None:
    st.subheader("信号质量闸门")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("健康度", f"{health['score']}/100", health["label"])
    c2.metric(
        "天气覆盖",
        f"{health['weather_rows']}/{health['expected_regions']}",
        pct_text(health["weather_coverage"]),
    )
    latest_date = health.get("latest_date")
    stale_days = health.get("stale_days")
    c3.metric("行情样本", f"{health['price_bars']} 日", "目标 120 日")
    c4.metric(
        "最新行情日",
        latest_date.isoformat() if latest_date else "n/a",
        "n/a" if stale_days is None else f"滞后 {stale_days} 天",
    )

    contract = health.get("contract_check", {})
    if contract:
        st.write("**\u5408\u7ea6\u6d41\u52a8\u6027 / \u79fb\u4ed3\u98ce\u9669**")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("\u5408\u7ea6\u72b6\u6001", contract.get("label", "n/a"), contract.get("symbol", "n/a"))
        k2.metric("\u6210\u4ea4\u91cf / OI", contract.get("liquidity_summary", "n/a"), f"score {contract.get('score', 'n/a')}")
        k3.metric("\u4e3b\u529b\u8fde\u7eed\u6027", contract.get("continuity_summary", "n/a"), contract.get("roll_summary", "n/a"))
        k4.metric("\u6700\u65b0\u4ea4\u6613\u65e5", contract.get("date_summary", "n/a"), "n/a" if contract.get("stale_days") is None else f"lag {contract.get('stale_days')}d")

        selector_summary = contract.get("selector_summary", "")
        selector_candidates = contract.get("selector_candidates", [])
        if selector_summary:
            st.caption(f"候选合约选择：{selector_summary}")
        if selector_candidates:
            with st.expander("候选合约比较", expanded=False):
                st.dataframe(pd.DataFrame(selector_candidates), hide_index=True, width="stretch")

    if health["blockers"]:
        st.error(" / ".join(health["blockers"]))
    if health["warnings"]:
        st.warning(" / ".join(health["warnings"]))

def build_position_plan(
    price_signal: dict[str, Any],
    commodity: dict[str, Any],
    account_size: float,
    risk_pct: float,
    max_margin_pct: float,
    trial_fraction: float,
    margin_rate: float,
) -> dict[str, Any]:
    entry = price_signal.get("entry")
    stop = price_signal.get("stop")
    latest = price_signal.get("latest")
    multiplier = float(commodity.get("contract_multiplier", 10))
    if entry is None or stop is None or pd.isna(entry) or pd.isna(stop):
        return {"ok": False, "reason": "缺少入场价或止损价，无法计算仓位。"}

    entry = float(entry)
    stop = float(stop)
    if entry <= stop:
        return {"ok": False, "reason": "入场价不高于止损价，风险距离无效。"}
    if account_size <= 0 or risk_pct <= 0 or max_margin_pct <= 0:
        return {"ok": False, "reason": "账户资金、单笔风险或保证金上限未有效设置。"}

    atr = np.nan
    latest_close = np.nan
    if latest is not None:
        atr = float(latest["atr14"]) if "atr14" in latest and not pd.isna(latest["atr14"]) else np.nan
        latest_close = float(latest["close"]) if "close" in latest and not pd.isna(latest["close"]) else np.nan

    risk_per_unit = entry - stop
    risk_per_lot = risk_per_unit * multiplier
    risk_budget = account_size * risk_pct
    max_lots_by_risk = math.floor(risk_budget / risk_per_lot) if risk_per_lot > 0 else 0

    margin_per_lot = entry * multiplier * margin_rate
    margin_budget = account_size * max_margin_pct
    max_lots_by_margin = math.floor(margin_budget / margin_per_lot) if margin_per_lot > 0 else 0
    max_lots = max(0, min(max_lots_by_risk, max_lots_by_margin))
    trial_lots = math.floor(max_lots * trial_fraction)
    if max_lots > 0 and trial_lots < 1:
        trial_lots = 1

    used_risk = trial_lots * risk_per_lot
    used_margin = trial_lots * margin_per_lot
    notional = trial_lots * entry * multiplier
    stop_distance_pct = risk_per_unit / entry if entry > 0 else np.nan
    atr_multiple = risk_per_unit / atr if not pd.isna(atr) and atr > 0 else np.nan

    return {
        "ok": True,
        "entry": entry,
        "stop": stop,
        "latest_close": latest_close,
        "multiplier": multiplier,
        "margin_rate": margin_rate,
        "risk_per_unit": risk_per_unit,
        "risk_per_lot": risk_per_lot,
        "risk_budget": risk_budget,
        "max_lots_by_risk": max_lots_by_risk,
        "margin_per_lot": margin_per_lot,
        "margin_budget": margin_budget,
        "max_lots_by_margin": max_lots_by_margin,
        "max_lots": max_lots,
        "trial_fraction": trial_fraction,
        "trial_lots": trial_lots,
        "used_risk": used_risk,
        "used_margin": used_margin,
        "notional": notional,
        "stop_distance_pct": stop_distance_pct,
        "atr_multiple": atr_multiple,
        "risk_budget_pct": risk_budget / account_size if account_size > 0 else np.nan,
        "risk_usage_pct": used_risk / account_size if account_size > 0 else np.nan,
        "margin_usage_pct": used_margin / account_size if account_size > 0 else np.nan,
    }


def portfolio_lot_count(value: Any) -> int:
    numeric = numeric_value(value)
    if pd.isna(numeric):
        return 0
    return max(0, int(math.floor(numeric)))


def portfolio_return_correlation(snapshots: dict[str, dict[str, Any]], lookback_days: int = 60) -> dict[str, Any]:
    series_by_name: dict[str, pd.Series] = {}
    for name, snapshot in snapshots.items():
        data = snapshot.get("price_signal", {}).get("data")
        if not isinstance(data, pd.DataFrame) or data.empty or "date" not in data.columns or "close" not in data.columns:
            continue
        frame = data[["date", "close"]].copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame = frame.dropna(subset=["date", "close"]).sort_values("date")
        if frame.empty:
            continue
        returns = frame.set_index("date")["close"].pct_change(fill_method=None).dropna().tail(max(lookback_days, 20))
        if not returns.empty:
            series_by_name[name] = returns.rename(name)

    ordered_names = [name for name in COMMODITIES if name in series_by_name]
    if len(ordered_names) < 2:
        return {
            "pair": " / ".join(COMMODITIES.keys()),
            "correlation": np.nan,
            "samples": 0,
            "lookback_days": lookback_days,
            "label": "样本不足",
            "reason": "至少需要两个品种都有可比收盘收益率。",
        }

    joined = pd.concat([series_by_name[name] for name in ordered_names[:2]], axis=1).dropna().tail(lookback_days)
    if len(joined) < 20:
        return {
            "pair": " / ".join(ordered_names[:2]),
            "correlation": np.nan,
            "samples": int(len(joined)),
            "lookback_days": lookback_days,
            "label": "样本不足",
            "reason": "重叠收益率样本少于 20 个交易日，暂不使用相关性做硬约束。",
        }

    corr = float(joined[ordered_names[0]].corr(joined[ordered_names[1]]))
    label = "高相关" if not pd.isna(corr) and corr >= 0.65 else ("负相关/分散" if not pd.isna(corr) and corr <= 0 else "中低相关")
    return {
        "pair": " / ".join(ordered_names[:2]),
        "correlation": corr,
        "samples": int(len(joined)),
        "lookback_days": lookback_days,
        "label": label,
        "reason": f"使用最近 {int(len(joined))} 个重叠交易日的日收益率。",
    }


def portfolio_candidate_record(name: str, snapshot: dict[str, Any], risk_config: dict[str, float]) -> dict[str, Any]:
    plan = snapshot.get("position_plan", {}) or {}
    entry_playbook = snapshot.get("entry_playbook", {}) or {}
    post_entry = snapshot.get("post_entry_playbook", {}) or {}
    health = snapshot.get("signal_health", {}) or {}
    commodity = snapshot.get("commodity_config", COMMODITIES.get(name, {})) or {}

    plan_ok = bool(plan.get("ok"))
    stage = str(entry_playbook.get("current_stage", ""))
    health_gate = str(health.get("gate", "pass"))
    trial_lots = portfolio_lot_count(plan.get("trial_lots", 0)) if plan_ok else 0
    actual_position = bool(post_entry.get("actual_position"))
    actual_lots = portfolio_lot_count(post_entry.get("position_lots", 0)) if actual_position else 0

    requested_new_lots = 0
    if plan_ok and trial_lots > 0 and health_gate != "block" and stage in {"试仓", "加仓"}:
        requested_new_lots = 0 if actual_position and stage != "加仓" else trial_lots

    multiplier = numeric_value(plan.get("multiplier", commodity.get("contract_multiplier", 10)))
    if pd.isna(multiplier) or multiplier <= 0:
        multiplier = 10.0
    margin_rate = numeric_value(plan.get("margin_rate", risk_config.get("margin_rate", commodity.get("default_margin_rate", 0.12))))
    if pd.isna(margin_rate) or margin_rate <= 0:
        margin_rate = 0.12

    risk_per_lot = numeric_value(plan.get("risk_per_lot", np.nan))
    if pd.isna(risk_per_lot) or risk_per_lot < 0:
        risk_per_lot = 0.0
    margin_per_lot = numeric_value(plan.get("margin_per_lot", np.nan))
    entry = numeric_value(plan.get("entry", np.nan))
    if (pd.isna(margin_per_lot) or margin_per_lot < 0) and not pd.isna(entry):
        margin_per_lot = entry * multiplier * margin_rate
    if pd.isna(margin_per_lot) or margin_per_lot < 0:
        margin_per_lot = 0.0

    actual_avg = numeric_value(post_entry.get("position_avg_entry", plan.get("entry", np.nan)))
    stop_reference = numeric_value(post_entry.get("trailing_stop", plan.get("stop", np.nan)))
    actual_risk_per_lot = np.nan
    if actual_position and not pd.isna(actual_avg) and not pd.isna(stop_reference):
        actual_risk_per_lot = max(0.0, (actual_avg - stop_reference) * multiplier)
    if pd.isna(actual_risk_per_lot):
        actual_risk_per_lot = risk_per_lot

    actual_margin_per_lot = margin_per_lot
    if actual_position and not pd.isna(actual_avg) and actual_avg > 0:
        actual_margin_per_lot = actual_avg * multiplier * margin_rate

    return {
        "commodity": name,
        "stage": stage,
        "health_gate": health_gate,
        "plan_ok": plan_ok,
        "actual_position": actual_position,
        "actual_lots": actual_lots,
        "requested_new_lots": requested_new_lots,
        "approved_new_lots": 0,
        "risk_per_new_lot": risk_per_lot,
        "margin_per_new_lot": margin_per_lot,
        "current_risk": actual_lots * actual_risk_per_lot,
        "current_margin": actual_lots * actual_margin_per_lot,
        "requested_new_risk": requested_new_lots * risk_per_lot,
        "requested_new_margin": requested_new_lots * margin_per_lot,
        "approved_new_risk": 0.0,
        "approved_new_margin": 0.0,
        "combined_score": numeric_value(snapshot.get("combined", np.nan)),
        "weather_score": numeric_value(snapshot.get("weather_score", np.nan)),
        "price_score": numeric_value(snapshot.get("price_signal", {}).get("score", np.nan)),
        "decision": "无新增需求",
        "status": "pass",
        "reason": "当前不处于试仓或加仓触发阶段。",
    }


def build_portfolio_stacking_gate(
    snapshots: dict[str, dict[str, Any]],
    risk_config: dict[str, float],
) -> dict[str, Any]:
    account_size = numeric_value(risk_config.get("account_size", np.nan))
    risk_pct = numeric_value(risk_config.get("risk_pct", np.nan))
    max_margin_pct = numeric_value(risk_config.get("max_margin_pct", np.nan))
    risk_multiplier = numeric_value(risk_config.get("portfolio_risk_multiplier", 1.5))
    margin_multiplier = numeric_value(risk_config.get("portfolio_margin_multiplier", 1.5))
    correlation_trigger = numeric_value(risk_config.get("correlation_trigger", 0.65))
    correlation_lookback = portfolio_lot_count(risk_config.get("correlation_lookback", 60)) or 60

    if pd.isna(risk_multiplier) or risk_multiplier <= 0:
        risk_multiplier = 1.5
    if pd.isna(margin_multiplier) or margin_multiplier <= 0:
        margin_multiplier = 1.5
    if pd.isna(correlation_trigger):
        correlation_trigger = 0.65

    portfolio_risk_cap = account_size * risk_pct * risk_multiplier if not pd.isna(account_size) and not pd.isna(risk_pct) else np.nan
    portfolio_margin_cap = account_size * max_margin_pct * margin_multiplier if not pd.isna(account_size) and not pd.isna(max_margin_pct) else np.nan
    correlation = portfolio_return_correlation(snapshots, correlation_lookback)
    records = {name: portfolio_candidate_record(name, snapshot, risk_config) for name, snapshot in snapshots.items()}

    current_risk = sum(float(record["current_risk"]) for record in records.values())
    current_margin = sum(float(record["current_margin"]) for record in records.values())
    candidates = [record for record in records.values() if record["requested_new_lots"] > 0]

    corr_value = numeric_value(correlation.get("correlation", np.nan))
    high_correlation_stack = len(candidates) > 1 and not pd.isna(corr_value) and corr_value >= correlation_trigger
    ranked = sorted(
        candidates,
        key=lambda record: (
            -1 if pd.isna(record["combined_score"]) else record["combined_score"],
            -1 if pd.isna(record["weather_score"]) else record["weather_score"],
            -1 if pd.isna(record["price_score"]) else record["price_score"],
        ),
        reverse=True,
    )
    primary_name = ranked[0]["commodity"] if high_correlation_stack and ranked else ""

    approved_risk = current_risk
    approved_margin = current_margin
    current_over_risk = not pd.isna(portfolio_risk_cap) and current_risk > portfolio_risk_cap
    current_over_margin = not pd.isna(portfolio_margin_cap) and current_margin > portfolio_margin_cap

    for record in ranked:
        requested = int(record["requested_new_lots"])
        risk_per_lot = float(record["risk_per_new_lot"])
        margin_per_lot = float(record["margin_per_new_lot"])
        if pd.isna(portfolio_risk_cap) or pd.isna(portfolio_margin_cap) or account_size <= 0:
            approved = 0
            reason = "账户权益或组合上限无效，不能批准新增仓位。"
        elif current_over_risk or current_over_margin:
            approved = 0
            reason = "已有持仓风险或保证金已经超过组合上限，先降风险再新增。"
        elif high_correlation_stack and record["commodity"] != primary_name:
            approved = 0
            reason = f"{correlation['pair']} 相关性 {corr_value:.2f} >= {correlation_trigger:.2f}，只允许强信号 {primary_name} 先扩仓。"
        else:
            remaining_risk = max(0.0, portfolio_risk_cap - approved_risk)
            remaining_margin = max(0.0, portfolio_margin_cap - approved_margin)
            max_by_risk = math.floor(remaining_risk / risk_per_lot) if risk_per_lot > 0 else 0
            max_by_margin = math.floor(remaining_margin / margin_per_lot) if margin_per_lot > 0 else 0
            approved = max(0, min(requested, max_by_risk, max_by_margin))
            if approved >= requested:
                reason = "组合风险和保证金仍在上限内。"
            elif approved > 0:
                reason = f"组合剩余额度只支持新增 {approved} 手，低于单品种计划 {requested} 手。"
            else:
                reason = "组合风险或保证金剩余额度不足，不能新增。"

        record["approved_new_lots"] = approved
        record["approved_new_risk"] = approved * risk_per_lot
        record["approved_new_margin"] = approved * margin_per_lot
        if approved <= 0:
            record["decision"] = "禁止新增"
            record["status"] = "block"
        elif approved < requested:
            record["decision"] = "削减新增"
            record["status"] = "caution"
        else:
            record["decision"] = "允许新增"
            record["status"] = "pass"
        record["reason"] = reason
        approved_risk += record["approved_new_risk"]
        approved_margin += record["approved_new_margin"]

    requested_new_risk = sum(float(record["requested_new_risk"]) for record in records.values())
    requested_new_margin = sum(float(record["requested_new_margin"]) for record in records.values())
    approved_new_risk = sum(float(record["approved_new_risk"]) for record in records.values())
    approved_new_margin = sum(float(record["approved_new_margin"]) for record in records.values())
    risk_if_requested = current_risk + requested_new_risk
    margin_if_requested = current_margin + requested_new_margin
    risk_if_approved = current_risk + approved_new_risk
    margin_if_approved = current_margin + approved_new_margin

    blockers: list[str] = []
    warnings: list[str] = []
    if current_over_risk or current_over_margin:
        blockers.append("已有持仓风险或保证金超过组合上限。")
    if high_correlation_stack:
        blockers.append(f"同向信号相关性过高：{correlation['pair']} corr={corr_value:.2f}，禁止两边同时扩仓。")
    if not pd.isna(portfolio_risk_cap) and risk_if_requested > portfolio_risk_cap:
        warnings.append("若全部执行单品种计划，组合止损风险会超过上限。")
    if not pd.isna(portfolio_margin_cap) and margin_if_requested > portfolio_margin_cap:
        warnings.append("若全部执行单品种计划，组合保证金占用会超过上限。")
    if correlation.get("label") == "样本不足":
        warnings.append(str(correlation.get("reason", "相关性样本不足。")))

    candidate_statuses = [record["status"] for record in records.values() if record["requested_new_lots"] > 0]
    if "block" in candidate_statuses or blockers:
        status = "block"
        label = "组合限制"
    elif "caution" in candidate_statuses or warnings:
        status = "caution"
        label = "组合谨慎"
    elif candidates:
        status = "pass"
        label = "组合通过"
    else:
        status = "pass"
        label = "暂无新增"

    account = account_size if not pd.isna(account_size) and account_size > 0 else np.nan
    return {
        "status": status,
        "label": label,
        "primary_signal": primary_name,
        "risk_cap": portfolio_risk_cap,
        "margin_cap": portfolio_margin_cap,
        "risk_cap_pct": portfolio_risk_cap / account if not pd.isna(account) and not pd.isna(portfolio_risk_cap) else np.nan,
        "margin_cap_pct": portfolio_margin_cap / account if not pd.isna(account) and not pd.isna(portfolio_margin_cap) else np.nan,
        "current_risk": current_risk,
        "current_margin": current_margin,
        "requested_new_risk": requested_new_risk,
        "requested_new_margin": requested_new_margin,
        "approved_new_risk": approved_new_risk,
        "approved_new_margin": approved_new_margin,
        "risk_if_requested": risk_if_requested,
        "margin_if_requested": margin_if_requested,
        "risk_if_approved": risk_if_approved,
        "margin_if_approved": margin_if_approved,
        "risk_if_requested_pct": risk_if_requested / account if not pd.isna(account) else np.nan,
        "margin_if_requested_pct": margin_if_requested / account if not pd.isna(account) else np.nan,
        "risk_if_approved_pct": risk_if_approved / account if not pd.isna(account) else np.nan,
        "margin_if_approved_pct": margin_if_approved / account if not pd.isna(account) else np.nan,
        "correlation": correlation,
        "correlation_trigger": correlation_trigger,
        "blockers": blockers,
        "warnings": warnings,
        "commodities": records,
    }


def apply_portfolio_gate_to_snapshots(
    snapshots: dict[str, dict[str, Any]],
    risk_config: dict[str, float],
) -> dict[str, Any]:
    portfolio = build_portfolio_stacking_gate(snapshots, risk_config)
    commodity_gates = portfolio.get("commodities", {})
    for name, snapshot in snapshots.items():
        gate = commodity_gates.get(name, portfolio_candidate_record(name, snapshot, risk_config))
        snapshot["portfolio_gate"] = gate
        snapshot["portfolio_summary"] = portfolio
        plan = snapshot.get("position_plan", {}) or {}
        plan["portfolio_requested_new_lots"] = gate.get("requested_new_lots", 0)
        plan["portfolio_approved_new_lots"] = gate.get("approved_new_lots", 0)
        plan["portfolio_decision"] = gate.get("decision", "无新增需求")
        plan["portfolio_reason"] = gate.get("reason", "")
        entry_playbook = snapshot.get("entry_playbook", {}) or {}
        entry_playbook["portfolio_gate"] = gate

        requested = int(gate.get("requested_new_lots", 0) or 0)
        approved = int(gate.get("approved_new_lots", 0) or 0)
        if requested <= 0:
            continue

        portfolio_note = f"组合闸门：{gate.get('decision', 'n/a')}，申请 {requested} 手，批准 {approved} 手；{gate.get('reason', '')}"
        entry_playbook["next_trigger"] = f"{entry_playbook.get('next_trigger', 'n/a')} {portfolio_note}"
        session = snapshot.get("market_session", {}) or {}
        session_allows_action = bool(session.get("action_allowed", True))
        if not session_allows_action:
            snapshot["note"] = f"{snapshot.get('note', '')} {portfolio_note}"
            for row in entry_playbook.get("rows", []):
                if row.get("阶段") in {"试仓", "加仓"} and row.get("状态") == "已触发":
                    row["状态"] = "时段限制"
                    row["执行动作"] = f"交易时段闸门未放行：{session.get('label', 'n/a')}；{session.get('reason', '')}"
            continue
        for row in entry_playbook.get("rows", []):
            if row.get("阶段") not in {"试仓", "加仓"}:
                continue
            if approved <= 0:
                if row.get("状态") == "已触发":
                    row["状态"] = "组合限制"
                row["执行动作"] = f"组合闸门禁止新增仓位；{gate.get('reason', '')}"
            elif approved < requested:
                row["执行动作"] = f"组合闸门只批准新增 {approved} 手，低于单品种计划 {requested} 手；其余等待下一次复核。"

        current_stage = entry_playbook.get("current_stage", "")
        if current_stage in {"试仓", "加仓"} and approved <= 0:
            entry_playbook["current_stage"] = "组合限制"
            snapshot["action"] = "组合限仓"
            snapshot["note"] = f"{snapshot.get('note', '')} {portfolio_note}"
        elif current_stage in {"试仓", "加仓"} and approved < requested:
            snapshot["action"] = "限额试仓" if current_stage == "试仓" else "限额加仓"
            snapshot["note"] = f"{snapshot.get('note', '')} {portfolio_note}"
    return portfolio


def portfolio_summary_from_snapshots(snapshots: dict[str, dict[str, Any]]) -> dict[str, Any]:
    for snapshot in snapshots.values():
        summary = snapshot.get("portfolio_summary")
        if isinstance(summary, dict) and summary:
            return summary
    return {}

def render_position_sizing(plan: dict[str, Any], unit: str) -> None:
    st.subheader("仓位与风险预算")
    if not plan.get("ok"):
        st.warning(plan.get("reason", "仓位计算不可用。"))
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("每手止损风险", f"{number_text(plan['risk_per_lot'], 0)} 元", f"{number_text(plan['risk_per_unit'], 0)} {unit}")
    c2.metric("账户风险预算", f"{number_text(plan['risk_budget'], 0)} 元", f"上限 {plan['risk_budget_pct']:.2%}")
    c3.metric("建议试仓手数", f"{plan['trial_lots']} 手", f"风险上限 {plan['max_lots_by_risk']} 手")
    c4.metric("保证金占用", f"{number_text(plan['used_margin'], 0)} 元", f"上限 {plan['max_lots_by_margin']} 手")

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("试仓名义市值", f"{number_text(plan['notional'], 0)} 元")
    d2.metric("试仓实际风险", f"{number_text(plan['used_risk'], 0)} 元", f"{plan['risk_usage_pct']:.2%}")
    d3.metric("保证金占账户", f"{plan['margin_usage_pct']:.2%}", f"保证金率 {plan['margin_rate']:.1%}")
    d4.metric(
        "止损距离",
        f"{plan['stop_distance_pct']:.2%}",
        "n/a" if pd.isna(plan["atr_multiple"]) else f"{plan['atr_multiple']:.1f} ATR",
    )

    if plan["trial_lots"] <= 0:
        st.error("按当前风险预算无法开 1 手；需要降低风险距离、提高风险预算，或放弃本次交易。")
    elif plan["max_lots_by_margin"] < plan["max_lots_by_risk"]:
        st.warning("保证金上限先于止损风险约束生效；不要把手数加到风险预算以外。")
    else:
        st.info("手数按止损风险和保证金占用双重约束计算；实际下单还需检查交易所/券商最新保证金。")

def log_scalar(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if isinstance(value, float) and pd.isna(value):
        return ""
    return value


def build_signal_log_row(
    selected: str,
    commodity: dict[str, Any],
    anchor: dt.date,
    action: str,
    note: str,
    combined: float,
    weather_score: float,
    price_signal: dict[str, Any],
    signal_health: dict[str, Any],
    regime_context: dict[str, Any],
    position_plan: dict[str, Any],
    weather: pd.DataFrame,
    price_source: str,
    ifind_symbol: str,
    entry_playbook: dict[str, Any] | None = None,
    post_entry_playbook: dict[str, Any] | None = None,
    portfolio_gate: dict[str, Any] | None = None,
    portfolio_summary: dict[str, Any] | None = None,
    market_session: dict[str, Any] | None = None,
) -> dict[str, Any]:
    latest = price_signal.get("latest")
    latest_price_date = ""
    latest_close = np.nan
    if latest is not None:
        latest_price_date = log_scalar(latest["date"]) if "date" in latest else ""
        latest_close = float(latest["close"]) if "close" in latest and not pd.isna(latest["close"]) else np.nan

    weather_pressure_score = weather_pressure_score_for(selected, weather)
    top_region = ""
    top_driver = ""
    top_weather_score = np.nan
    top_weather_entry_score = np.nan
    top_weather_impact = ""
    top_weather_impact_reason = ""
    if not weather.empty and "commodity" in weather.columns:
        subset = weather.loc[weather["commodity"] == selected].copy()
        if not subset.empty:
            sort_column = "entry_weighted_score" if "entry_weighted_score" in subset.columns else "weighted_score"
            row = subset.sort_values(sort_column, ascending=False).iloc[0]
            top_region = row.get("region", "")
            top_driver = row.get("driver", "")
            top_weather_score = float(row.get("stress_score", np.nan))
            top_weather_entry_score = numeric_value(row.get("entry_ready_score", row.get("stress_score", np.nan)))
            top_weather_impact = row.get("impact_label", "")
            top_weather_impact_reason = row.get("impact_reason", "")

    contract = signal_health.get("contract_check", {})
    enso = regime_context.get("enso", {})
    entry_playbook = entry_playbook or {}
    post_entry_playbook = post_entry_playbook or {}
    portfolio_gate = portfolio_gate or {}
    portfolio_summary = portfolio_summary or {}
    market_session = market_session or {}
    portfolio_correlation = portfolio_summary.get("correlation", {}) or {}
    return {
        "run_timestamp": dt.datetime.now(APP_TZ).isoformat(timespec="seconds"),
        "signal_date": anchor.isoformat(),
        "commodity": selected,
        "price_source": price_source,
        "weather_source_status": weather.attrs.get("data_source_status", "live"),
        "weather_source_message": weather.attrs.get("data_source_message", ""),
        "price_source_status": price_signal.get("data", pd.DataFrame()).attrs.get("data_source_status", "live") if isinstance(price_signal.get("data"), pd.DataFrame) else "",
        "price_source_message": price_signal.get("data", pd.DataFrame()).attrs.get("data_source_message", "") if isinstance(price_signal.get("data"), pd.DataFrame) else "",
        "ifind_symbol": ifind_symbol,
        "action": action,
        "market_session_status": market_session.get("status", ""),
        "market_session_label": market_session.get("label", ""),
        "market_session_action_allowed": market_session.get("action_allowed", ""),
        "market_session_close_confirmed": market_session.get("close_confirmed", ""),
        "market_session_latest_price_date": log_scalar(market_session.get("latest_price_date", "")),
        "market_session_reason": market_session.get("reason", ""),
        "playbook_stage": entry_playbook.get("current_stage", ""),
        "playbook_next_trigger": entry_playbook.get("next_trigger", ""),
        "playbook_invalidate": entry_playbook.get("invalidate", ""),
        "management_stage": post_entry_playbook.get("current_management", ""),
        "management_r_multiple": "" if pd.isna(post_entry_playbook.get("r_multiple", np.nan)) else round(float(post_entry_playbook.get("r_multiple")), 2),
        "management_trailing_stop": log_scalar(post_entry_playbook.get("trailing_stop", "")),
        "management_partial_exit": log_scalar(post_entry_playbook.get("partial_exit_level", "")),
        "management_time_stop_days": post_entry_playbook.get("time_stop_days", ""),
        "management_time_stop_triggered": post_entry_playbook.get("time_stop_triggered", ""),
        "position_actual": post_entry_playbook.get("actual_position", ""),
        "position_entry_date": post_entry_playbook.get("position_entry_date", ""),
        "position_avg_entry": log_scalar(post_entry_playbook.get("position_avg_entry", "")),
        "position_lots": post_entry_playbook.get("position_lots", ""),
        "position_holding_days": "" if post_entry_playbook.get("holding_days") is None else post_entry_playbook.get("holding_days"),
        "portfolio_status": portfolio_gate.get("status", ""),
        "portfolio_label": portfolio_summary.get("label", ""),
        "portfolio_decision": portfolio_gate.get("decision", ""),
        "portfolio_requested_new_lots": portfolio_gate.get("requested_new_lots", ""),
        "portfolio_approved_new_lots": portfolio_gate.get("approved_new_lots", ""),
        "portfolio_reason": portfolio_gate.get("reason", ""),
        "portfolio_risk_if_approved_pct": log_scalar(portfolio_summary.get("risk_if_approved_pct", "")),
        "portfolio_margin_if_approved_pct": log_scalar(portfolio_summary.get("margin_if_approved_pct", "")),
        "portfolio_pair_correlation": log_scalar(portfolio_correlation.get("correlation", "")),
        "portfolio_correlation_samples": portfolio_correlation.get("samples", ""),
        "note": note,
        "combined_score": round(float(combined), 2),
        "weather_score": "" if pd.isna(weather_score) else round(float(weather_score), 2),
        "weather_pressure_score": "" if pd.isna(weather_pressure_score) else round(float(weather_pressure_score), 2),
        "price_score": int(price_signal.get("score", 0)),
        "health_score": signal_health.get("score"),
        "health_gate": signal_health.get("gate"),
        "health_label": signal_health.get("label"),
        "contract_status": contract.get("status"),
        "contract_label": contract.get("label"),
        "contract_symbol": contract.get("symbol"),
        "contract_score": contract.get("score"),
        "contract_liquidity": contract.get("liquidity_summary"),
        "contract_continuity": contract.get("continuity_summary"),
        "contract_roll": contract.get("roll_summary"),
        "contract_selector": contract.get("selector_summary"),
        "contract_candidate_count": len(contract.get("selector_candidates", [])),
        "contract_warnings": " | ".join(contract.get("warnings", [])),
        "contract_blockers": " | ".join(contract.get("blockers", [])),
        "health_warnings": " | ".join(signal_health.get("warnings", [])),
        "health_blockers": " | ".join(signal_health.get("blockers", [])),
        "regime_label": regime_context.get("label"),
        "enso_phase": enso.get("phase"),
        "enso_confidence": enso.get("confidence"),
        "latest_oni": log_scalar(enso.get("latest_anom")),
        "top_weather_region": top_region,
        "top_weather_driver": top_driver,
        "top_weather_score": "" if pd.isna(top_weather_score) else round(top_weather_score, 2),
        "top_weather_entry_score": "" if pd.isna(top_weather_entry_score) else round(float(top_weather_entry_score), 2),
        "top_weather_impact": top_weather_impact,
        "top_weather_impact_reason": top_weather_impact_reason,
        "latest_price_date": latest_price_date,
        "latest_close": "" if pd.isna(latest_close) else round(latest_close, 2),
        "entry": log_scalar(price_signal.get("entry")),
        "stop": log_scalar(price_signal.get("stop")),
        "add_level": log_scalar(price_signal.get("add_level")),
        "trial_lots": position_plan.get("trial_lots", ""),
        "max_lots_by_risk": position_plan.get("max_lots_by_risk", ""),
        "max_lots_by_margin": position_plan.get("max_lots_by_margin", ""),
        "risk_per_lot": log_scalar(position_plan.get("risk_per_lot", "")),
        "used_risk": log_scalar(position_plan.get("used_risk", "")),
        "risk_usage_pct": log_scalar(position_plan.get("risk_usage_pct", "")),
        "used_margin": log_scalar(position_plan.get("used_margin", "")),
        "margin_usage_pct": log_scalar(position_plan.get("margin_usage_pct", "")),
        "stop_distance_pct": log_scalar(position_plan.get("stop_distance_pct", "")),
        "atr_multiple": log_scalar(position_plan.get("atr_multiple", "")),
        "contract_multiplier": commodity.get("contract_multiplier", ""),
    }


def load_signal_log() -> pd.DataFrame:
    if not os.path.exists(SIGNAL_LOG_PATH):
        return pd.DataFrame()
    try:
        return pd.read_csv(SIGNAL_LOG_PATH, keep_default_na=False)
    except Exception:
        return pd.DataFrame()


def append_signal_log(row: dict[str, Any]) -> int:
    os.makedirs(SIGNAL_LOG_DIR, exist_ok=True)
    current = load_signal_log()
    new_row = pd.DataFrame([row])
    if current.empty:
        updated = new_row
    else:
        updated = pd.concat([current, new_row], ignore_index=True)
    updated.to_csv(SIGNAL_LOG_PATH, index=False, encoding="utf-8-sig")
    return len(updated)



def parse_position_date(value: Any) -> dt.date | None:
    if value is None or value == "":
        return None
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    try:
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed.date()
    except Exception:
        return None


def empty_position_state(commodity: str) -> dict[str, Any]:
    return {
        "commodity": commodity,
        "has_position": False,
        "entry_date": "",
        "avg_entry_price": np.nan,
        "lots": 0,
        "notes": "",
        "updated_at": "",
    }


def normalize_position_state(row: dict[str, Any], commodity: str) -> dict[str, Any]:
    entry_date = parse_position_date(row.get("entry_date", ""))
    avg_entry_price = numeric_value(row.get("avg_entry_price", np.nan))
    lots = numeric_value(row.get("lots", 0))
    has_position_raw = str(row.get("has_position", "")).strip().lower()
    has_position = has_position_raw in {"1", "true", "yes", "y", "是"} or (
        not pd.isna(lots) and lots > 0 and not pd.isna(avg_entry_price) and avg_entry_price > 0
    )
    if pd.isna(lots) or lots < 0:
        lots = 0
    return {
        "commodity": commodity,
        "has_position": bool(has_position),
        "entry_date": entry_date.isoformat() if entry_date else "",
        "avg_entry_price": "" if pd.isna(avg_entry_price) else float(avg_entry_price),
        "lots": int(lots),
        "notes": str(row.get("notes", "")),
        "updated_at": str(row.get("updated_at", "")),
    }


def load_position_states() -> dict[str, dict[str, Any]]:
    states = {name: empty_position_state(name) for name in COMMODITIES}
    if not os.path.exists(POSITION_STATE_PATH):
        return states
    try:
        frame = pd.read_csv(POSITION_STATE_PATH, keep_default_na=False)
    except Exception:
        return states
    for _, row in frame.iterrows():
        commodity = str(row.get("commodity", ""))
        if commodity in states:
            states[commodity] = normalize_position_state(row.to_dict(), commodity)
    return states


def save_position_states(states: dict[str, dict[str, Any]]) -> None:
    os.makedirs(SIGNAL_LOG_DIR, exist_ok=True)
    rows = []
    for commodity in COMMODITIES:
        rows.append(normalize_position_state(states.get(commodity, {}), commodity))
    pd.DataFrame(rows).to_csv(POSITION_STATE_PATH, index=False, encoding="utf-8-sig")


def position_state_for(states: dict[str, dict[str, Any]] | None, commodity: str) -> dict[str, Any]:
    states = states or {}
    return normalize_position_state(states.get(commodity, {}), commodity)


def position_state_is_active(position_state: dict[str, Any] | None) -> bool:
    state = position_state or {}
    lots = numeric_value(state.get("lots", 0))
    avg_entry = numeric_value(state.get("avg_entry_price", np.nan))
    return bool(state.get("has_position")) and not pd.isna(lots) and lots > 0 and not pd.isna(avg_entry) and avg_entry > 0


def holding_days_for_position(position_state: dict[str, Any] | None, price_signal: dict[str, Any], anchor: dt.date) -> int | None:
    state = position_state or {}
    entry_date = parse_position_date(state.get("entry_date", ""))
    if entry_date is None:
        return None
    data = price_signal.get("data")
    if isinstance(data, pd.DataFrame) and not data.empty and "date" in data.columns:
        dates = pd.to_datetime(data["date"], errors="coerce").dt.date
        count = int(((dates >= entry_date) & (dates <= anchor)).sum())
        if count > 0:
            return max(0, count - 1)
    return max(0, (anchor - entry_date).days)


def render_position_state_editor(
    selected: str,
    states: dict[str, dict[str, Any]],
    anchor: dt.date,
) -> dict[str, dict[str, Any]]:
    current = position_state_for(states, selected)
    current_date = parse_position_date(current.get("entry_date", "")) or anchor
    current_avg = numeric_value(current.get("avg_entry_price", np.nan))
    current_lots = numeric_value(current.get("lots", 0))
    with st.expander("实际持仓状态", expanded=True):
        st.caption(f"持久化文件：{POSITION_STATE_PATH}")
        with st.form(f"position-state-{selected}"):
            has_position = st.checkbox("当前有试仓/持仓", value=position_state_is_active(current))
            entry_date = st.date_input("实际试仓日期", value=current_date)
            avg_entry_price = st.number_input(
                "实际试仓均价",
                min_value=0.0,
                value=0.0 if pd.isna(current_avg) else float(current_avg),
                step=1.0,
            )
            lots = st.number_input(
                "当前手数",
                min_value=0,
                value=0 if pd.isna(current_lots) else int(current_lots),
                step=1,
            )
            notes = st.text_area("持仓备注", value=str(current.get("notes", "")), height=70)
            saved = st.form_submit_button("保存持仓状态")
        if saved:
            states[selected] = normalize_position_state(
                {
                    "commodity": selected,
                    "has_position": bool(has_position),
                    "entry_date": entry_date.isoformat() if has_position else "",
                    "avg_entry_price": avg_entry_price if has_position else "",
                    "lots": lots if has_position else 0,
                    "notes": notes,
                    "updated_at": dt.datetime.now(APP_TZ).isoformat(timespec="seconds"),
                },
                selected,
            )
            save_position_states(states)
            st.success("持仓状态已保存。")
        refreshed = position_state_for(states, selected)
        holding_days = holding_days_for_position(refreshed, {"data": pd.DataFrame()}, anchor)
        if position_state_is_active(refreshed):
            st.write(
                f"当前记录：{refreshed['lots']} 手，均价 {number_text(refreshed['avg_entry_price'], 0)}，"
                f"试仓日 {refreshed['entry_date']}，持仓约 {holding_days if holding_days is not None else 'n/a'} 天。"
            )
        else:
            st.info("当前未记录实际试仓；持仓管理按预案显示，不自动触发时间止损。")
    return states
def numeric_value(value: Any) -> float:
    if value is None or value == "":
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def fmt_delta(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return "n/a"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.{digits}f}"


def threshold_crossing(prev: float, curr: float, threshold: float) -> str | None:
    if pd.isna(prev) or pd.isna(curr):
        return None
    if prev < threshold <= curr:
        return "up"
    if prev >= threshold > curr:
        return "down"
    return None


def previous_log_row(log: pd.DataFrame, commodity: str) -> dict[str, Any] | None:
    if log.empty or "commodity" not in log.columns:
        return None
    subset = log.loc[log["commodity"] == commodity]
    if subset.empty:
        return None
    return subset.iloc[-1].to_dict()


def build_change_summary(
    current: dict[str, Any],
    previous: dict[str, Any] | None,
    settings: RuleSettings,
) -> tuple[list[str], list[str], pd.DataFrame]:
    if previous is None:
        return ["当前品种暂无上一条运行日志；写入一次后即可开始比较变化。"], [], pd.DataFrame()

    alerts: list[str] = []
    notes: list[str] = []
    rows: list[dict[str, Any]] = []

    def add_metric(label: str, key: str, material_abs: float | None = None, as_pct: bool = False) -> tuple[float, float, float]:
        prev = numeric_value(previous.get(key))
        curr = numeric_value(current.get(key))
        delta = curr - prev if not (pd.isna(prev) or pd.isna(curr)) else np.nan
        if as_pct:
            prev_text = "n/a" if pd.isna(prev) else f"{prev:.2%}"
            curr_text = "n/a" if pd.isna(curr) else f"{curr:.2%}"
            delta_text = "n/a" if pd.isna(delta) else f"{delta:+.2%}"
        else:
            prev_text = "n/a" if pd.isna(prev) else f"{prev:.2f}"
            curr_text = "n/a" if pd.isna(curr) else f"{curr:.2f}"
            delta_text = fmt_delta(delta, 2)
        rows.append({"指标": label, "上次": prev_text, "当前": curr_text, "变化": delta_text})
        if material_abs is not None and not pd.isna(delta) and abs(delta) >= material_abs:
            direction = "上升" if delta > 0 else "下降"
            alerts.append(f"{label}{direction} {fmt_delta(delta, 1)}，超过变动阈值 {material_abs:g}。")
        return prev, curr, delta

    if previous.get("action") != current.get("action"):
        alerts.append(f"行动状态变化：{previous.get('action', 'n/a')} -> {current.get('action', 'n/a')}。")
    if previous.get("playbook_stage") != current.get("playbook_stage"):
        alerts.append(f"执行阶段变化：{previous.get('playbook_stage', 'n/a')} -> {current.get('playbook_stage', 'n/a')}。")
    if previous.get("management_stage") != current.get("management_stage"):
        alerts.append(f"持仓管理变化：{previous.get('management_stage', 'n/a')} -> {current.get('management_stage', 'n/a')}。")
    if previous.get("portfolio_decision") != current.get("portfolio_decision"):
        alerts.append(f"组合闸门变化：{previous.get('portfolio_decision', 'n/a')} -> {current.get('portfolio_decision', 'n/a')}。")
    if previous.get("health_gate") != current.get("health_gate"):
        alerts.append(f"数据质量闸门变化：{previous.get('health_gate', 'n/a')} -> {current.get('health_gate', 'n/a')}。")
    if previous.get("regime_label") != current.get("regime_label"):
        notes.append(f"Regime 背景变化：{previous.get('regime_label', 'n/a')} -> {current.get('regime_label', 'n/a')}。")
    if previous.get("enso_phase") != current.get("enso_phase"):
        notes.append(f"ENSO 状态变化：{previous.get('enso_phase', 'n/a')} -> {current.get('enso_phase', 'n/a')}。")

    prev_combined, curr_combined, _ = add_metric("综合分", "combined_score", material_abs=8)
    prev_weather, curr_weather, _ = add_metric("天气分", "weather_score", material_abs=10)
    prev_price, curr_price, _ = add_metric("价格确认分", "price_score", material_abs=15)
    add_metric("健康度", "health_score", material_abs=10)
    add_metric("最新收盘", "latest_close", material_abs=None)
    add_metric("突破试仓价", "entry", material_abs=None)
    add_metric("防守止损", "stop", material_abs=None)
    add_metric("试仓手数", "trial_lots", material_abs=1)
    add_metric("试仓风险占账户", "risk_usage_pct", material_abs=0.0025, as_pct=True)
    add_metric("组合批准新增手数", "portfolio_approved_new_lots", material_abs=1)
    add_metric("批准后组合风险", "portfolio_risk_if_approved_pct", material_abs=0.0025, as_pct=True)

    for label, prev, curr, threshold in [
        ("天气分门槛", prev_weather, curr_weather, settings.weather_trigger),
        ("价格确认门槛", prev_price, curr_price, settings.price_trigger),
        ("综合建仓门槛", prev_combined, curr_combined, settings.build_trigger),
    ]:
        crossing = threshold_crossing(prev, curr, threshold)
        if crossing == "up":
            alerts.append(f"{label}向上穿越 {threshold}。")
        elif crossing == "down":
            alerts.append(f"{label}跌回 {threshold} 下方。")

    if not alerts:
        notes.append("较上一条同品种记录未出现重大阈值穿越或材料级分数变化。")
    return alerts, notes, pd.DataFrame(rows)


def render_change_summary(row: dict[str, Any], log: pd.DataFrame, settings: RuleSettings) -> None:
    st.write("**变化提醒**")
    previous = previous_log_row(log, str(row.get("commodity", "")))
    alerts, notes, summary = build_change_summary(row, previous, settings)
    for alert in alerts:
        st.warning(alert)
    for note in notes:
        st.info(note)
    if not summary.empty:
        st.dataframe(summary, hide_index=True, width="stretch")

def is_actionable_trade_day(row: dict[str, Any]) -> bool:
    action = str(row.get("action", ""))
    stage = str(row.get("playbook_stage", ""))
    requested_lots = numeric_value(row.get("portfolio_requested_new_lots", np.nan))
    approved_lots = numeric_value(row.get("portfolio_approved_new_lots", np.nan))
    build_actions = {"开始试仓", "限额试仓", "限额加仓"}
    return (
        action in build_actions
        or stage in {"试仓", "加仓"}
        or (not pd.isna(requested_lots) and requested_lots > 0)
        or (not pd.isna(approved_lots) and approved_lots > 0)
    )


def pre_trade_item(label: str, passed: bool, detail: str, required: bool = True) -> dict[str, Any]:
    return {
        "项目": label,
        "状态": "通过" if passed else ("阻断" if required else "提示"),
        "必需": "是" if required else "否",
        "说明": detail,
        "passed": bool(passed),
        "required": bool(required),
    }


def build_pre_trade_checklist(row: dict[str, Any]) -> dict[str, Any]:
    actionable = is_actionable_trade_day(row)
    weather_status = str(row.get("weather_source_status", "live") or "live")
    price_status = str(row.get("price_source_status", "live") or "live")
    health_gate = str(row.get("health_gate", ""))
    contract_status = str(row.get("contract_status", ""))
    portfolio_status = str(row.get("portfolio_status", ""))
    market_session_allowed = truthy_log_value(row.get("market_session_action_allowed", ""))
    market_session_label = str(row.get("market_session_label", "n/a") or "n/a")
    planned_lots = numeric_value(row.get("portfolio_approved_new_lots", np.nan))
    if pd.isna(planned_lots) or planned_lots <= 0:
        planned_lots = numeric_value(row.get("trial_lots", np.nan))
    stop = numeric_value(row.get("stop", np.nan))

    items = [
        pre_trade_item(
            "数据状态",
            health_gate != "block",
            f"天气 {weather_status}，行情 {price_status}，健康闸门 {row.get('health_label', health_gate)}。",
        ),
        pre_trade_item(
            "合约状态",
            contract_status not in {"block", "fail"} and str(row.get("contract_label", "")) not in {"阻断", "不可用"},
            f"{row.get('contract_label', 'n/a')}；{row.get('contract_symbol', 'n/a')}；{row.get('contract_liquidity', 'n/a')}。",
        ),
        pre_trade_item(
            "组合闸门",
            portfolio_status != "block",
            f"{row.get('portfolio_decision', 'n/a')}；申请 {row.get('portfolio_requested_new_lots', 0)} 手，批准 {row.get('portfolio_approved_new_lots', 0)} 手。",
        ),
        pre_trade_item(
            "交易时段",
            market_session_allowed,
            f"{market_session_label}；{row.get('market_session_reason', 'n/a')}；最新行情日 {row.get('market_session_latest_price_date', 'n/a')}。",
        ),
        pre_trade_item(
            "计划手数",
            not pd.isna(planned_lots) and planned_lots > 0,
            f"计划/批准手数 {0 if pd.isna(planned_lots) else int(planned_lots)} 手；单品种试仓 {row.get('trial_lots', 'n/a')} 手。",
        ),
        pre_trade_item(
            "失效价位",
            not pd.isna(stop) and stop > 0,
            f"防守止损/失效参考 {number_text(stop, 0)}；Playbook 失效：{row.get('playbook_invalidate', 'n/a')}。",
        ),
    ]
    blockers = [item["项目"] for item in items if item["required"] and not item["passed"]]
    return {
        "actionable": actionable,
        "items": items,
        "system_passed": len(blockers) == 0,
        "blockers": blockers,
        "planned_lots": 0 if pd.isna(planned_lots) else int(planned_lots),
        "invalidation_level": "" if pd.isna(stop) else stop,
    }


def render_pre_trade_checklist(row: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    checklist = build_pre_trade_checklist(row)
    updated = dict(row)
    st.write("**盘前交易 Checklist / 人工确认**")
    st.caption("只有试仓/加仓类可行动信号需要人工确认；观察类信号可直接记录为研究快照。")
    table = pd.DataFrame([{k: v for k, v in item.items() if k not in {"passed", "required"}} for item in checklist["items"]])
    st.dataframe(table, hide_index=True, width="stretch")

    actionable = bool(checklist["actionable"])
    if actionable and checklist["blockers"]:
        st.error("系统检查未通过：" + " / ".join(checklist["blockers"]))
    elif actionable:
        st.warning("当前为可行动交易日；写入日志前需要操作员确认数据、合约、组合闸门、手数和失效位。")
    else:
        st.info("当前不是试仓/加仓类可行动信号；日志将按研究快照记录。")

    key_base = f"pretrade-{row.get('signal_date', '')}-{row.get('commodity', '')}"
    operator = st.text_input("操作员", value="", key=f"{key_base}-operator") if actionable else ""
    approved = st.checkbox("我已复核数据状态、合约状态、组合闸门、计划手数和失效价位", value=False, key=f"{key_base}-approved") if actionable else False
    notes = st.text_area("确认备注", value="", height=70, key=f"{key_base}-notes") if actionable else ""

    can_log = (not actionable) or (checklist["system_passed"] and approved and bool(operator.strip()))
    status = "not_required"
    if actionable:
        if can_log:
            status = "approved"
        elif not checklist["system_passed"]:
            status = "blocked"
        else:
            status = "pending"

    updated.update(
        {
            "pretrade_actionable": actionable,
            "pretrade_checklist_status": status,
            "pretrade_system_passed": checklist["system_passed"],
            "pretrade_manual_approved": bool(approved),
            "pretrade_operator": operator.strip(),
            "pretrade_approved_at": dt.datetime.now(APP_TZ).isoformat(timespec="seconds") if can_log and actionable else "",
            "pretrade_planned_lots": checklist["planned_lots"],
            "pretrade_invalidation_level": log_scalar(checklist["invalidation_level"]),
            "pretrade_blockers": " | ".join(checklist["blockers"]),
            "pretrade_notes": notes.strip(),
            "pretrade_items": json.dumps(checklist["items"], ensure_ascii=False, default=str),
        }
    )
    return updated, can_log

def truthy_log_value(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y", "是", "approved"}


def post_trade_signal_id(row: dict[str, Any]) -> str:
    timestamp = str(row.get("run_timestamp", "")).replace(":", "-")
    return cache_key_slug(f"{row.get('signal_date', '')}_{row.get('commodity', '')}_{timestamp}")


def load_post_trade_notes() -> pd.DataFrame:
    if not os.path.exists(POST_TRADE_NOTES_PATH):
        return pd.DataFrame()
    try:
        return pd.read_csv(POST_TRADE_NOTES_PATH, keep_default_na=False)
    except Exception:
        return pd.DataFrame()


def upsert_post_trade_note(note: dict[str, Any]) -> int:
    os.makedirs(SIGNAL_LOG_DIR, exist_ok=True)
    current = load_post_trade_notes()
    note_frame = pd.DataFrame([note])
    if current.empty or "signal_id" not in current.columns:
        updated = note_frame
    else:
        kept = current.loc[current["signal_id"].astype(str) != str(note.get("signal_id", ""))]
        updated = pd.concat([kept, note_frame], ignore_index=True)
    updated.to_csv(POST_TRADE_NOTES_PATH, index=False, encoding="utf-8-sig")
    return len(updated)


def note_for_signal(notes: pd.DataFrame, signal_id: str) -> dict[str, Any]:
    if notes.empty or "signal_id" not in notes.columns:
        return {}
    subset = notes.loc[notes["signal_id"].astype(str) == str(signal_id)]
    if subset.empty:
        return {}
    return subset.iloc[-1].to_dict()


def signed_action_rows(log: pd.DataFrame) -> pd.DataFrame:
    if log.empty:
        return pd.DataFrame()
    data = log.copy()
    if "pretrade_actionable" not in data.columns or "pretrade_checklist_status" not in data.columns:
        return pd.DataFrame()
    mask = data["pretrade_actionable"].map(truthy_log_value) & (data["pretrade_checklist_status"].astype(str) == "approved")
    return data.loc[mask].copy()


def outcome_price_frame(price_frames: dict[str, pd.DataFrame] | None, commodity: str) -> pd.DataFrame:
    if not price_frames:
        return pd.DataFrame()
    frame = price_frames.get(commodity, pd.DataFrame())
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    data = frame.copy()
    if "date" not in data.columns or "close" not in data.columns:
        return pd.DataFrame()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    for column in ["open", "high", "low", "close"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    if "high" not in data.columns:
        data["high"] = data["close"]
    if "low" not in data.columns:
        data["low"] = data["close"]
    return data.dropna(subset=["date", "close"]).sort_values("date")


def classify_post_trade_outcome(
    bars: int,
    stop_hit_date: str,
    partial_hit_date: str,
    time_stop_days: int | None,
    mfe_r: float,
    latest_r: float,
) -> str:
    if stop_hit_date:
        return "止损触发"
    if partial_hit_date:
        return "部分止盈触发"
    if time_stop_days and bars >= time_stop_days and (pd.isna(mfe_r) or mfe_r < 0.5 or latest_r < 0):
        return "时间止损触发"
    if not pd.isna(latest_r) and latest_r >= 1.0:
        return "顺势持有"
    return "跟踪中"


def build_post_trade_outcomes(log: pd.DataFrame, price_frames: dict[str, pd.DataFrame] | None) -> pd.DataFrame:
    signed = signed_action_rows(log)
    if signed.empty:
        return pd.DataFrame()
    notes = load_post_trade_notes()
    rows: list[dict[str, Any]] = []
    for _, raw in signed.iterrows():
        row = raw.to_dict()
        commodity = str(row.get("commodity", ""))
        signal_id = post_trade_signal_id(row)
        price_data = outcome_price_frame(price_frames, commodity)
        signal_date = parse_position_date(row.get("signal_date", ""))
        entry = numeric_value(row.get("entry", np.nan))
        stop = numeric_value(row.get("pretrade_invalidation_level", row.get("stop", np.nan)))
        partial = numeric_value(row.get("management_partial_exit", np.nan))
        time_stop_days_numeric = numeric_value(row.get("management_time_stop_days", np.nan))
        time_stop_days = None if pd.isna(time_stop_days_numeric) else int(time_stop_days_numeric)
        risk_per_unit = entry - stop if not pd.isna(entry) and not pd.isna(stop) else np.nan
        path = pd.DataFrame()
        if signal_date is not None and not price_data.empty:
            path = price_data.loc[price_data["date"].dt.date >= signal_date].copy()
        bars = int(len(path))
        latest_close = np.nan
        latest_date = ""
        max_high = np.nan
        min_low = np.nan
        stop_hit_date = ""
        partial_hit_date = ""
        if not path.empty:
            latest = path.iloc[-1]
            latest_close = numeric_value(latest.get("close", np.nan))
            latest_date = latest.get("date").date().isoformat() if pd.notna(latest.get("date")) else ""
            max_high = numeric_value(path["high"].max())
            min_low = numeric_value(path["low"].min())
            if not pd.isna(stop):
                stop_hits = path.loc[path["low"] <= stop]
                if not stop_hits.empty:
                    stop_hit_date = stop_hits.iloc[0]["date"].date().isoformat()
            if not pd.isna(partial):
                partial_hits = path.loc[path["high"] >= partial]
                if not partial_hits.empty:
                    partial_hit_date = partial_hits.iloc[0]["date"].date().isoformat()
        mfe = max_high - entry if not pd.isna(max_high) and not pd.isna(entry) else np.nan
        mae = min_low - entry if not pd.isna(min_low) and not pd.isna(entry) else np.nan
        latest_pnl = latest_close - entry if not pd.isna(latest_close) and not pd.isna(entry) else np.nan
        mfe_r = mfe / risk_per_unit if not pd.isna(mfe) and not pd.isna(risk_per_unit) and risk_per_unit > 0 else np.nan
        mae_r = mae / risk_per_unit if not pd.isna(mae) and not pd.isna(risk_per_unit) and risk_per_unit > 0 else np.nan
        latest_r = latest_pnl / risk_per_unit if not pd.isna(latest_pnl) and not pd.isna(risk_per_unit) and risk_per_unit > 0 else np.nan
        note = note_for_signal(notes, signal_id)
        rows.append(
            {
                "signal_id": signal_id,
                "信号日": row.get("signal_date", ""),
                "品种": commodity,
                "行动": row.get("action", ""),
                "操作员": row.get("pretrade_operator", ""),
                "计划手数": row.get("pretrade_planned_lots", row.get("trial_lots", "")),
                "入场参考": log_scalar(entry),
                "失效价": log_scalar(stop),
                "部分止盈": log_scalar(partial),
                "跟踪K数": bars,
                "最新日期": latest_date,
                "最新收盘": log_scalar(latest_close),
                "MFE": log_scalar(mfe),
                "MAE": log_scalar(mae),
                "MFE_R": "" if pd.isna(mfe_r) else round(float(mfe_r), 2),
                "MAE_R": "" if pd.isna(mae_r) else round(float(mae_r), 2),
                "最新R": "" if pd.isna(latest_r) else round(float(latest_r), 2),
                "止损日": stop_hit_date,
                "部分止盈日": partial_hit_date,
                "结果": classify_post_trade_outcome(bars, stop_hit_date, partial_hit_date, time_stop_days, mfe_r, latest_r),
                "实际成交": note.get("execution_status", ""),
                "实际均价": note.get("realized_avg_price", ""),
                "实际手数": note.get("realized_lots", ""),
                "复盘备注": note.get("realized_notes", ""),
                "更新时间": note.get("updated_at", ""),
            }
        )
    return pd.DataFrame(rows)


def signed_log_with_outcomes(log: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    signed = signed_action_rows(log)
    if signed.empty or outcomes.empty:
        return pd.DataFrame()
    signed = signed.copy()
    signed["signal_id"] = signed.apply(lambda row: post_trade_signal_id(row.to_dict()), axis=1)
    merged = signed.merge(outcomes, on="signal_id", how="left", suffixes=("_log", ""))
    for column in ["MFE_R", "MAE_R", "最新R", "top_weather_score", "top_weather_entry_score", "combined_score", "weather_score", "price_score"]:
        if column in merged.columns:
            merged[column] = pd.to_numeric(merged[column], errors="coerce")
    return merged


def summarize_calibration_group(data: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if data.empty or group_col not in data.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for value, subset in data.groupby(group_col, dropna=False):
        total = len(subset)
        if total <= 0:
            continue
        result = subset.get("结果", pd.Series([], dtype=str)).astype(str)
        mfe = pd.to_numeric(subset.get("MFE_R", pd.Series([], dtype=float)), errors="coerce")
        mae = pd.to_numeric(subset.get("MAE_R", pd.Series([], dtype=float)), errors="coerce")
        latest_r = pd.to_numeric(subset.get("最新R", pd.Series([], dtype=float)), errors="coerce")
        rows.append(
            {
                "维度": group_col,
                "分组": "n/a" if pd.isna(value) or value == "" else value,
                "样本数": total,
                "正向命中率": float((mfe >= 1.0).mean()) if len(mfe) else np.nan,
                "部分止盈率": float(result.str.contains("部分止盈", na=False).mean()) if len(result) else np.nan,
                "止损率": float(result.str.contains("止损触发", na=False).mean()) if len(result) else np.nan,
                "时间止损率": float(result.str.contains("时间止损", na=False).mean()) if len(result) else np.nan,
                "平均MFE_R": float(mfe.mean(skipna=True)) if mfe.notna().any() else np.nan,
                "平均MAE_R": float(mae.mean(skipna=True)) if mae.notna().any() else np.nan,
                "平均最新R": float(latest_r.mean(skipna=True)) if latest_r.notna().any() else np.nan,
                "建议": calibration_suggestion(total, mfe, mae, result),
            }
        )
    return pd.DataFrame(rows).sort_values(["维度", "样本数"], ascending=[True, False]).reset_index(drop=True)


def calibration_suggestion(total: int, mfe: pd.Series, mae: pd.Series, result: pd.Series) -> str:
    if total < 3:
        return "样本不足，仅记录不调参。"
    hit_rate = float((mfe >= 1.0).mean()) if len(mfe) else np.nan
    stop_rate = float(result.astype(str).str.contains("止损触发", na=False).mean()) if len(result) else np.nan
    avg_mae = float(mae.mean(skipna=True)) if mae.notna().any() else np.nan
    if not pd.isna(hit_rate) and hit_rate >= 0.60 and (pd.isna(stop_rate) or stop_rate <= 0.30):
        return "保留当前阈值；该分组正向跟随较好。"
    if not pd.isna(stop_rate) and stop_rate >= 0.50:
        return "复核入场/止损阈值；该分组止损占比偏高。"
    if not pd.isna(avg_mae) and avg_mae <= -1.0:
        return "复核等待价格确认或缩小试仓；不利波动偏大。"
    return "继续积累样本，暂不主动调参。"


def build_model_calibration_diagnostics(log: pd.DataFrame, outcomes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = signed_log_with_outcomes(log, outcomes)
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame()
    groups = [
        "commodity",
        "top_weather_driver",
        "playbook_stage",
        "health_label",
        "portfolio_decision",
    ]
    summaries = [summarize_calibration_group(merged, group) for group in groups]
    summary = pd.concat([item for item in summaries if not item.empty], ignore_index=True) if summaries else pd.DataFrame()
    detail_cols = [
        "signal_date",
        "commodity",
        "top_weather_driver",
        "playbook_stage",
        "health_label",
        "portfolio_decision",
        "combined_score",
        "weather_score",
        "price_score",
        "结果",
        "MFE_R",
        "MAE_R",
        "最新R",
        "pretrade_operator",
    ]
    details = merged[[col for col in detail_cols if col in merged.columns]].copy()
    return summary, details


def candidate_threshold_values(current: float, low: int = 40, high: int = 90) -> list[int]:
    values = {int(round(current))}
    for delta in (-10, -5, 5, 10):
        values.add(int(clamp(current + delta, low, high)))
    return sorted(values)


def threshold_review_suggestion(sample_count: int, hit_rate: float, stop_rate: float, avg_mfe: float, current_selected: bool) -> str:
    if sample_count < 3:
        return "样本不足，不建议调参。"
    if current_selected:
        return "当前阈值基准；只作为对照，不自动修改。"
    if not pd.isna(hit_rate) and hit_rate >= 0.60 and (pd.isna(stop_rate) or stop_rate <= 0.30) and (pd.isna(avg_mfe) or avg_mfe >= 1.0):
        return "可进入人工审阅；候选阈值表现较稳。"
    if not pd.isna(stop_rate) and stop_rate >= 0.45:
        return "不建议放宽；止损率偏高。"
    return "继续观察；证据不足以替换当前阈值。"


def build_threshold_review_workbench(log: pd.DataFrame, outcomes: pd.DataFrame, settings: RuleSettings) -> pd.DataFrame:
    _, details = build_model_calibration_diagnostics(log, outcomes)
    if details.empty:
        return pd.DataFrame()
    data = details.copy()
    for column in ["combined_score", "weather_score", "price_score", "MFE_R", "MAE_R", "最新R"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    if not {"combined_score", "weather_score", "price_score"}.issubset(data.columns):
        return pd.DataFrame()
    result = data.get("结果", pd.Series([""] * len(data))).astype(str)
    data["hit_1r"] = pd.to_numeric(data.get("MFE_R", pd.Series([], dtype=float)), errors="coerce") >= 1.0
    data["stop_hit"] = result.str.contains("止损触发", na=False)
    rows: list[dict[str, Any]] = []
    for weather_threshold in candidate_threshold_values(settings.weather_trigger, 40, 85):
        for price_threshold in candidate_threshold_values(settings.price_trigger, 40, 85):
            for build_threshold in candidate_threshold_values(settings.build_trigger, 50, 90):
                selected = data.loc[
                    (data["weather_score"] >= weather_threshold)
                    & (data["price_score"] >= price_threshold)
                    & (data["combined_score"] >= build_threshold)
                ].copy()
                sample_count = int(len(selected))
                hit_rate = float(selected["hit_1r"].mean()) if sample_count else np.nan
                stop_rate = float(selected["stop_hit"].mean()) if sample_count else np.nan
                avg_mfe = float(selected["MFE_R"].mean(skipna=True)) if sample_count and selected["MFE_R"].notna().any() else np.nan
                avg_mae = float(selected["MAE_R"].mean(skipna=True)) if sample_count and selected["MAE_R"].notna().any() else np.nan
                current_selected = (
                    weather_threshold == int(settings.weather_trigger)
                    and price_threshold == int(settings.price_trigger)
                    and build_threshold == int(settings.build_trigger)
                )
                strictness_delta = (
                    weather_threshold - int(settings.weather_trigger)
                    + price_threshold - int(settings.price_trigger)
                    + build_threshold - int(settings.build_trigger)
                )
                rows.append(
                    {
                        "天气门槛": weather_threshold,
                        "价格门槛": price_threshold,
                        "综合门槛": build_threshold,
                        "相对当前": "当前" if current_selected else ("更严格" if strictness_delta > 0 else "更宽松" if strictness_delta < 0 else "换权重"),
                        "样本数": sample_count,
                        "正向命中率": hit_rate,
                        "止损率": stop_rate,
                        "平均MFE_R": avg_mfe,
                        "平均MAE_R": avg_mae,
                        "建议": threshold_review_suggestion(sample_count, hit_rate, stop_rate, avg_mfe, current_selected),
                    }
                )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["排序命中"] = table["正向命中率"].fillna(-1)
    table["排序止损"] = table["止损率"].fillna(1)
    table["排序样本"] = table["样本数"].fillna(0)
    table = table.sort_values(["排序命中", "排序止损", "排序样本"], ascending=[False, True, False]).drop(columns=["排序命中", "排序止损", "排序样本"])
    current = table.loc[table["相对当前"] == "当前"]
    non_current = table.loc[table["相对当前"] != "当前"].head(12)
    return pd.concat([current, non_current], ignore_index=True)


def render_threshold_review_workbench(log: pd.DataFrame, outcomes: pd.DataFrame, settings: RuleSettings) -> None:
    st.write("**阈值审阅 Workbench**")
    st.caption("候选阈值只用于人工审阅，不会自动修改侧边栏 live 规则。")
    review = build_threshold_review_workbench(log, outcomes, settings)
    if review.empty:
        st.info("签核 outcome 样本不足，暂不能比较天气/价格/综合门槛。")
        return
    current = review.loc[review["相对当前"] == "当前"]
    if not current.empty:
        row = current.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("当前门槛", f"{int(row['天气门槛'])}/{int(row['价格门槛'])}/{int(row['综合门槛'])}")
        c2.metric("当前样本", int(row["样本数"]))
        c3.metric("当前命中率", "n/a" if pd.isna(row["正向命中率"]) else f"{row['正向命中率']:.1%}")
        c4.metric("当前止损率", "n/a" if pd.isna(row["止损率"]) else f"{row['止损率']:.1%}")
    st.dataframe(
        review.style.format(
            {
                "正向命中率": "{:.1%}",
                "止损率": "{:.1%}",
                "平均MFE_R": "{:.2f}",
                "平均MAE_R": "{:.2f}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        width="stretch",
    )
    st.download_button(
        "下载阈值审阅 CSV",
        review.to_csv(index=False).encode("utf-8-sig"),
        file_name="threshold_review_workbench.csv",
        mime="text/csv",
    )

def render_model_calibration_diagnostics(log: pd.DataFrame, outcomes: pd.DataFrame, settings: RuleSettings | None = None) -> None:
    st.write("**模型校准诊断**")
    summary, details = build_model_calibration_diagnostics(log, outcomes)
    if summary.empty:
        st.info("签核 outcome 样本不足；至少写入并跟踪若干可行动交易日后再评估阈值。")
        if settings is not None:
            render_threshold_review_workbench(log, outcomes, settings)
        return
    metric_cols = st.columns(4)
    sample_count = int(details.shape[0]) if not details.empty else 0
    hit_rate = pd.to_numeric(details.get("MFE_R", pd.Series([], dtype=float)), errors="coerce").ge(1.0).mean() if sample_count else np.nan
    stop_rate = details.get("结果", pd.Series([], dtype=str)).astype(str).str.contains("止损触发", na=False).mean() if sample_count else np.nan
    avg_mfe = pd.to_numeric(details.get("MFE_R", pd.Series([], dtype=float)), errors="coerce").mean(skipna=True) if sample_count else np.nan
    avg_mae = pd.to_numeric(details.get("MAE_R", pd.Series([], dtype=float)), errors="coerce").mean(skipna=True) if sample_count else np.nan
    metric_cols[0].metric("签核样本", sample_count)
    metric_cols[1].metric("正向命中率", "n/a" if pd.isna(hit_rate) else f"{hit_rate:.1%}")
    metric_cols[2].metric("平均MFE_R", "n/a" if pd.isna(avg_mfe) else f"{avg_mfe:.2f}R")
    metric_cols[3].metric("平均MAE_R", "n/a" if pd.isna(avg_mae) else f"{avg_mae:.2f}R", "止损率 n/a" if pd.isna(stop_rate) else f"止损率 {stop_rate:.1%}")
    st.dataframe(
        summary.style.format(
            {
                "正向命中率": "{:.1%}",
                "部分止盈率": "{:.1%}",
                "止损率": "{:.1%}",
                "时间止损率": "{:.1%}",
                "平均MFE_R": "{:.2f}",
                "平均MAE_R": "{:.2f}",
                "平均最新R": "{:.2f}",
            },
            na_rep="n/a",
        ),
        hide_index=True,
        width="stretch",
    )
    with st.expander("校准明细样本", expanded=False):
        st.dataframe(details, hide_index=True, width="stretch")
    st.download_button(
        "下载模型校准诊断 CSV",
        summary.to_csv(index=False).encode("utf-8-sig"),
        file_name="model_calibration_diagnostics.csv",
        mime="text/csv",
    )
    if settings is not None:
        st.divider()
        render_threshold_review_workbench(log, outcomes, settings)

def render_post_trade_outcome_tracker(log: pd.DataFrame, price_frames: dict[str, pd.DataFrame] | None, settings: RuleSettings | None = None) -> None:
    st.write("**Post-trade Outcome Tracker**")
    outcomes = build_post_trade_outcomes(log, price_frames)
    if outcomes.empty:
        st.info("暂无已签核的可行动交易日；完成盘前确认并写入日志后，这里会跟踪 MFE/MAE、止损、部分止盈和时间止损结果。")
        render_model_calibration_diagnostics(log, outcomes, settings)
        return
    st.dataframe(outcomes, hide_index=True, width="stretch")
    render_model_calibration_diagnostics(log, outcomes, settings)
    st.download_button(
        "下载 outcome tracker CSV",
        outcomes.to_csv(index=False).encode("utf-8-sig"),
        file_name="post_trade_outcomes.csv",
        mime="text/csv",
    )
    options = [f"{row['信号日']} | {row['品种']} | {row['行动']} | {row['signal_id']}" for _, row in outcomes.iterrows()]
    selected_option = st.selectbox("选择复盘记录", options=options, key="post-trade-outcome-select")
    selected_id = selected_option.split(" | ")[-1]
    notes = load_post_trade_notes()
    current_note = note_for_signal(notes, selected_id)
    with st.form(f"post-trade-note-{selected_id}"):
        execution_status = st.selectbox(
            "实际执行",
            ["未填写", "未成交", "已试仓", "已加仓", "已减仓", "已退出"],
            index=["未填写", "未成交", "已试仓", "已加仓", "已减仓", "已退出"].index(current_note.get("execution_status", "未填写")) if current_note.get("execution_status", "未填写") in ["未填写", "未成交", "已试仓", "已加仓", "已减仓", "已退出"] else 0,
        )
        avg_value = numeric_value(current_note.get("realized_avg_price", np.nan))
        lots_value = numeric_value(current_note.get("realized_lots", 0))
        realized_avg_price = st.number_input("实际成交均价", min_value=0.0, value=0.0 if pd.isna(avg_value) else float(avg_value), step=1.0)
        realized_lots = st.number_input("实际成交手数", min_value=0, value=0 if pd.isna(lots_value) else int(lots_value), step=1)
        realized_notes = st.text_area("复盘备注", value=str(current_note.get("realized_notes", "")), height=80)
        saved = st.form_submit_button("保存 outcome 备注")
    if saved:
        count = upsert_post_trade_note(
            {
                "signal_id": selected_id,
                "execution_status": execution_status,
                "realized_avg_price": realized_avg_price if realized_avg_price > 0 else "",
                "realized_lots": realized_lots,
                "realized_notes": realized_notes,
                "updated_at": dt.datetime.now(APP_TZ).isoformat(timespec="seconds"),
            }
        )
        st.success(f"已保存 outcome 备注到 {POST_TRADE_NOTES_PATH}，当前共 {count} 条。")

def render_signal_log_panel(row: dict[str, Any], settings: RuleSettings, price_frames: dict[str, pd.DataFrame] | None = None) -> None:
    st.subheader("每日运行日志 / 导出")
    st.caption("点击记录才会写入本地 CSV；Streamlit 自动刷新不会重复写日志。")
    row, can_log = render_pre_trade_checklist(row)
    current = pd.DataFrame([row])
    log = load_signal_log()
    display_cols = [
        "run_timestamp",
        "signal_date",
        "commodity",
        "pretrade_checklist_status",
        "pretrade_manual_approved",
        "pretrade_operator",
        "pretrade_planned_lots",
        "pretrade_invalidation_level",
        "action",
        "playbook_stage",
        "playbook_next_trigger",
        "management_stage",
        "management_trailing_stop",
        "management_partial_exit",
        "position_entry_date",
        "position_avg_entry",
        "position_lots",
        "position_holding_days",
        "combined_score",
        "weather_score",
        "price_score",
        "health_label",
        "contract_label",
        "contract_symbol",
        "contract_selector",
        "regime_label",
        "latest_close",
        "entry",
        "stop",
        "trial_lots",
        "risk_usage_pct",
        "margin_usage_pct",
    ]
    st.write("**当前快照**")
    st.dataframe(current[[col for col in display_cols if col in current.columns]], hide_index=True, width="stretch")
    render_change_summary(row, log, settings)
    st.download_button(
        "下载当前快照 CSV",
        current.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"signal_snapshot_{row['signal_date']}_{row['commodity'].replace(' ', '_')}.csv",
        mime="text/csv",
    )

    if st.button("记录当前快照到运行日志", key=f"write-log-{row['signal_date']}-{row['commodity']}", disabled=not can_log):
        count = append_signal_log(row)
        st.success(f"已写入 {SIGNAL_LOG_PATH}，当前共 {count} 条记录。")
        log = load_signal_log()
    if not can_log:
        st.warning("可行动交易日尚未完成 checklist 或人工确认，暂不允许写入行动日志。")
    if log.empty:
        st.info("暂无历史运行日志。")
        return
    st.write("**最近 20 条记录**")
    st.caption(f"日志文件：{SIGNAL_LOG_PATH}；记录数：{len(log)}")
    st.dataframe(log.tail(20), hide_index=True, width="stretch")
    st.download_button(
        "导出完整运行日志 CSV",
        log.to_csv(index=False).encode("utf-8-sig"),
        file_name="signal_run_log.csv",
        mime="text/csv",
    )
    st.divider()
    render_post_trade_outcome_tracker(log, price_frames, settings)

def render_entry_plan(
    selected: str,
    commodity: dict[str, Any],
    weather_score: float,
    price_signal: dict[str, Any],
    settings: RuleSettings,
    risk_config: dict[str, float],
    signal_health: dict[str, Any] | None = None,
    entry_playbook: dict[str, Any] | None = None,
    post_entry_playbook: dict[str, Any] | None = None,
    portfolio_gate: dict[str, Any] | None = None,
    market_session: dict[str, Any] | None = None,
) -> None:
    st.subheader("建仓执行框架")
    latest = price_signal.get("latest")
    price_score = int(price_signal.get("score", 0))
    weather_ready = not pd.isna(weather_score) and weather_score >= settings.weather_trigger
    price_ready = price_score >= settings.price_trigger
    signal_health = signal_health or {}
    portfolio_gate = portfolio_gate or {}
    market_session = market_session or {}
    trade_gate = signal_health.get("gate", "pass")

    if latest is None:
        st.info("行情数据不足，先等待 iFinD/AKShare 返回至少 60 个交易日的数据。")
        return

    combined = (
        (0 if pd.isna(weather_score) else weather_score) * float(commodity["weather_weight"])
        + price_score * float(commodity["price_weight"])
    )
    if market_session:
        st.write("**交易时段闸门**")
        s1, s2, s3 = st.columns(3)
        s1.metric("时段状态", market_session.get("label", "n/a"), market_session.get("status", ""))
        s2.metric("行动允许", "是" if market_session.get("action_allowed") else "否")
        s3.metric("收盘确认", "是" if market_session.get("close_confirmed") else "否", log_scalar(market_session.get("latest_price_date", "")))
        if market_session.get("action_allowed"):
            st.success(market_session.get("reason", "交易时段已放行。"))
        else:
            st.warning(market_session.get("reason", "交易时段尚未放行。"))

    if trade_gate == "block":
        st.error("\u5408\u7ea6/\u6570\u636e\u95f8\u95e8\u672a\u901a\u8fc7\uff1a\u6682\u4e0d\u751f\u6210\u5efa\u4ed3\u6267\u884c\u63d0\u793a\uff0c\u4ec5\u4fdd\u7559\u4ef7\u683c\u4f4d\u4f9b\u590d\u6838\u3002")
    elif trade_gate == "caution":
        st.warning("\u5408\u7ea6/\u6570\u636e\u68c0\u67e5\u4e3a\u9700\u590d\u6838\uff1a\u5373\u4f7f\u4ef7\u683c\u4e0e\u5929\u6c14\u8fbe\u6807\uff0c\u4e5f\u5148\u786e\u8ba4\u4e3b\u529b\u5408\u7ea6\u3001\u6210\u4ea4/\u6301\u4ed3\u548c\u79fb\u4ed3\u98ce\u9669\u3002")

    if trade_gate == "block":
        pass
    elif weather_ready and price_ready and combined >= settings.build_trigger:
        st.success("建仓条件达标：先按计划仓位 20%-30% 试仓，突破延续或回踩不破再加仓。")
    elif weather_ready and not price_ready:
        st.warning("天气主题已抬升，但价格确认不足；等突破位或 20 日均线回踩企稳。")
    elif price_ready and not weather_ready:
        st.info("价格结构偏强，但天气主题不足；按普通趋势观察，不把它归因于天气行情。")
    else:
        st.info("天气和价格尚未共振，维持观察。")

    if portfolio_gate:
        st.write("**组合风险闸门**")
        g1, g2, g3 = st.columns(3)
        g1.metric("组合决策", portfolio_gate.get("decision", "n/a"), portfolio_gate.get("status", "n/a"))
        g2.metric("申请新增", f"{portfolio_gate.get('requested_new_lots', 0)} 手")
        g3.metric("批准新增", f"{portfolio_gate.get('approved_new_lots', 0)} 手")
        if portfolio_gate.get("status") == "block":
            st.error(portfolio_gate.get("reason", "组合闸门禁止新增仓位。"))
        elif portfolio_gate.get("status") == "caution":
            st.warning(portfolio_gate.get("reason", "组合闸门要求削减新增仓位。"))
        else:
            st.info(portfolio_gate.get("reason", "组合闸门未限制当前品种。"))
    unit = commodity.get("unit", "")
    entry = price_signal.get("entry")
    pullback_low = price_signal.get("pullback_low")
    pullback_high = price_signal.get("pullback_high")
    stop = price_signal.get("stop")
    add_level = price_signal.get("add_level")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("突破试仓价", number_text(entry, 0), unit)
    if pd.isna(pullback_low) or pd.isna(pullback_high):
        pullback_text = "n/a"
    else:
        pullback_text = f"{number_text(pullback_low, 0)} - {number_text(pullback_high, 0)}"
    c2.metric("回踩观察区", pullback_text, "20日线附近")
    c3.metric("防守止损", number_text(stop, 0), unit)
    c4.metric("加仓确认价", number_text(add_level, 0), unit)

    latest_close = float(latest["close"]) if not pd.isna(latest["close"]) else np.nan
    st.write(
        f"当前收盘价 {number_text(latest_close, 0)} {unit}。"
        "若触发突破试仓，单笔风险应以防守止损倒推仓位；"
        "若先回踩观察区，则等待日线重新收强再执行。"
    )
    if entry_playbook:
        st.write("**情景化入场 Playbook**")
        p1, p2, p3 = st.columns(3)
        p1.metric("当前阶段", entry_playbook.get("current_stage", "n/a"))
        p2.metric("天气就绪", "是" if entry_playbook.get("weather_ready") else "否", f"门槛 {settings.weather_trigger}")
        p3.metric("价格就绪", "是" if entry_playbook.get("price_ready") else "否", f"门槛 {settings.price_trigger}")
        st.write(f"下一触发：{entry_playbook.get('next_trigger', 'n/a')}")
        st.write(f"失效条件：{entry_playbook.get('invalidate', 'n/a')}")
        playbook_rows = entry_playbook.get("rows", [])
        if playbook_rows:
            st.dataframe(pd.DataFrame(playbook_rows), hide_index=True, width="stretch")
    if post_entry_playbook:
        st.write("**试仓后管理 Playbook**")
        r_multiple = post_entry_playbook.get("r_multiple")
        r_text = "n/a" if pd.isna(r_multiple) else f"{float(r_multiple):.2f}R"
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("管理状态", post_entry_playbook.get("current_management", "n/a"))
        m2.metric("当前浮盈R", r_text)
        m3.metric("移动止损", number_text(post_entry_playbook.get("trailing_stop"), 0), unit)
        m4.metric("部分止盈参考", number_text(post_entry_playbook.get("partial_exit_level"), 0), unit)
        if post_entry_playbook.get("actual_position"):
            st.write(
                f"实际持仓：{post_entry_playbook.get('position_lots', 0)} 手，"
                f"均价 {number_text(post_entry_playbook.get('position_avg_entry'), 0)} {unit}，"
                f"试仓日 {post_entry_playbook.get('position_entry_date', 'n/a')}，"
                f"持仓 {post_entry_playbook.get('holding_days', 'n/a')} 个交易日。"
            )
        else:
            st.info("未记录实际试仓；以下管理规则按预案显示，时间止损不会自动触发。")
        st.write(f"时间止损：试仓后 {post_entry_playbook.get('time_stop_days', 'n/a')} 个交易日未兑现至少 +0.5R，则减半或退出试仓。")
        management_rows = post_entry_playbook.get("rows", [])
        if management_rows:
            st.write("管理项摘要：")
            for row in management_rows:
                st.write(f"- {row['管理项']}：{row['状态']}；{row['触发条件']}")
            st.dataframe(pd.DataFrame(management_rows), hide_index=True, width="stretch")
    position_plan = build_position_plan(
        price_signal,
        commodity,
        risk_config["account_size"],
        risk_config["risk_pct"],
        risk_config["max_margin_pct"],
        risk_config["trial_fraction"],
        risk_config["margin_rate"],
    )
    render_position_sizing(position_plan, unit)

def render_data_source_status(weather: pd.DataFrame, price_frames: dict[str, pd.DataFrame], enso_frame: pd.DataFrame) -> None:
    rows: list[dict[str, Any]] = []
    rows.append(
        {
            "数据": "天气",
            "状态": weather.attrs.get("data_source_status", "live"),
            "缓存时间": weather.attrs.get("cache_saved_at", ""),
            "陈旧天数": weather.attrs.get("cache_stale_days", ""),
            "说明": weather.attrs.get("data_source_message", "实时数据"),
        }
    )
    rows.append(
        {
            "数据": "ENSO/ONI",
            "状态": enso_frame.attrs.get("data_source_status", "live") if isinstance(enso_frame, pd.DataFrame) else "",
            "缓存时间": enso_frame.attrs.get("cache_saved_at", "") if isinstance(enso_frame, pd.DataFrame) else "",
            "陈旧天数": enso_frame.attrs.get("cache_stale_days", "") if isinstance(enso_frame, pd.DataFrame) else "",
            "说明": enso_frame.attrs.get("data_source_message", "实时数据") if isinstance(enso_frame, pd.DataFrame) else "",
        }
    )
    for name, frame in price_frames.items():
        rows.append(
            {
                "数据": f"行情 {name}",
                "状态": frame.attrs.get("data_source_status", "live") if isinstance(frame, pd.DataFrame) else "",
                "缓存时间": frame.attrs.get("cache_saved_at", "") if isinstance(frame, pd.DataFrame) else "",
                "陈旧天数": frame.attrs.get("cache_stale_days", "") if isinstance(frame, pd.DataFrame) else "",
                "说明": frame.attrs.get("data_source_message", "实时数据") if isinstance(frame, pd.DataFrame) else "",
            }
        )
    stale_rows = [row for row in rows if row["状态"] in {"cache", "partial-cache"}]
    if stale_rows:
        st.warning("部分数据使用 last-good 缓存；信号可参考，但必须复核陈旧程度和原始接口错误。")
    else:
        st.info("天气、ENSO 和行情当前均使用实时取数结果。")
    with st.expander("数据源状态 / last-good 缓存", expanded=bool(stale_rows)):
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

def load_manual_csv(label: str, key: str) -> pd.DataFrame | None:
    uploaded = st.file_uploader(
        f"{label} 手动上传行情 CSV",
        type=["csv"],
        key=key,
        help="字段至少包含 date/open/high/low/close，或中文字段 日期/开盘价/最高价/最低价/收盘价。",
    )
    if uploaded is None:
        return None
    return pd.read_csv(uploaded)


def price_cache_key(commodity_name: str, price_source: str, symbol: str, lookback_months: int) -> str:
    return f"{commodity_name}_{price_source}_{symbol}_{lookback_months}m"


def save_price_cache(
    commodity_name: str,
    price_source: str,
    symbol: str,
    lookback_months: int,
    frame: pd.DataFrame,
) -> None:
    if frame is None or frame.empty:
        return
    metadata = {
        "commodity": commodity_name,
        "price_source": price_source,

        "symbol": symbol,
        "lookback_months": lookback_months,
        "price_symbol": frame.attrs.get("price_symbol", symbol),
        "contract_selector": frame.attrs.get("contract_selector", {}),
    }
    save_last_good_frame("price", price_cache_key(commodity_name, price_source, symbol, lookback_months), frame, metadata)
    if price_source in {PRICE_SOURCE_IFIND, PRICE_SOURCE_AKSHARE}:
        save_public_price_snapshot(commodity_name, symbol, frame)


def load_price_cache(
    commodity_name: str,
    price_source: str,
    symbol: str,
    lookback_months: int,
    original_error: Any,
) -> tuple[pd.DataFrame, str | None]:
    frame, meta = load_last_good_frame("price", price_cache_key(commodity_name, price_source, symbol, lookback_months))
    if frame.empty:
        return pd.DataFrame(), None
    frame = normalize_price_frame(frame)
    if frame.empty:
        return pd.DataFrame(), None
    price_symbol = str((meta or {}).get("price_symbol") or symbol)
    attach_price_metadata(frame, price_symbol)
    if meta and isinstance(meta.get("contract_selector"), dict):
        frame.attrs["contract_selector"] = meta.get("contract_selector", {})
    message = cached_data_message(f"{commodity_name} 行情", meta, original_error)
    mark_frame_cache_source(frame, "cache", message)
    return frame, message


def save_public_price_snapshot(commodity_name: str, symbol: str, frame: pd.DataFrame) -> None:
    if frame is None or frame.empty:
        return
    csv_path, meta_path = public_price_paths(commodity_name)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    metadata = {
        "commodity": commodity_name,
        "symbol": symbol,
        "price_symbol": frame.attrs.get("price_symbol", symbol),
        "contract_selector": frame.attrs.get("contract_selector", {}),
        "saved_at": dt.datetime.now(APP_TZ).isoformat(timespec="seconds"),
        "rows": int(len(frame)),
        "latest_date": log_scalar(latest_price_date_from_frame(frame)),
        "source": frame.attrs.get("data_source_status", "live"),
    }
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, default=str)


def load_public_price_snapshot(commodity_name: str, symbol: str) -> tuple[pd.DataFrame, str | None]:
    csv_path, meta_path = public_price_paths(commodity_name)
    if not os.path.exists(csv_path):
        return pd.DataFrame(), None
    try:
        frame = pd.read_csv(csv_path, keep_default_na=False)
        frame = normalize_price_frame(frame)
        if frame.empty:
            return pd.DataFrame(), None
        meta: dict[str, Any] = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
        price_symbol = str(meta.get("price_symbol") or symbol)
        attach_price_metadata(frame, price_symbol)
        if isinstance(meta.get("contract_selector"), dict):
            frame.attrs["contract_selector"] = meta.get("contract_selector", {})
        saved_at = parse_cache_saved_at(meta.get("saved_at"))
        stale_days = (today_china() - saved_at.date()).days if saved_at else None
        frame.attrs["public_snapshot"] = True
        frame.attrs["cache_saved_at"] = meta.get("saved_at", "")
        frame.attrs["cache_stale_days"] = stale_days
        latest_date = meta.get("latest_date") or log_scalar(latest_price_date_from_frame(frame))
        stale_text = "n/a" if stale_days is None else f"{stale_days}d"
        message = f"{commodity_name} uses public static price snapshot; latest bar {latest_date}; saved {meta.get('saved_at', 'unknown')}; stale {stale_text}."
        mark_frame_cache_source(frame, "static", message)
        return frame, message
    except Exception as exc:
        return pd.DataFrame(), f"{commodity_name} static price snapshot failed: {exc}"
def get_price_data(
    commodity_name: str,
    commodity: dict[str, Any],
    lookback_months: int,
    price_source: str,
    ifind_symbol: str | None = None,
    ifind_refresh_token: str | None = None,
    ifind_username: str | None = None,
    ifind_password: str | None = None,
) -> tuple[pd.DataFrame, str | None]:
    end = today_china()
    start = end - dt.timedelta(days=lookback_days_for_months(lookback_months))
    requested_symbol = ifind_symbol or commodity.get("ifind_symbol", commodity.get("symbol", ""))

    def fallback(error_message: str) -> tuple[pd.DataFrame, str | None]:
        frame, cache_message = load_price_cache(commodity_name, price_source, requested_symbol, lookback_months, error_message)
        if not frame.empty and cache_message:
            return frame, cache_message
        return pd.DataFrame(), error_message

    if price_source == PRICE_SOURCE_STATIC:
        frame, static_message = load_public_price_snapshot(commodity_name, requested_symbol)
        if not frame.empty:
            return frame, static_message
        return fallback(static_message or f"{commodity_name} static price snapshot is unavailable")

    if price_source == PRICE_SOURCE_AKSHARE:
        requested_symbol = commodity["symbol"]
        try:
            raw = fetch_price_history(
                requested_symbol,
                start.strftime("%Y%m%d"),
                end.strftime("%Y%m%d"),
            )
            frame = normalize_price_frame(raw)
            if frame.empty:
                return fallback(f"{commodity_name} AKShare 未返回有效行情")
            attach_price_metadata(frame, requested_symbol)
            mark_frame_cache_source(frame, "live")
            save_price_cache(commodity_name, price_source, requested_symbol, lookback_months, frame)
            return frame, None
        except Exception as exc:
            return fallback(f"{commodity_name} AKShare 取数失败: {exc}")
    if price_source == PRICE_SOURCE_IFIND:
        code = requested_symbol or commodity["ifind_symbol"]
        requested_symbol = code
        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")
        if ifind_refresh_token:
            try:
                access_token = fetch_ifind_access_token(ifind_refresh_token)
                def fetch_http_symbol(symbol: str) -> pd.DataFrame:
                    raw_symbol = fetch_price_history_ifind_http(
                        symbol,
                        start_str,
                        end_str,
                        access_token,
                    )
                    return normalize_price_frame(raw_symbol)
                frame = fetch_http_symbol(code)
                if frame.empty:
                    return fallback(f"{commodity_name} iFinD HTTP 未返回有效行情")
                frame = select_active_contract_frame(
                    commodity_name,
                    commodity,
                    frame,
                    code,
                    end,
                    fetch_http_symbol,
                )
                mark_frame_cache_source(frame, "live")
                save_price_cache(commodity_name, price_source, requested_symbol, lookback_months, frame)
                return frame, None
            except Exception as exc:
                if not (ifind_username and ifind_password):
                    return fallback(f"{commodity_name} iFinD HTTP 取数失败: {exc}")
                # HTTP token 取数失败后，继续尝试 SDK 登录回退。
        if ifind_username and ifind_password:
            try:
                def fetch_sdk_symbol(symbol: str) -> pd.DataFrame:
                    raw_symbol = fetch_price_history_ifind_sdk(
                        symbol,
                        start_str,
                        end_str,
                        ifind_username,
                        ifind_password,
                    )
                    return normalize_price_frame(raw_symbol)
                frame = fetch_sdk_symbol(code)
                if frame.empty:
                    return fallback(f"{commodity_name} iFinD SDK 未返回有效行情")
                frame = select_active_contract_frame(
                    commodity_name,
                    commodity,
                    frame,
                    code,
                    end,
                    fetch_sdk_symbol,
                )
                mark_frame_cache_source(frame, "live")
                save_price_cache(commodity_name, price_source, requested_symbol, lookback_months, frame)
                return frame, None
            except Exception as exc:
                return fallback(f"{commodity_name} iFinD SDK 取数失败: {exc}")
        return fallback(f"{commodity_name} iFinD 需要有效 refresh_token 或账号密码")
    return pd.DataFrame(), None

def signal_box(title: str, action: str, note: str, score: float) -> None:
    st.markdown(
        f"""
        <div class="signal-box">
            <div class="signal-title">{title}</div>
            <div class="signal-action">{action}</div>
            <div class="signal-note">{score:.1f} / 100<br>{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_price_chart(name: str, signal: dict[str, Any], unit: str) -> None:
    data = signal.get("data")
    if data is None or data.empty:
        st.info("暂无可画图的行情数据。")
        return

    view = data.tail(260)
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=view["date"],
            open=view["open"],
            high=view["high"],
            low=view["low"],
            close=view["close"],
            name="K线",
            increasing_line_color="#b91c1c",
            decreasing_line_color="#166534",
        )
    )
    fig.add_trace(go.Scatter(x=view["date"], y=view["ma20"], name="MA20", line=dict(color="#2563eb", width=1.4)))
    fig.add_trace(go.Scatter(x=view["date"], y=view["ma60"], name="MA60", line=dict(color="#7c3aed", width=1.4)))
    fig.add_trace(go.Scatter(x=view["date"], y=view["ma120"], name="MA120", line=dict(color="#6b7280", width=1.1)))

    for key, label, color in [
        ("entry", "突破触发", "#dc2626"),
        ("stop", "防守止损", "#111827"),
        ("add_level", "加仓确认", "#ea580c"),
    ]:
        value = signal.get(key)
        if value is not None and not pd.isna(value):
            fig.add_hline(
                y=value,
                line_dash="dot",
                line_color=color,
                annotation_text=f"{label}: {value:,.0f}",
                annotation_position="top right",
            )

    fig.update_layout(
        title=f"{name} 主力连续价格结构（{unit}）",
        height=520,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=52, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, width="stretch")


def render_weather_map(weather: pd.DataFrame, selected: str) -> None:
    subset = weather.loc[weather["commodity"] == selected].copy()
    if subset.empty:
        st.info("暂无天气数据。")
        return
    color_column = "entry_ready_score" if "entry_ready_score" in subset.columns else "stress_score"
    color_title = "入场天气" if color_column == "entry_ready_score" else "天气压力"
    fig = px.scatter_geo(
        subset,
        lat="lat",
        lon="lon",
        color=color_column,
        size="weight",
        hover_name="region",
        hover_data={
            "driver": True,
            "persistence_label": True,
            "impact_label": True,
            "impact_reason": True,
            "entry_ready_score": ":.1f",
            "impact_multiplier": ":.2f",
            "precip_ratio": ":.0%",
            "precip_percentile": ":.0%",
            "recent_precip_ratio": ":.0%",
            "raw_stress_score": ":.1f",
            "stress_score": ":.1f",
            "tmax_anom_c": ":.1f",
            "lat": False,
            "lon": False,
            "weight": ":.2f",
        },
        color_continuous_scale=["#1a9850", "#fee08b", "#d73027"],
        range_color=(0, 100),
        projection="natural earth",
    )
    fig.update_geos(
        showcountries=True,
        showcoastlines=True,
        showland=True,
        landcolor="#f8fafc",
        showocean=True,
        oceancolor="#eff6ff",
    )
    fig.update_layout(
        height=430,
        margin=dict(l=0, r=0, t=10, b=0),
        coloraxis_colorbar=dict(title=color_title),
    )
    st.plotly_chart(fig, width="stretch")




def filtered_errors(errors: list[str], commodity_name: str) -> list[str]:
    return [error for error in errors if error.startswith(commodity_name)]


def top_weather_driver(weather: pd.DataFrame, commodity_name: str) -> dict[str, Any]:
    fallback = {
        "region": "n/a",
        "driver": "n/a",
        "score": np.nan,
        "raw_score": np.nan,
        "entry_score": np.nan,
        "impact_timing": "n/a",
        "impact_label": "n/a",
        "impact_multiplier": np.nan,
        "impact_reason": "",
        "persistence": "n/a",
        "precip_percentile": np.nan,
        "recent_precip_ratio": np.nan,
    }
    if weather.empty or "commodity" not in weather.columns:
        return fallback
    subset = weather.loc[weather["commodity"] == commodity_name].copy()
    if subset.empty:
        return fallback
    sort_column = "entry_weighted_score" if "entry_weighted_score" in subset.columns else "weighted_score"
    row = subset.sort_values(sort_column, ascending=False).iloc[0]
    return {
        "region": row.get("region", "n/a"),
        "driver": row.get("driver", "n/a"),
        "score": numeric_value(row.get("stress_score", np.nan)),
        "raw_score": numeric_value(row.get("raw_stress_score", np.nan)),
        "entry_score": numeric_value(row.get("entry_ready_score", row.get("stress_score", np.nan))),
        "impact_timing": row.get("impact_timing", "n/a"),
        "impact_label": row.get("impact_label", "n/a"),
        "impact_multiplier": numeric_value(row.get("impact_multiplier", np.nan)),
        "impact_reason": row.get("impact_reason", ""),
        "persistence": row.get("persistence_label", "n/a"),
        "precip_percentile": numeric_value(row.get("precip_percentile", np.nan)),
        "recent_precip_ratio": numeric_value(row.get("recent_precip_ratio", np.nan)),
    }


def build_commodity_signal_snapshot(
    commodity_name: str,
    commodity: dict[str, Any],
    weather: pd.DataFrame,
    weather_errors: list[str],
    price_frame: pd.DataFrame,
    price_errors: list[str],
    price_symbol: str,
    enso_frame: pd.DataFrame,
    enso_error: str | None,
    settings: RuleSettings,
    risk_config: dict[str, float],
    anchor: dt.date,
    position_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    commodity_weather_errors = filtered_errors(weather_errors, commodity_name)
    commodity_price_errors = filtered_errors(price_errors, commodity_name)
    weather_score = weather_score_for(commodity_name, weather)
    weather_pressure_score = weather_pressure_score_for(commodity_name, weather)
    price_signal = score_price_signal(price_frame, commodity["tick"])
    market_session = classify_market_session(anchor, price_frame)
    contract_check = evaluate_contract_liquidity(
        commodity_name,
        commodity,
        price_frame,
        price_symbol,
        anchor,
    )
    signal_health = evaluate_signal_health(
        commodity_name,
        weather,
        price_frame,
        commodity_weather_errors,
        commodity_price_errors,
        anchor,
        contract_check,
    )
    regime_context = build_regime_context(commodity_name, enso_frame, enso_error, anchor)
    action, note, combined = classify_signal(weather_score, price_signal["score"], settings, commodity)
    action, note = apply_signal_health_gate(action, note, signal_health)
    action, note = apply_market_session_guard(action, note, market_session)
    position_plan = build_position_plan(
        price_signal,
        commodity,
        risk_config["account_size"],
        risk_config["risk_pct"],
        risk_config["max_margin_pct"],
        risk_config["trial_fraction"],
        risk_config["margin_rate"],
    )
    entry_playbook = build_entry_playbook(
        commodity_name,
        commodity,
        weather_score,
        combined,
        price_signal,
        settings,
        signal_health,
        position_plan,
    )
    post_entry_playbook = build_post_entry_management_playbook(
        commodity_name,
        commodity,
        weather_score,
        price_signal,
        settings,
        signal_health,
        position_plan,
        entry_playbook,
        position_state,
        anchor,
    )
    return {
        "commodity": commodity_name,
        "commodity_config": commodity,
        "weather_score": weather_score,
        "weather_pressure_score": weather_pressure_score,
        "price_signal": price_signal,
        "signal_health": signal_health,
        "contract_check": contract_check,
        "regime_context": regime_context,
        "market_session": market_session,
        "action": action,
        "note": note,
        "combined": combined,
        "position_plan": position_plan,
        "entry_playbook": entry_playbook,
        "post_entry_playbook": post_entry_playbook,
        "top_weather": top_weather_driver(weather, commodity_name),
    }


def build_all_commodity_snapshots(
    weather: pd.DataFrame,
    weather_errors: list[str],
    price_frames: dict[str, pd.DataFrame],
    price_errors: list[str],
    enso_frame: pd.DataFrame,
    enso_error: str | None,
    settings: RuleSettings,
    risk_config: dict[str, float],
    anchor: dt.date,
    price_symbols: dict[str, str] | None = None,
    position_states: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    price_symbols = price_symbols or {}
    position_states = position_states or {}
    snapshots = {
        name: build_commodity_signal_snapshot(
            name,
            commodity,
            weather,
            weather_errors,
            price_frames.get(name, pd.DataFrame()),
            price_errors,
            price_symbols.get(name, commodity.get("ifind_symbol", commodity.get("symbol", ""))),
            enso_frame,
            enso_error,
            settings,
            risk_config,
            anchor,
            position_state_for(position_states, name),
        )
        for name, commodity in COMMODITIES.items()
    }
    apply_portfolio_gate_to_snapshots(snapshots, risk_config)
    return snapshots

def format_score(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.1f}"


def format_pct_value(value: Any) -> str:
    numeric = numeric_value(value)
    if pd.isna(numeric):
        return "n/a"
    return f"{numeric:.2%}"


def render_portfolio_stacking_gate(portfolio: dict[str, Any]) -> None:
    if not portfolio:
        return
    correlation = portfolio.get("correlation", {}) or {}
    corr_value = numeric_value(correlation.get("correlation", np.nan))
    corr_text = "n/a" if pd.isna(corr_value) else f"{corr_value:.2f}"
    st.write("**组合风险/叠加闸门**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("组合闸门", portfolio.get("label", "n/a"), portfolio.get("status", "n/a"))
    c2.metric(
        "批准后止损风险",
        format_pct_value(portfolio.get("risk_if_approved_pct")),
        f"上限 {format_pct_value(portfolio.get('risk_cap_pct'))}",
    )
    c3.metric(
        "批准后保证金",
        format_pct_value(portfolio.get("margin_if_approved_pct")),
        f"上限 {format_pct_value(portfolio.get('margin_cap_pct'))}",
    )
    c4.metric(
        "SR/P 相关性",
        corr_text,
        f"阈值 {number_text(portfolio.get('correlation_trigger'), 2)}",
    )

    for blocker in portfolio.get("blockers", []):
        st.error(blocker)
    for warning in portfolio.get("warnings", []):
        st.warning(warning)
    if not portfolio.get("blockers") and not portfolio.get("warnings"):
        st.info("组合风险、保证金和相关性约束未阻断当前新增计划。")

    rows: list[dict[str, Any]] = []
    for name, gate in (portfolio.get("commodities", {}) or {}).items():
        rows.append(
            {
                "品种": name,
                "阶段": gate.get("stage", "n/a"),
                "组合决策": gate.get("decision", "n/a"),
                "申请新增": gate.get("requested_new_lots", 0),
                "批准新增": gate.get("approved_new_lots", 0),
                "已有手数": gate.get("actual_lots", 0),
                "当前风险": number_text(gate.get("current_risk"), 0),
                "申请新增风险": number_text(gate.get("requested_new_risk"), 0),
                "批准新增风险": number_text(gate.get("approved_new_risk"), 0),
                "原因": gate.get("reason", ""),
            }
        )
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

def stress_position_lots(snapshot: dict[str, Any]) -> int:
    post_entry = snapshot.get("post_entry_playbook", {}) or {}
    if post_entry.get("actual_position"):
        return portfolio_lot_count(post_entry.get("position_lots", 0))
    gate = snapshot.get("portfolio_gate", {}) or {}
    approved = portfolio_lot_count(gate.get("approved_new_lots", 0))
    if approved > 0:
        return approved
    plan = snapshot.get("position_plan", {}) or {}
    return portfolio_lot_count(plan.get("trial_lots", 0)) if plan.get("ok") else 0


def stress_entry_price(snapshot: dict[str, Any]) -> float:
    post_entry = snapshot.get("post_entry_playbook", {}) or {}
    actual_entry = numeric_value(post_entry.get("position_avg_entry", np.nan))
    if post_entry.get("actual_position") and not pd.isna(actual_entry):
        return actual_entry
    plan = snapshot.get("position_plan", {}) or {}
    planned_entry = numeric_value(plan.get("entry", np.nan))
    if not pd.isna(planned_entry):
        return planned_entry
    return numeric_value(snapshot.get("price_signal", {}).get("entry", np.nan))


def stress_latest_close(snapshot: dict[str, Any]) -> float:
    latest = snapshot.get("price_signal", {}).get("latest")
    if latest is not None and "close" in latest:
        value = numeric_value(latest["close"])
        if not pd.isna(value):
            return value
    plan_close = numeric_value(snapshot.get("position_plan", {}).get("latest_close", np.nan))
    if not pd.isna(plan_close):
        return plan_close
    return stress_entry_price(snapshot)


def stress_stop_price(snapshot: dict[str, Any]) -> float:
    post_entry = snapshot.get("post_entry_playbook", {}) or {}
    trailing = numeric_value(post_entry.get("trailing_stop", np.nan))
    if not pd.isna(trailing):
        return trailing
    plan_stop = numeric_value(snapshot.get("position_plan", {}).get("stop", np.nan))
    if not pd.isna(plan_stop):
        return plan_stop
    return numeric_value(snapshot.get("price_signal", {}).get("stop", np.nan))


def build_regime_stress_tests(
    snapshots: dict[str, dict[str, Any]],
    risk_config: dict[str, float],
    settings: RuleSettings,
) -> pd.DataFrame:
    account_size = numeric_value(risk_config.get("account_size", np.nan))
    if pd.isna(account_size) or account_size <= 0:
        account_size = np.nan
    margin_rate = numeric_value(risk_config.get("margin_rate", 0.12))
    if pd.isna(margin_rate) or margin_rate <= 0:
        margin_rate = 0.12
    max_margin_pct = numeric_value(risk_config.get("max_margin_pct", np.nan))
    rows: list[dict[str, Any]] = []
    for name, snapshot in snapshots.items():
        commodity = snapshot.get("commodity_config", COMMODITIES.get(name, {})) or {}
        multiplier = numeric_value(commodity.get("contract_multiplier", 10))
        if pd.isna(multiplier) or multiplier <= 0:
            multiplier = 10.0
        lots = stress_position_lots(snapshot)
        entry = stress_entry_price(snapshot)
        latest = stress_latest_close(snapshot)
        stop = stress_stop_price(snapshot)
        weather_score = numeric_value(snapshot.get("weather_score", np.nan))
        regime = snapshot.get("regime_context", {}) or {}
        top_weather = snapshot.get("top_weather", {}) or {}
        atr = np.nan
        latest_row = snapshot.get("price_signal", {}).get("latest")
        if latest_row is not None and "atr14" in latest_row:
            atr = numeric_value(latest_row["atr14"])
        if pd.isna(atr) or atr <= 0:
            atr = latest * 0.015 if not pd.isna(latest) else np.nan
        active = lots > 0 and not pd.isna(entry) and not pd.isna(latest)
        scenarios = [
            ("天气反转", latest, max(0.0, settings.weather_trigger * 0.65), "天气分跌破入场阈值 65%，天气交易逻辑失效。"),
            ("隔夜跳空-1ATR", latest - atr if not pd.isna(atr) else np.nan, weather_score, "隔夜低开约 1 ATR，检验试仓承受力。"),
            ("隔夜跳空-2ATR", latest - 2 * atr if not pd.isna(atr) else np.nan, weather_score, "隔夜低开约 2 ATR，需要优先降风险。"),
            ("跌破止损", stop, weather_score, "价格直接触及防守止损或移动止损。"),
            ("5%价格缺口", latest * 0.95 if not pd.isna(latest) else np.nan, weather_score, "极端流动性/外盘冲击导致 5% 不利缺口。"),
        ]
        for scenario, shocked_price, shocked_weather, description in scenarios:
            if not active or pd.isna(shocked_price):
                loss = np.nan
                loss_pct = np.nan
                stressed_margin = np.nan
                margin_pct = np.nan
                action = "无持仓/无计划手数"
            else:
                pnl = (shocked_price - entry) * multiplier * lots
                loss = min(0.0, pnl)
                loss_pct = abs(loss) / account_size if not pd.isna(account_size) and account_size > 0 else np.nan
                stressed_margin = max(shocked_price, 0) * multiplier * lots * margin_rate
                margin_pct = stressed_margin / account_size if not pd.isna(account_size) and account_size > 0 else np.nan
                stop_breached = not pd.isna(stop) and shocked_price <= stop
                weather_failed = not pd.isna(shocked_weather) and shocked_weather < settings.weather_trigger * 0.70
                margin_stressed = not pd.isna(margin_pct) and not pd.isna(max_margin_pct) and margin_pct > max_margin_pct
                if stop_breached:
                    action = "强制退出/止损"
                elif scenario == "天气反转" or weather_failed:
                    action = "取消加仓并减半试仓"
                elif loss_pct >= 0.015 or margin_stressed:
                    action = "降风险到半仓以下"
                elif loss_pct >= 0.0075:
                    action = "禁止加仓，盘中复核"
                else:
                    action = "维持但不加仓"
            rows.append(
                {
                    "品种": name,
                    "场景": scenario,
                    "Regime": regime.get("label", "n/a"),
                    "主天气": f"{top_weather.get('driver', 'n/a')} / {top_weather.get('region', 'n/a')}",
                    "持仓/计划手数": lots,
                    "入场/均价": log_scalar(entry),
                    "当前价": log_scalar(latest),
                    "冲击价": log_scalar(shocked_price),
                    "冲击天气分": "" if pd.isna(shocked_weather) else round(float(shocked_weather), 1),
                    "止损价": log_scalar(stop),
                    "估算损益": log_scalar(loss),
                    "账户损失": "" if pd.isna(loss_pct) else loss_pct,
                    "保证金占用": log_scalar(stressed_margin),
                    "保证金占账户": "" if pd.isna(margin_pct) else margin_pct,
                    "强制动作": action,
                    "说明": description,
                }
            )
    return pd.DataFrame(rows)


def render_regime_stress_test_panel(
    snapshots: dict[str, dict[str, Any]],
    risk_config: dict[str, float],
    settings: RuleSettings,
) -> None:
    st.subheader("Regime-aware 压力测试")
    st.caption("模拟天气信号反转和价格不利跳空；用于决定是否取消加仓、减半试仓或强制止损。")
    stress = build_regime_stress_tests(snapshots, risk_config, settings)
    if stress.empty:
        st.info("暂无可测试的品种快照。")
        return
    active = stress.loc[stress["持仓/计划手数"] > 0].copy()
    if active.empty:
        st.info("当前没有实际持仓或被组合闸门批准的计划手数；压力测试以 0 手显示，先等待可行动仓位。")
    severe = active.loc[active["强制动作"].isin(["强制退出/止损", "取消加仓并减半试仓", "降风险到半仓以下"])]
    if not severe.empty:
        st.error("压力测试存在强制动作：" + " / ".join(severe["品种"].astype(str).unique()))
    else:
        st.info("当前压力情景未触发强制退出；仍需按盘中价格和保证金实时复核。")
    stress_display = stress.copy()
    for column in ["\u8d26\u6237\u635f\u5931", "\u4fdd\u8bc1\u91d1\u5360\u8d26\u6237"]:
        if column in stress_display.columns:
            numeric = pd.to_numeric(stress_display[column], errors="coerce")
            stress_display[column] = numeric.map(lambda value: "n/a" if pd.isna(value) else f"{value:.2%}")
    st.dataframe(stress_display, hide_index=True, width="stretch")
    st.download_button(
        "下载压力测试 CSV",
        stress.to_csv(index=False).encode("utf-8-sig"),
        file_name="regime_stress_tests.csv",
        mime="text/csv",
    )

def build_morning_summary_table(snapshots: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, snapshot in snapshots.items():
        price_signal = snapshot["price_signal"]
        health = snapshot["signal_health"]
        regime = snapshot["regime_context"]
        plan = snapshot["position_plan"]
        top_weather = snapshot["top_weather"]
        entry_playbook = snapshot.get("entry_playbook", {})
        post_entry = snapshot.get("post_entry_playbook", {})
        portfolio_gate = snapshot.get("portfolio_gate", {})
        market_session = snapshot.get("market_session", {}) or {}
        contract = health.get("contract_check", {})
        plan_ok = bool(plan.get("ok")) and health.get("gate") != "block"
        latest = price_signal.get("latest")
        latest_close = np.nan
        latest_date = "n/a"
        if latest is not None:
            latest_close = float(latest["close"]) if "close" in latest and not pd.isna(latest["close"]) else np.nan
            latest_date = log_scalar(latest["date"]) if "date" in latest else "n/a"
        rows.append(
            {
                "品种": name,
                "行动": snapshot["action"],
                "执行阶段": entry_playbook.get("current_stage", "n/a"),
                "持仓管理": post_entry.get("current_management", "n/a"),
                "实际手数": post_entry.get("position_lots", 0),
                "试仓日期": post_entry.get("position_entry_date", ""),
                "持仓天数": "" if post_entry.get("holding_days") is None else post_entry.get("holding_days"),
                "组合闸门": portfolio_gate.get("decision", "n/a"),
                "申请新增": portfolio_gate.get("requested_new_lots", 0),
                "批准新增": portfolio_gate.get("approved_new_lots", 0),
                "组合原因": portfolio_gate.get("reason", ""),
                "交易时段": market_session.get("label", "n/a"),
                "行动允许": "是" if market_session.get("action_allowed") else "否",
                "时段原因": market_session.get("reason", ""),
                "下一触发": entry_playbook.get("next_trigger", ""),
                "综合分": format_score(snapshot["combined"]),
                "天气分": format_score(snapshot["weather_score"]),
                "原始天气分": format_score(snapshot.get("weather_pressure_score")),
                "价格确认": int(price_signal.get("score", 0)),
                "数据健康": f"{health['score']}/100 {health['label']}",
                "\u5408\u7ea6\u72b6\u6001": f"{contract.get('label', 'n/a')} {contract.get('symbol', 'n/a')}",
                "\u6d41\u52a8\u6027": contract.get("liquidity_summary", "n/a"),
                "\u8fde\u7eed/\u79fb\u4ed3": f"{contract.get('continuity_summary', 'n/a')} / {contract.get('roll_summary', 'n/a')}",
                "\u5019\u9009\u9009\u62e9": contract.get("selector_summary", ""),
                "Regime": regime.get("label", "n/a"),
                "ENSO": regime.get("enso", {}).get("phase", "n/a"),
                "最新行情日": latest_date,
                "最新收盘": number_text(latest_close, 0),
                "突破试仓": number_text(price_signal.get("entry"), 0),
                "防守止损": number_text(price_signal.get("stop"), 0),
                "移动止损": number_text(post_entry.get("trailing_stop"), 0),
                "部分止盈": number_text(post_entry.get("partial_exit_level"), 0),
                "试仓手数": plan.get("trial_lots", "n/a") if plan_ok else "n/a",
                "风险占账户": format_pct_value(plan.get("risk_usage_pct")) if plan_ok else "n/a",
                "保证金占账户": format_pct_value(plan.get("margin_usage_pct")) if plan_ok else "n/a",
                "主驱动区域": top_weather["region"],
                "天气驱动": f"{top_weather['driver']} {format_score(top_weather['score'])}",
                "影响时滞": top_weather["impact_label"],
                "入场天气": format_score(top_weather["entry_score"]),
                "持续性": top_weather["persistence"],
                "降雨分位": format_pct_value(top_weather["precip_percentile"]),
                "近期降雨比": format_pct_value(top_weather["recent_precip_ratio"]),
            }
        )
    return pd.DataFrame(rows)


def render_morning_summary(
    snapshots: dict[str, dict[str, Any]],
    settings: RuleSettings,
    anchor: dt.date,
) -> None:
    st.subheader("盘前双品种总览")
    st.caption(
        f"{anchor.isoformat()} 盘前扫描：天气门槛 {settings.weather_trigger}，"
        f"价格门槛 {settings.price_trigger}，综合建仓门槛 {settings.build_trigger}。"
    )
    summary = build_morning_summary_table(snapshots)
    if summary.empty:
        st.info("暂无可用的双品种摘要。")
        return
    render_portfolio_stacking_gate(portfolio_summary_from_snapshots(snapshots))
    cols = st.columns(len(snapshots))
    for col, (name, snapshot) in zip(cols, snapshots.items()):
        price_signal = snapshot["price_signal"]
        health = snapshot["signal_health"]
        regime = snapshot["regime_context"]
        plan = snapshot["position_plan"]
        contract = health.get("contract_check", {})
        market_session = snapshot.get("market_session", {}) or {}
        plan_ok = bool(plan.get("ok")) and health.get("gate") != "block"
        latest = price_signal.get("latest")
        latest_close = np.nan
        if latest is not None and "close" in latest and not pd.isna(latest["close"]):
            latest_close = float(latest["close"])
        with col:
            st.write(f"**{name}**")
            st.metric("行动", snapshot["action"], f"综合 {format_score(snapshot['combined'])}")
            st.metric(
                "天气 / 价格确认",
                f"{format_score(snapshot['weather_score'])} / {price_signal.get('score', 0)}",
                f"门槛 {settings.weather_trigger} / {settings.price_trigger}",
            )
            st.write(f"数据健康：{health['score']}/100 {health['label']}")
            st.write(f"\u5408\u7ea6\u68c0\u67e5\uff1a{contract.get('label', 'n/a')}\uff0c{contract.get('symbol', 'n/a')}\uff0c{contract.get('liquidity_summary', 'n/a')}\uff1b{contract.get('continuity_summary', 'n/a')}\u3002")
            if contract.get("selector_summary"):
                st.write(f"候选选择：{contract.get('selector_summary')}。")
            st.write(f"Regime：{regime.get('label', 'n/a')}；ENSO：{regime.get('enso', {}).get('phase', 'n/a')}")
            st.write(f"交易时段：{market_session.get('label', 'n/a')}；行动允许：{'是' if market_session.get('action_allowed') else '否'}；{market_session.get('reason', '')}")
            top_weather = snapshot["top_weather"]
            entry_playbook = snapshot.get("entry_playbook", {})
            post_entry = snapshot.get("post_entry_playbook", {})
            portfolio_gate = snapshot.get("portfolio_gate", {})
            st.write(f"执行阶段：{entry_playbook.get('current_stage', 'n/a')}；下一触发：{entry_playbook.get('next_trigger', 'n/a')}")
            st.write(f"组合闸门：{portfolio_gate.get('decision', 'n/a')}；申请 {portfolio_gate.get('requested_new_lots', 0)} 手，批准 {portfolio_gate.get('approved_new_lots', 0)} 手；{portfolio_gate.get('reason', '')}" )
            st.write(f"持仓管理：{post_entry.get('current_management', 'n/a')}；实际 {post_entry.get('position_lots', 0)} 手，持仓 {post_entry.get('holding_days') if post_entry.get('holding_days') is not None else 'n/a'} 天；移动止损 {number_text(post_entry.get('trailing_stop'), 0)}，部分止盈 {number_text(post_entry.get('partial_exit_level'), 0)}。")
            st.write(
                f"主天气：{top_weather['region']}，{top_weather['driver']}，"
                f"影响时滞 {top_weather['impact_label']}，"
                f"入场天气 {format_score(top_weather['entry_score'])} / 原始压力 {format_score(top_weather['score'])}，"
                f"持续性 {top_weather['persistence']}，"
                f"降雨分位 {format_pct_value(top_weather['precip_percentile'])}，"
                f"近期降雨比 {format_pct_value(top_weather['recent_precip_ratio'])}。"
            )
            st.write(
                f"价格确认：收盘 {number_text(latest_close, 0)}，"
                f"试仓 {number_text(price_signal.get('entry'), 0)}，"
                f"止损 {number_text(price_signal.get('stop'), 0)}。"
            )
            if plan_ok:
                st.write(
                    f"风险输出：试仓 {plan.get('trial_lots', 'n/a')} 手，"
                    f"风险占账户 {format_pct_value(plan.get('risk_usage_pct'))}，"
                    f"保证金占账户 {format_pct_value(plan.get('margin_usage_pct'))}。"
                )
            else:
                st.write(f"风险输出：{plan.get('reason', '仓位计算不可用。')}")
    st.dataframe(summary, hide_index=True, width="stretch")

    priority_notes: list[str] = []
    for name, snapshot in snapshots.items():
        action = snapshot["action"]
        if action in ("开始试仓", "待人工复核", "等待价格触发"):
            priority_notes.append(
                f"{name}：{action}；综合分 {format_score(snapshot['combined'])}，"
                f"天气分 {format_score(snapshot['weather_score'])}，"
                f"价格确认 {snapshot['price_signal'].get('score', 0)}。"
            )
    if priority_notes:
        st.warning(" / ".join(priority_notes))
    else:
        st.info("当前没有品种同时满足天气和价格建仓共振；优先跟踪天气已抬升但价格未确认的品种。")
def render_multi_commodity_price_snapshot(
    price_frames: dict[str, pd.DataFrame],
    lookback_months: int,
) -> None:
    st.subheader(f"近{lookback_months}个月白糖/棕榈油对照")

    summary_rows: list[dict[str, Any]] = []
    compare_parts: list[pd.DataFrame] = []

    for name, _ in COMMODITIES.items():
        frame = price_frames.get(name, pd.DataFrame())
        if frame.empty:
            summary_rows.append(
                {
                    "Commodity": name,
                    "Latest": "n/a",
                    "MA20": "n/a",
                    "ATR14": "n/a",
                    f"{lookback_months}m Change": "n/a",
                }
            )
            continue

        scored = add_price_indicators(frame)
        latest = scored.iloc[-1]
        start_close = scored["close"].iloc[0]
        latest_close = float(latest["close"])
        ma20 = latest["ma20"]
        atr14 = latest["atr14"]
        if (not pd.isna(start_close) and start_close > 0 and not pd.isna(latest_close)):
            month_change = (latest_close / float(start_close) - 1) * 100
        else:
            month_change = np.nan

        series = scored[["date", "close"]].copy()
        base = float(series["close"].iloc[0])
        if base > 0 and not pd.isna(base):
            series["normalized_index"] = series["close"] / base * 100
            series["Commodity"] = name
            compare_parts.append(series[["date", "normalized_index", "Commodity"]])

        summary_rows.append(
            {
                "Commodity": name,
                "Latest": number_text(latest_close, 0),
                "MA20": number_text(ma20, 0),
                "ATR14": number_text(atr14, 0),
                f"{lookback_months}m Change": f"{number_text(month_change, 2)}%" if not pd.isna(month_change) else "n/a",
            }
        )

    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, width="stretch")

    if compare_parts:
        compare = pd.concat(compare_parts, ignore_index=True)
        fig = px.line(
            compare,
            x="date",
            y="normalized_index",
            color="Commodity",
            title="归一化指数（起点=100）",
        )
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="日期", yaxis_title="指数")
        st.plotly_chart(fig, width="stretch")

@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def fetch_case_replay_prices(
    ifind_symbol: str,
    start_date: str,
    end_date: str,
    ifind_refresh_token: str,
    ifind_username: str,
    ifind_password: str,
) -> tuple[pd.DataFrame, str | None]:
    if ifind_refresh_token:
        try:
            access_token = fetch_ifind_access_token(ifind_refresh_token)
            raw = fetch_price_history_ifind_http(ifind_symbol, start_date, end_date, access_token)
            return normalize_price_frame(raw), None
        except Exception as exc:
            if not (ifind_username and ifind_password):
                return pd.DataFrame(), f"iFinD HTTP 回放取数失败: {exc}"
    if ifind_username and ifind_password:
        try:
            raw = fetch_price_history_ifind_sdk(ifind_symbol, start_date, end_date, ifind_username, ifind_password)
            return normalize_price_frame(raw), None
        except Exception as exc:
            return pd.DataFrame(), f"iFinD SDK 回放取数失败: {exc}"
    return pd.DataFrame(), "历史回放需要 iFinD refresh_token 或账号密码。"


def analyze_case_replay(frame: pd.DataFrame, config: dict[str, str], tick: float) -> dict[str, Any]:
    if frame.empty or len(frame) < 80:
        return {"ok": False, "reason": "历史行情不足，无法回放。"}
    data = add_price_indicators(frame)
    narrative_date = pd.to_datetime(config["narrative_trigger"])
    post_narrative = data.loc[data["date"] >= narrative_date].copy()
    if post_narrative.empty:
        return {"ok": False, "reason": "叙事触发日之后没有行情。"}

    narrative_row = post_narrative.iloc[0]
    confirmed = post_narrative.loc[
        (post_narrative["close"] > post_narrative["high20_prev"])
        & (post_narrative["close"] > post_narrative["ma20"])
        & (post_narrative["close"] > post_narrative["ma60"])
    ].copy()
    if confirmed.empty:
        trigger_row = narrative_row
        market_confirmed = False
    else:
        trigger_row = confirmed.iloc[0]
        market_confirmed = True

    trigger_date = trigger_row["date"]
    entry_price = float(trigger_row["close"])
    ma20 = float(trigger_row["ma20"]) if not pd.isna(trigger_row["ma20"]) else np.nan
    atr = float(trigger_row["atr14"]) if not pd.isna(trigger_row["atr14"]) else max(entry_price * 0.015, tick)
    low20 = float(trigger_row["low20_prev"]) if not pd.isna(trigger_row["low20_prev"]) else entry_price
    stop = min(ma20 - 1.15 * atr, low20 - tick) if not pd.isna(ma20) else low20 - tick

    path = data.loc[data["date"] >= trigger_date].copy()
    max_high_idx = path["high"].idxmax()
    min_low_idx = path["low"].idxmin()
    max_high_row = path.loc[max_high_idx]
    min_low_row = path.loc[min_low_idx]
    max_gain = float(max_high_row["high"] / entry_price - 1)
    max_drawdown = float(min_low_row["low"] / entry_price - 1)
    stop_hits = path.loc[path["low"] <= stop]
    stop_hit_date = None if stop_hits.empty else stop_hits.iloc[0]["date"].date()
    lag_days = int((trigger_date.date() - narrative_row["date"].date()).days)

    verdict = "价格确认有效"
    if not market_confirmed:
        verdict = "未出现市场确认"
    elif stop_hit_date is not None and stop_hit_date <= max_high_row["date"].date():
        verdict = "先触发止损"
    elif max_gain < 0.08:
        verdict = "确认后延伸不足"

    worksheet = pd.DataFrame(
        [
            {
                "节点": "叙事触发",
                "日期": narrative_row["date"].date().isoformat(),
                "价格": number_text(float(narrative_row["close"]), 0),
                "说明": config.get("note", ""),
            },
            {
                "节点": "市场确认",
                "日期": trigger_date.date().isoformat(),
                "价格": number_text(entry_price, 0),
                "说明": "收盘站上 MA20/MA60 且突破前 20 日高点" if market_confirmed else "未确认，使用叙事触发日作为观察起点",
            },
            {
                "节点": "最大有利波动",
                "日期": max_high_row["date"].date().isoformat(),
                "价格": number_text(float(max_high_row["high"]), 0),
                "说明": f"最高收益 {max_gain:.1%}",
            },
            {
                "节点": "最大不利波动",
                "日期": min_low_row["date"].date().isoformat(),
                "价格": number_text(float(min_low_row["low"]), 0),
                "说明": f"最大回撤 {max_drawdown:.1%}; 止损 {'触发 ' + stop_hit_date.isoformat() if stop_hit_date else '未触发'}",
            },
        ]
    )

    return {
        "ok": True,
        "data": data,
        "worksheet": worksheet,
        "narrative_date": narrative_row["date"].date(),
        "trigger_date": trigger_date.date(),
        "market_confirmed": market_confirmed,
        "lag_days": lag_days,
        "entry_price": entry_price,
        "stop": stop,
        "max_gain": max_gain,
        "max_drawdown": max_drawdown,
        "stop_hit_date": stop_hit_date,
        "verdict": verdict,
    }


def render_case_replay_chart(result: dict[str, Any], title: str) -> None:
    data = result["data"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["date"], y=data["close"], name="收盘价", line=dict(color="#111827", width=1.4)))
    fig.add_trace(go.Scatter(x=data["date"], y=data["ma20"], name="MA20", line=dict(color="#2563eb", width=1.0)))
    fig.add_trace(go.Scatter(x=data["date"], y=data["ma60"], name="MA60", line=dict(color="#7c3aed", width=1.0)))
    fig.add_hline(y=result["stop"], line_dash="dot", line_color="#dc2626", annotation_text="回放止损")
    for event_date, label, color in [
        (result["narrative_date"], "叙事触发", "#f59e0b"),
        (result["trigger_date"], "市场确认", "#16a34a"),
    ]:
        x_value = pd.Timestamp(event_date).isoformat()
        fig.add_shape(
            type="line",
            x0=x_value,
            x1=x_value,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color=color, dash="dash", width=1.3),
        )
        fig.add_annotation(
            x=x_value,
            y=1,
            xref="x",
            yref="paper",
            text=label,
            showarrow=False,
            yanchor="bottom",
            font=dict(color=color, size=11),
        )
    fig.update_layout(height=430, title=title, margin=dict(l=10, r=10, t=48, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")


def render_case_replay_worksheet(
    selected: str,
    commodity: dict[str, Any],
    ifind_refresh_token: str,
    ifind_username: str,
    ifind_password: str,
) -> None:
    st.subheader("历史案例价格路径回放")
    st.caption("叙事触发日为预设观察点；市场确认日由价格规则自动识别：收盘站上 MA20/MA60 且突破前 20 日高点。")
    cases = [case for case in HISTORICAL_CASES if case["commodity"] == selected]
    replay_cases = [case for case in cases if (case["commodity"], case["case"]) in CASE_REPLAY_CONFIG]
    if not replay_cases:
        st.info("当前品种还没有配置可回放案例。")
        return

    case_names = [case["case"] for case in replay_cases]
    chosen_name = st.selectbox("回放案例", case_names, key=f"case-replay-{selected}")
    config = CASE_REPLAY_CONFIG[(selected, chosen_name)]
    symbol = commodity.get("ifind_symbol", commodity["symbol"])
    frame, error = fetch_case_replay_prices(
        symbol,
        config["start"],
        config["end"],
        ifind_refresh_token,
        ifind_username,
        ifind_password,
    )
    if error:
        st.warning(error)
    if frame.empty:
        st.info("没有可回放行情数据。")
        return

    result = analyze_case_replay(frame, config, commodity["tick"])
    if not result.get("ok"):
        st.warning(result.get("reason", "回放分析失败。"))
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("市场确认", "是" if result["market_confirmed"] else "否", f"滞后 {result['lag_days']} 天")
    c2.metric("确认价", number_text(result["entry_price"], 0), f"止损 {number_text(result['stop'], 0)}")
    c3.metric("最大有利波动", f"{result['max_gain']:.1%}", result["verdict"])
    c4.metric("最大不利波动", f"{result['max_drawdown']:.1%}", "止损未触发" if result["stop_hit_date"] is None else f"止损 {result['stop_hit_date']}")
    st.dataframe(result["worksheet"], hide_index=True, width="stretch")
    render_case_replay_chart(result, f"{selected} - {chosen_name} 回放")

def render_case_library(selected: str) -> None:
    cases = [case for case in HISTORICAL_CASES if case["commodity"] == selected]
    for case in cases:
        with st.expander(case["case"], expanded=False):
            st.write(f"**天气背景：**{case['weather']}")
            st.write(f"**入场模板：**{case['entry_rule']}")
            st.write(f"**失效条件：**{case['failure_rule']}")


def main() -> None:
    inject_style()
    anchor = today_china()
    position_states = load_position_states()

    st.title("全球天气驱动的白糖/棕榈油多头监控")
    st.caption(f"每日监控日期：{anchor.isoformat()}（Asia/Shanghai）")
    st.markdown(
        '<div class="risk-callout">这是研究和交易纪律工具，不构成投资建议。面板只提示“天气主题 + 价格确认”的条件是否出现，具体下单、仓位和风控由你自行决定。</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("监控参数")
        selected = st.radio("品种", list(COMMODITIES.keys()), horizontal=False)
        commodity = COMMODITIES[selected]
        ifind_default_symbol = commodity.get("ifind_symbol", commodity["symbol"])
        ifind_default_refresh_token = secret_env_value(
            "IFIND_REFRESH_TOKEN",
            "IFIND_TOKEN",
            "IFINDB_TOKEN",
            "IFIN_D_REFRESH_TOKEN",
        ) or IFIND_DEFAULT_REFRESH_TOKEN
        ifind_default_username = secret_env_value(
            "IFIND_USERNAME",
            "IFIND_USER",
            "IFINDD_USERNAME",
        ) or IFIND_DEFAULT_USERNAME
        ifind_default_password = secret_env_value(
            "IFIND_PASSWORD",
            "IFIND_PASS",
            "IFIND_PWD",
            "IFIN_D_PASSWORD",
        ) or IFIND_DEFAULT_PASSWORD
        forecast_days = st.slider("天气前瞻天数", 7, 16, 14)
        baseline_years = st.slider("历史同期基准年数", 3, 9, 7)
        dry_ratio_trigger = st.slider("偏干触发：降雨/常年同期", 0.45, 0.95, 0.75, 0.05)
        wet_ratio_trigger = st.slider("过雨触发：降雨/常年同期", 1.10, 2.00, 1.35, 0.05)
        heat_trigger_c = st.slider("高温触发：均温高于同期", 0.5, 4.0, 1.5, 0.5)
        weather_trigger = st.slider("天气分数门槛", 40, 85, 60)
        price_trigger = st.slider("价格确认门槛", 40, 85, 55)
        build_trigger = st.slider("综合建仓门槛", 50, 90, 70)
        lookback_months = st.slider("回看期数（个月）", 1, 12, 3)
        with st.expander("资金与风险", expanded=True):
            account_size = st.number_input("账户权益（元）", min_value=10000.0, value=2000000.0, step=100000.0)
            risk_pct = st.slider("单笔止损风险", 0.2, 3.0, 1.0, 0.1) / 100.0
            max_margin_pct = st.slider("单品种保证金占用上限", 2.0, 40.0, 15.0, 1.0) / 100.0
            trial_fraction = st.slider("试仓占风险上限比例", 10, 50, 25, 5) / 100.0
            margin_rate = st.slider(
                "估算保证金率",
                5.0,
                25.0,
                float(commodity.get("default_margin_rate", 0.12)) * 100,
                0.5,
            ) / 100.0
            portfolio_risk_multiplier = st.slider("组合止损风险上限（倍单笔）", 1.0, 3.0, 1.5, 0.1)
            portfolio_margin_multiplier = st.slider("组合保证金上限（倍单品种）", 1.0, 3.0, 1.5, 0.1)
            correlation_trigger = st.slider("同向相关性限制", 0.30, 0.90, 0.65, 0.05)
            correlation_lookback = st.slider("相关性回看交易日", 20, 120, 60, 10)
        risk_config = {
            "account_size": float(account_size),
            "risk_pct": float(risk_pct),
            "max_margin_pct": float(max_margin_pct),
            "trial_fraction": float(trial_fraction),
            "margin_rate": float(margin_rate),
            "portfolio_risk_multiplier": float(portfolio_risk_multiplier),
            "portfolio_margin_multiplier": float(portfolio_margin_multiplier),
            "correlation_trigger": float(correlation_trigger),
            "correlation_lookback": int(correlation_lookback),
        }
        position_states = render_position_state_editor(selected, position_states, anchor)
        ifind_symbol = ifind_default_symbol
        ifind_refresh_token = ifind_default_refresh_token
        ifind_username = ifind_default_username
        ifind_password = ifind_default_password
        price_source = resolve_price_source(ifind_refresh_token, ifind_username, ifind_password)
        credential_status = "loaded" if has_ifind_credentials(ifind_refresh_token, ifind_username, ifind_password) else "not configured"
        st.caption(f"Market data mode: {price_source}; iFinD credentials: {credential_status}. Inputs are hidden from the UI.")

        settings = RuleSettings(
            forecast_days=forecast_days,
            baseline_years=baseline_years,
            dry_ratio_trigger=dry_ratio_trigger,
            wet_ratio_trigger=wet_ratio_trigger,
            heat_trigger_c=heat_trigger_c,
            weather_trigger=weather_trigger,
            price_trigger=price_trigger,
            build_trigger=build_trigger,
        )

    with st.spinner("正在更新天气和行情数据..."):
        weather, weather_errors = build_all_weather_table(settings, anchor)
        enso_frame, enso_error = fetch_enso_oni()
        price_frames: dict[str, pd.DataFrame] = {}
        price_errors: list[str] = []
        price_symbols: dict[str, str] = {}

        for loop_name, loop_commodity in COMMODITIES.items():
            loop_ifind_symbol = loop_commodity.get("ifind_symbol", loop_commodity["symbol"])
            if price_source == PRICE_SOURCE_IFIND and loop_name == selected:
                loop_ifind_symbol = ifind_symbol
            price_symbols[loop_name] = loop_ifind_symbol if price_source in {PRICE_SOURCE_IFIND, PRICE_SOURCE_STATIC} else loop_commodity["symbol"]
            frame, error = get_price_data(
                loop_name,
                loop_commodity,
                lookback_months,
                price_source,
                ifind_symbol=loop_ifind_symbol,
                ifind_refresh_token=ifind_refresh_token,
                ifind_username=ifind_username,
                ifind_password=ifind_password,
            )
            price_symbols[loop_name] = frame.attrs.get("price_symbol", price_symbols.get(loop_name, loop_ifind_symbol))
            price_frames[loop_name] = frame
            if error:
                price_errors.append(error)


    if weather.empty and weather_errors:
        st.warning("天气接口出现异常，部分区域数据未取到；面板会继续使用已返回的数据。")

    if weather_errors:
        with st.expander("天气接口错误明细", expanded=weather.empty):
            for error in weather_errors:
                st.write(error)

    if price_errors:
        with st.expander("行情接口错误", expanded=True):
            for error in price_errors:
                st.write(error)

    with st.expander("手动行情 CSV", expanded=price_source != PRICE_SOURCE_AKSHARE or bool(price_errors)):
        manual = load_manual_csv(selected, key=f"manual-{selected}")
        if manual is not None:
            manual_frame = normalize_price_frame(manual)
            attach_price_metadata(manual_frame, price_symbols.get(selected, ifind_symbol if price_source in {PRICE_SOURCE_IFIND, PRICE_SOURCE_STATIC} else commodity["symbol"]))
            price_frames[selected] = manual_frame
            price_symbols[selected] = manual_frame.attrs.get("price_symbol", price_symbols.get(selected, ""))

    render_data_source_status(weather, price_frames, enso_frame)

    snapshots = build_all_commodity_snapshots(
        weather,
        weather_errors,
        price_frames,
        price_errors,
        enso_frame,
        enso_error,
        settings,
        risk_config,
        anchor,
        price_symbols=price_symbols,
        position_states=position_states,
    )
    selected_snapshot = snapshots[selected]
    selected_price_frame = price_frames.get(selected, pd.DataFrame())
    weather_score = selected_snapshot["weather_score"]
    price_signal = selected_snapshot["price_signal"]
    signal_health = selected_snapshot["signal_health"]
    regime_context = selected_snapshot["regime_context"]
    action = selected_snapshot["action"]
    note = selected_snapshot["note"]
    combined = selected_snapshot["combined"]
    current_position_plan = selected_snapshot["position_plan"]
    signal_log_row = build_signal_log_row(
        selected,
        commodity,
        anchor,
        action,
        note,
        combined,
        weather_score,
        price_signal,
        signal_health,
        regime_context,
        current_position_plan,
        weather,
        price_source,
        ifind_symbol,
        selected_snapshot.get("entry_playbook"),
        selected_snapshot.get("post_entry_playbook"),
        selected_snapshot.get("portfolio_gate"),
        selected_snapshot.get("portfolio_summary"),
        selected_snapshot.get("market_session"),
    )

    render_morning_summary(snapshots, settings, anchor)
    st.divider()
    top1, top2, top3 = st.columns([1.1, 1.1, 1.4])
    with top1:
        signal_box(selected, action, note, combined)
    with top2:
        st.metric("入场天气分数", "n/a" if pd.isna(weather_score) else f"{weather_score:.1f}", f"门槛 {settings.weather_trigger}")
        st.metric("价格确认分", f"{price_signal['score']}", f"门槛 {settings.price_trigger}")
    with top3:
        st.write("**交易逻辑**")
        st.write(commodity["thesis"])
        st.markdown(
            f'<div class="small-muted">历史模板：{" / ".join(commodity["case_tags"])}</div>',
            unsafe_allow_html=True,
        )

    tab_signal, tab_regime, tab_weather, tab_price, tab_pair, tab_cases, tab_log, tab_rules = st.tabs(
        ["建仓信号", "Regime背景", "天气信息", "价格图表", "白糖/棕榈对照", "历史模板", "运行日志", "规则说明"]
    )

    with tab_signal:
        render_signal_health(signal_health)
        st.divider()
        render_entry_plan(selected, commodity, weather_score, price_signal, settings, risk_config, signal_health, selected_snapshot.get("entry_playbook"), selected_snapshot.get("post_entry_playbook"), selected_snapshot.get("portfolio_gate"), selected_snapshot.get("market_session"))
        st.divider()
        render_regime_stress_test_panel(snapshots, risk_config, settings)
        st.divider()
        st.subheader("触发条件明细")
        col_a, col_b = st.columns(2)
        with col_a:
            st.write("**价格条件**")
            for item in price_signal["conditions"]:
                st.write(f"- {item}")
        with col_b:
            st.write("**天气主驱动区域**")
            if weather.empty or "commodity" not in weather.columns:
                st.info("暂无天气数据。")
            else:
                selected_weather = weather.loc[weather["commodity"] == selected].copy()
                sort_column = "entry_weighted_score" if "entry_weighted_score" in selected_weather.columns else "weighted_score"
                top_weather = selected_weather.sort_values(sort_column, ascending=False).head(5)
                for _, row in top_weather.iterrows():
                    st.write(
                        f"- {row['region']}：入场 {numeric_value(row.get('entry_ready_score', row['stress_score'])):.1f} / 压力 {row['stress_score']:.1f}，"
                        f"{row['driver']}，时滞 {row.get('impact_label', 'n/a')}，"
                        f"降雨比 {pct_text(row['precip_ratio'])}，高温偏离 {number_text(row['tmax_anom_c'], 1)}°C"
                    )

    with tab_regime:
        render_regime_context(regime_context)

    with tab_weather:
        render_weather_map(weather, selected)
        cols = [
            "region",
            "country",
            "driver",
            "persistence_label",
            "weight",
            "raw_stress_score",
            "stress_score",
            "impact_label",
            "entry_ready_score",
            "impact_multiplier",
            "impact_reason",
            "entry_weighted_score",
            "persistence_score",
            "persistence_multiplier",
            "dry_score",
            "wet_score",
            "precip_mm",
            "normal_precip_mm",
            "precip_ratio",
            "precip_percentile",
            "recent_precip_ratio",
            "recent_precip_percentile",
            "tmax_c",
            "normal_tmax_c",
            "tmax_anom_c",
            "tmax_percentile",
            "water_balance_mm",
            "water_balance_percentile",
            "climatology_sample_count",
            "note",
        ]
        if weather.empty or "commodity" not in weather.columns:
            st.info("暂无天气数据。")
        else:
            available_cols = [col for col in cols if col in weather.columns]
            show = weather.loc[weather["commodity"] == selected, available_cols].copy()
            st.dataframe(
                show.style.format(
                    {
                        "weight": "{:.2f}",
                        "raw_stress_score": "{:.1f}",
                        "stress_score": "{:.1f}",
                        "entry_ready_score": "{:.1f}",
                        "impact_multiplier": "{:.2f}",
                        "entry_weighted_score": "{:.1f}",
                        "persistence_score": "{:.1f}",
                        "persistence_multiplier": "{:.2f}",
                        "dry_score": "{:.1f}",
                        "wet_score": "{:.1f}",
                        "precip_mm": "{:.1f}",
                        "normal_precip_mm": "{:.1f}",
                        "precip_ratio": "{:.0%}",
                        "precip_percentile": "{:.0%}",
                        "recent_precip_ratio": "{:.0%}",
                        "recent_precip_percentile": "{:.0%}",
                        "tmax_c": "{:.1f}",
                        "normal_tmax_c": "{:.1f}",
                        "tmax_anom_c": "{:.1f}",
                        "tmax_percentile": "{:.0%}",
                        "water_balance_mm": "{:.1f}",
                        "water_balance_percentile": "{:.0%}",
                    }
                ),
                width="stretch",
                hide_index=True,
            )

    with tab_price:
        render_price_chart(selected, price_signal, commodity["unit"])
        latest = price_signal.get("latest")
        if latest is not None:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("MA20", number_text(latest["ma20"], 0))
            m2.metric("MA60", number_text(latest["ma60"], 0))
            m3.metric("20 日前高", number_text(latest["high20_prev"], 0))
            m4.metric("ATR14", number_text(latest["atr14"], 0))


    with tab_pair:
        render_multi_commodity_price_snapshot(price_frames, lookback_months)

    with tab_cases:
        render_case_library(selected)
        st.divider()
        render_case_replay_worksheet(selected, commodity, ifind_refresh_token, ifind_username, ifind_password)

    with tab_log:
        render_signal_log_panel(signal_log_row, settings, price_frames)

    with tab_rules:
        st.write("**分数结构**")
        st.write(
            f"- 综合分 = 天气分数 * {commodity['weather_weight']:.0%} + 价格确认分 * {commodity['price_weight']:.0%}。"
        )
        st.write("- 天气分数按主产区权重汇总，核心观察降雨相对历史同期、最高温偏离、降雨减 ET0 的水分差。")
        st.write("- 价格确认分来自 20/60/120 日均线、20 日突破、成交量和持仓量。")
        st.write("**建仓纪律**")
        st.write("- 天气达标但价格未达标：只做观察，不提前重仓。")
        st.write("- 价格达标但天气未达标：按普通趋势处理，不把它归因于天气。")
        st.write("- 同时达标：先试仓，再等突破延续或 20 日均线回踩确认加仓。")
        st.write("**数据源**")
        st.write("- 天气：Open-Meteo Forecast API + Historical Weather API。")
        st.write("- Regime背景：NOAA CPC ONI + 静态生产季窗口；不计入短期天气分。")
        st.write("- 行情：AKShare `futures_main_sina`（SR0 白糖，P0 棕榈油）或 iFinD。")
        st.write(
            "- iFinD 优先使用 HTTP 的 `cmd_history_quotation`（需 refresh_token），"
            "如失败则可回退到 SDK 登录取数。"
        )


if __name__ == "__main__":
    main()


