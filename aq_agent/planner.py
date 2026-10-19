# -*- coding: utf-8 -*-
"""规划器：把自然语言问题解析为 TaskPlan。

双模设计：
  - 默认「本地规则规划器」：无需联网、无需 API Key，比赛现场稳定可用；
  - 可选「LLM 规划器」：设置环境变量 AQ_LLM_BASE_URL / AQ_LLM_API_KEY / AQ_LLM_MODEL
    后启用，由大模型输出结构化计划；任何失败都会自动回退到规则规划器。
"""
import json
import os
import re
import urllib.request
from dataclasses import dataclass, field

from .geo import CITY_TO_PROVINCE
from . import config

CITY_NAMES = sorted(CITY_TO_PROVINCE.keys(), key=len, reverse=True)

POLLUTANT_PATTERNS = [
    ("PM2.5", [r"pm2\.?5", r"细颗粒物"]),
    ("PM10", [r"pm10", r"pm\s*10", r"可吸入颗粒物"]),
    ("O3", [r"o₃", r"o3", r"臭氧"]),
    ("SO2", [r"so₂", r"so2", r"二氧化硫"]),
    ("NO2", [r"no₂", r"no2", r"二氧化氮"]),
    ("CO", [r"\bco\b", r"一氧化碳"]),
]

INTENT_PATTERNS = [
    ("monthly_trend", [r"月度", r"变化曲线", r"趋势", r"折线"]),
    ("exceed_share", [r"占比", r"比例", r"分布"]),
    ("city_season", [r"春季|夏季|秋季|冬季", r"总天数", r"多少天"]),
    ("anomaly", [r"异常", r"跳变", r"突发", r"告警"]),
    ("compare", [r"对比", r"比较", r"南北方", r"差异"]),
    ("annual_rank", [r"年均", r"年度", r"排名", r"top", r"最高", r"最低", r"前\s*\d+"]),
]

SEASON_WORDS = {"春": "春季", "夏": "夏季", "秋": "秋季", "冬": "冬季"}


@dataclass
class TaskPlan:
    intent: str = "annual_rank"
    pollutant: str = "PM2.5"
    cities: list = field(default_factory=list)
    provinces: list = field(default_factory=list)
    year: int = None
    season: str = None
    top_n: int = 5
    chart: str = None            # line / bar / pie
    audience: str = "simple"     # simple / pro
    dim: str = "季节"             # compare 的对比维度
    raw: str = ""
    thoughts: list = field(default_factory=list)   # 规划推理过程
    source: str = "rule"         # rule / llm


def _find_pollutant(text: str) -> str:
    t = text.lower()
    for name, pats in POLLUTANT_PATTERNS:
        for p in pats:
            if re.search(p, t):
                return name
    return "PM2.5"


def _find_cities(text: str):
    found = []
    for c in CITY_NAMES:  # 长的优先，避免「北京」截断「北京市」
        if c in text:
            found.append(c)
    return found


def _find_year(text: str, default_year=None):
    m = re.search(r"(20\d{2})\s*年", text)
    if m:
        return int(m.group(1))
    return default_year


def _find_top_n(text: str) -> int:
    m = re.search(r"top\s*(\d+)", text, re.I) or re.search(r"前\s*(\d+)", text) \
        or re.search(r"(\d+)\s*个", text)
    return int(m.group(1)) if m else 5


def _find_season(text: str):
    for w, s in SEASON_WORDS.items():
        if w + "季" in text:
            return s
    return None


def _find_intent(text: str) -> str:
    t = text.lower()
    for intent, pats in INTENT_PATTERNS:
        for p in pats:
            if re.search(p, t):
                return intent
    return "annual_rank"


def _find_chart(text: str, intent: str):
    t = text.lower()
    if "饼" in t or "占比" in t:
        return "pie"
    if "柱状" in t or "柱" in t or "排" in t:
        return "bar"
    if "曲线" in t or "折线" in t or "月度" in t or "趋势" in t:
        return "line"
    return {"monthly_trend": "line", "exceed_share": "pie",
            "annual_rank": "bar", "city_season": "bar",
            "anomaly": "line", "compare": "bar"}.get(intent, "bar")


def _find_audience(text: str) -> str:
    return "pro" if ("专业" in text or "从业者" in text) else "simple"


def rule_plan(question: str, memory=None, default_year=None) -> TaskPlan:
    """本地规则规划器。"""
    p = TaskPlan(raw=question)
    p.pollutant = _find_pollutant(question)
    p.intent = _find_intent(question)
    p.year = _find_year(question, default_year)
    p.top_n = _find_top_n(question)
    p.season = _find_season(question)
    p.chart = _find_chart(question, p.intent)
    p.audience = _find_audience(question)
    p.cities = _find_cities(question)
    p.dim = "南北方" if ("南方" in question or "北方" in question) else "季节"

    # 多轮记忆：「这几个城市」等指代继承上一轮城市
    if not p.cities and memory and memory.last_cities:
        if re.search(r"这|那|上述|刚才|上次|继续", question):
            p.cities = list(memory.last_cities)
            p.thoughts.append(f"[记忆] 继承上一轮分析城市: {p.cities}")
    if p.year is None and memory and memory.last_year:
        p.year = memory.last_year

    # 意图细化
    if p.intent == "exceed_share" and not re.search(r"省份|省", question):
        p.intent = "exceed_share"
    if p.cities and p.season and p.intent in ("annual_rank", "city_season"):
        p.intent = "city_season"
    if p.cities and p.intent == "annual_rank" and re.search(r"月度|趋势|曲线", question):
        p.intent = "monthly_trend"

    # 规划推理记录（ReAct 的 Thought 会引用）
    p.thoughts.append(f"[规划] 识别意图={p.intent}, 污染物={p.pollutant}")
    p.thoughts.append(f"[规划] 参数: 年份={p.year}, 城市={p.cities or '全国'}, "
                      f"季节={p.season or '全年'}, TOP={p.top_n}, 图表={p.chart}")
    return p


# ---------------- 可选 LLM 规划器 ----------------

LLM_SYS_PROMPT = """你是空气质量数据分析 Agent 的任务规划器。把用户的自然语言问题解析为 JSON 计划。
可用字段：intent(annual_rank/monthly_trend/exceed_share/city_season/anomaly/compare),
pollutant(PM2.5/PM10/O3/SO2/NO2/CO), cities(城市列表), year(年份或null),
season(春季/夏季/秋季/冬季/null), top_n(整数), chart(line/bar/pie/null),
audience(simple/pro), dim(季节/南北方)。
只输出 JSON，不要输出其他内容。"""


def llm_plan(question: str, memory=None, default_year=None) -> TaskPlan:
    """调用 OpenAI 兼容接口做规划；任何失败都抛异常，由调用方回退到规则规划器。"""
    base = os.environ.get("AQ_LLM_BASE_URL", "").rstrip("/")
    key = os.environ.get("AQ_LLM_API_KEY", "")
    model = os.environ.get("AQ_LLM_MODEL", "gpt-4o-mini")
    if not base or not key:
        raise RuntimeError("未配置 AQ_LLM_BASE_URL / AQ_LLM_API_KEY")

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": LLM_SYS_PROMPT},
            {"role": "user", "content": question},
        ],
        "temperature": 0,
    }
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    content = data["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", content, re.S)
    plan = json.loads(m.group(0))
    p = TaskPlan(raw=question, source="llm")
    p.intent = plan.get("intent", "annual_rank")
    p.pollutant = plan.get("pollutant", "PM2.5")
    p.cities = plan.get("cities") or []
    p.year = plan.get("year") or default_year
    p.season = plan.get("season")
    p.top_n = int(plan.get("top_n") or 5)
    p.chart = plan.get("chart") or _find_chart(question, p.intent)
    p.audience = plan.get("audience", "simple")
    p.thoughts.append("[规划] 由 LLM 生成任务计划")
    return p


def plan(question: str, memory=None, default_year=None, use_llm=None) -> TaskPlan:
    """规划入口：默认规则；配置了 LLM 且未禁用时优先 LLM，失败自动回退。"""
    if use_llm is None:
        use_llm = bool(os.environ.get("AQ_LLM_BASE_URL") and os.environ.get("AQ_LLM_API_KEY"))
    if use_llm:
        try:
            return llm_plan(question, memory, default_year)
        except Exception as e:
            p = rule_plan(question, memory, default_year)
            p.thoughts.append(f"[规划] LLM 规划失败({e})，已回退到本地规则规划器")
            return p
    return rule_plan(question, memory, default_year)
