# -*- coding: utf-8 -*-
"""规划器：把自然语言问题解析为「工具调用链」（steps）。

双模设计：
  - LLM 规划器（配置 AQ_LLM_BASE_URL / AQ_LLM_API_KEY / AQ_LLM_MODEL 后启用）：
    以 function calling 风格把工具 schema 交给大模型，由它针对每个问题
    现生成工具调用链——任意问法都能规划，不再是固定意图查表；
  - 本地规则规划器（默认兜底，无网无 Key 可跑）：
    意图识别后生成同构的 steps。
两条路径产出统一的 TaskPlan(steps=[Step(tool, args, thought)])，
由 ReAct 引擎逐步执行，每一步作为独立事件流式推送到前端。
"""
import json
import os
import re
import urllib.request
from dataclasses import dataclass, field

from .geo import CITY_TO_PROVINCE
from . import config

CITY_NAMES = sorted(CITY_TO_PROVINCE.keys(), key=len, reverse=True)

POLLUTANTS = ["PM2.5", "PM10", "O3", "SO2", "NO2", "CO"]

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

# ---------------- 工具调用链（steps）数据结构 ----------------

@dataclass
class Step:
    """一步工具调用：工具名 + 参数 + 这一步的思考（ReAct 的 Thought）。"""
    tool: str
    args: dict = field(default_factory=dict)
    thought: str = ""


@dataclass
class TaskPlan:
    """规划产物：工具调用链 + 关键槽位（兼容旧字段）。"""
    question: str = ""
    steps: list = field(default_factory=list)
    pollutant: str = "PM2.5"
    cities: list = field(default_factory=list)
    year: int = None
    season: str = None
    top_n: int = 5
    chart: str = "auto"
    audience: str = "simple"
    dim: str = "季节"
    source: str = "rule"           # rule / llm
    thoughts: list = field(default_factory=list)

    # ---- 兼容旧接口的属性 ----
    @property
    def raw(self):
        return self.question

    @property
    def intent(self):
        """主意图：由调用链推断（供任务看板 / 报告统计）。"""
        tools = [s.tool for s in self.steps]
        if "anomaly" in tools:
            return "anomaly"
        if "trend" in tools:
            return "monthly_trend"
        for s in self.steps:
            if s.tool == "aggregate":
                gb, m = s.args.get("group_by"), s.args.get("metric")
                if gb == "province" and m == "exceed_days":
                    return "exceed_share"
                if gb in ("season", "north_south"):
                    return "compare"
                if gb == "city" and m == "mean":
                    return "annual_rank"
                if gb == "city" and m == "exceed_days":
                    return "city_exceed"
        if "season_exceed" in tools:
            return "city_season"
        return "annual_rank"

    def step_summary(self):
        return [{"tool": s.tool, "thought": s.thought, "args": s.args} for s in self.steps]


# ---------------- 工具 schema（交给 LLM 的 function calling 描述） ----------------

TOOL_SPECS = {
    "load_data": {
        "desc": "分片读取并清洗本地空气质量数据集。必须是调度链的第一步。",
        "args": {},
    },
    "aggregate": {
        "desc": "分组聚合统计，可自由组合维度与指标。",
        "args": {
            "group_by": "city=按城市 | province=按省份 | season=按季节 | north_south=按南北方",
            "metric": "mean=平均浓度 | exceed_days=超标天数统计",
            "pollutant": "PM2.5 / PM10 / O3 / SO2 / NO2 / CO",
            "cities": "可选，限定城市列表，如 ['北京','上海']",
            "year": "可选，年份，如 2024",
            "top_n": "可选，返回前 N 行，默认 5",
        },
    },
    "trend": {
        "desc": "指定城市的月度浓度趋势（生成折线图数据）；cities 为空时取年均浓度 TOP5 城市。",
        "args": {"cities": "可选，城市列表", "pollutant": "污染物", "year": "可选，年份"},
    },
    "season_exceed": {
        "desc": "单个城市某季节（或全年）的超标天数统计。",
        "args": {"city": "城市名，如 北京", "season": "春季/夏季/秋季/冬季/全年",
                 "pollutant": "污染物", "year": "可选，年份"},
    },
    "anomaly": {
        "desc": "MAD 3σ 时序异常跳变检测，标记污染突发时段。",
        "args": {"city": "城市名", "pollutant": "污染物", "year": "可选，年份"},
    },
    "check_standard": {
        "desc": "查询 GB 3095-2012 二级标准限值口径，用于超标研判。",
        "args": {"pollutant": "污染物", "kind": "可选 annual/24h/8h/1h"},
    },
    "plot_chart": {
        "desc": "把最近一次聚合结果画成图。须放在 aggregate/trend/season_exceed/anomaly 之后。",
        "args": {"chart": "可选 auto/line/bar/pie，默认 auto 按结果类型自动选择"},
    },
    "validate": {
        "desc": "对统计结果做合理性校验（占比合计=100%、排名单调性、样本量）。建议作为最后一步。",
        "args": {},
    },
}

VALID_TOOLS = set(TOOL_SPECS)

LLM_SYS_PROMPT = """你是空气质量时序分析 Agent 的任务规划器。用户提出一个自然语言问题，你需要为它生成一条「工具调用链」（steps），让执行引擎按链调度工具完成分析。

数据集事实（不要臆造）：
- 来源：中国环境监测总站公开数据（CN-AIR），覆盖全国 375 个城市，逐日六项污染物 + AQI；
- 污染物与判定口径：PM2.5(24h 平均)、PM10(24h)、SO2(24h)、NO2(24h)、O3(日最大8h 平均)、CO(24h)；
- 可用年份：{years}（未提年份用默认年份 {default_year}）。

可用工具（function calling 风格）：
{tools}

输出要求：
1. 只输出一个 JSON 对象，不要输出任何其他文字；
2. 格式：{{"thoughts": ["规划思路…"], "pollutant": "PM2.5", "steps": [{{"tool": "工具名", "args": {{...}}, "thought": "这一步为什么这样做"}}, ...]}}；
3. steps 中第一个工具必须是 load_data，最后一个建议是 validate；plot_chart 可选，放在产出统计结果之后；
4. 城市名写常见名即可（如 北京、和田），系统会自动校正为标准名；
5. 每步都要有 thought 字段，写清楚这一步在解决用户问题的哪个部分。
"""


def _llm_sys_prompt(default_year, available_years):
    years = "、".join(str(y) for y in available_years) if available_years else "2024"
    tools = json.dumps(TOOL_SPECS, ensure_ascii=False, indent=1)
    return LLM_SYS_PROMPT.format(years=years, default_year=default_year or "最新年份",
                                 tools=tools)


# ---------------- 槽位识别（规则规划器用） ----------------

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


def normalize_city(name):
    """把 LLM/用户写的城市名校正为数据集标准名（北京→北京市、和田→和田地区）。"""
    if not name:
        return None
    n = str(name).strip()
    if n in CITY_TO_PROVINCE:
        return n
    for c in CITY_NAMES:            # 标准名以输入开头：北京 → 北京市
        if c.startswith(n):
            return c
    for c in CITY_NAMES:            # 输入比标准名长：北京市城区 → 北京市
        if n.startswith(c):
            return c
    return None


def normalize_cities(names):
    out = []
    for n in names or []:
        c = normalize_city(n)
        if c and c not in out:
            out.append(c)
    return out


def _valid_pollutant(p):
    p = str(p or "").strip().upper()
    return p if p in POLLUTANTS else None


def _valid_year(y, default_year=None):
    try:
        y = int(y)
    except (TypeError, ValueError):
        return default_year
    return y if 2000 <= y <= 2100 else default_year


# ---------------- 规则规划器：意图 → 同构 steps ----------------

def build_steps(intent, pollutant, cities, year, season, top_n, dim, chart):
    """按识别出的槽位生成标准工具调用链（与 LLM 路径同构）。"""
    S = Step
    load = S("load_data", {},
             "先分片读取本地数据集并清洗，确保后续统计基于真实监测数据")
    ck = S("check_standard", {"pollutant": pollutant},
           f"对照 GB 3095-2012 二级限值，明确 {pollutant} 的判定口径")
    val = S("validate", {}, "对统计结果做合理性校验，降低规划错误风险")
    plot = lambda c: S("plot_chart", {"chart": c}, f"将统计结果可视化为{c}图")

    if intent == "annual_rank":
        return [load,
                S("aggregate", {"group_by": "city", "metric": "mean", "pollutant": pollutant,
                                "year": year, "top_n": top_n},
                  f"按城市分组计算 {pollutant} 年均浓度，取 TOP{top_n}"),
                ck, plot(chart or "bar"), val]
    if intent == "monthly_trend":
        return [load,
                S("trend", {"cities": cities, "pollutant": pollutant, "year": year},
                  f"计算 {pollutant} 月度浓度变化曲线（未指定城市时取年均 TOP{top_n}）"),
                ck, plot(chart or "line"), val]
    if intent == "exceed_share":
        return [load,
                S("aggregate", {"group_by": "province", "metric": "exceed_days",
                                "pollutant": pollutant, "cities": cities, "year": year},
                  f"按省份统计 {pollutant} 超标天数并计算占全国比例"),
                ck, plot(chart or "pie"), val]
    if intent == "city_season":
        return [load,
                S("season_exceed", {"city": cities[0] if cities else "北京",
                                    "season": season or "全年",
                                    "pollutant": pollutant, "year": year},
                  f"统计{cities[0] if cities else '北京'}{season or '全年'}的 {pollutant} 超标天数"),
                ck, plot(chart or "bar"), val]
    if intent == "compare":
        return [load,
                S("aggregate", {"group_by": dim if dim in ("season", "north_south") else "season",
                                "metric": "mean", "pollutant": pollutant, "year": year},
                  f"按{dim}分组对比 {pollutant} 平均浓度"),
                ck, plot(chart or "bar"), val]
    if intent == "anomaly":
        return [load,
                S("anomaly", {"city": cities[0] if cities else "北京",
                              "pollutant": pollutant, "year": year},
                  f"检测{cities[0] if cities else '北京'} {pollutant} 时序异常跳变点"),
                ck, plot(chart or "line"), val]
    # 默认：年均排名
    return [load,
            S("aggregate", {"group_by": "city", "metric": "mean", "pollutant": pollutant,
                            "year": year, "top_n": top_n},
              f"按城市分组计算 {pollutant} 年均浓度，取 TOP{top_n}"),
            ck, plot(chart or "bar"), val]


def rule_plan(question: str, memory=None, default_year=None) -> TaskPlan:
    """本地规则规划器（无网无 Key 兜底）。"""
    p = TaskPlan(question=question, source="rule")
    p.pollutant = _find_pollutant(question)
    intent = _find_intent(question)
    p.year = _find_year(question, default_year)
    p.top_n = _find_top_n(question)
    p.season = _find_season(question)
    p.audience = _find_audience(question)
    p.cities = _find_cities(question)
    p.dim = "north_south" if ("南方" in question or "北方" in question) else "season"

    # 多轮记忆：「这几个城市」等指代继承上一轮城市
    if not p.cities and memory and memory.last_cities:
        if re.search(r"这|那|上述|刚才|上次|继续", question):
            p.cities = list(memory.last_cities)
            p.thoughts.append(f"[记忆] 继承上一轮分析城市: {p.cities}")
    if p.year is None and memory and memory.last_year:
        p.year = memory.last_year

    # 意图细化
    if p.cities and p.season and intent in ("annual_rank", "city_season"):
        intent = "city_season"
    if p.cities and intent == "annual_rank" and re.search(r"月度|趋势|曲线", question):
        intent = "monthly_trend"
    if intent == "exceed_share" and not re.search(r"省份|省", question):
        intent = "exceed_share"

    chart = _find_chart(question, intent)
    p.chart = chart
    p.steps = build_steps(intent, p.pollutant, p.cities, p.year, p.season,
                          p.top_n, p.dim, chart)
    p.thoughts.append(f"[规划] 识别意图={intent}, 污染物={p.pollutant}")
    p.thoughts.append(f"[规划] 参数: 年份={p.year}, 城市={p.cities or '全国'}, "
                      f"季节={p.season or '全年'}, TOP={p.top_n}, 图表={chart}")
    p.thoughts.append(f"[规划] 生成 {len(p.steps)} 步工具调用链: "
                      + " → ".join(s.tool for s in p.steps))
    return p


# ---------------- LLM 规划器：function calling 风格 ----------------

def _clean_args(tool, args):
    """清洗 LLM 输出的工具参数：白名单校验 + 城市名校正。"""
    a = dict(args) if isinstance(args, dict) else {}
    if tool == "aggregate":
        a["group_by"] = a.get("group_by") if a.get("group_by") in \
            ("city", "province", "season", "north_south") else "city"
        a["metric"] = a.get("metric") if a.get("metric") in ("mean", "exceed_days") else "mean"
    if "pollutant" in a:
        a["pollutant"] = _valid_pollutant(a.get("pollutant")) or "PM2.5"
    if "year" in a:
        a["year"] = _valid_year(a.get("year"))
        if a["year"] is None:
            a.pop("year")
    if "top_n" in a:
        try:
            a["top_n"] = max(1, min(50, int(a["top_n"])))
        except (TypeError, ValueError):
            a.pop("top_n")
    if "cities" in a:
        a["cities"] = normalize_cities(a.get("cities") if isinstance(a.get("cities"), list) else [])
        if not a["cities"]:
            a.pop("cities")
    if "city" in a:
        c = normalize_city(a.get("city"))
        if c:
            a["city"] = c
        else:
            a.pop("city")
    if "chart" in a and a.get("chart") not in ("auto", "line", "bar", "pie"):
        a["chart"] = "auto"
    if "season" in a and a.get("season") not in ("春季", "夏季", "秋季", "冬季", "全年", None):
        a.pop("season")
    return a


def _sanitize_llm_steps(data, question):
    """校验 LLM 输出的调用链：工具白名单 + 参数清洗 + 结构补齐。"""
    raw_steps = data.get("steps") or []
    if not isinstance(raw_steps, list):
        raise ValueError("LLM 输出的 steps 不是列表")
    steps = []
    for s in raw_steps:
        if not isinstance(s, dict):
            continue
        tool = str(s.get("tool", "")).strip()
        if tool not in VALID_TOOLS:
            continue
        steps.append(Step(tool, _clean_args(tool, s.get("args")),
                          str(s.get("thought", ""))[:200]))
    if not steps:
        raise ValueError("LLM 未给出有效工具调用")
    if not any(s.tool == "load_data" for s in steps):
        steps.insert(0, Step("load_data", {},
                             "先分片读取本地数据集并清洗，确保统计基于真实监测数据"))
    return steps


def llm_plan(question: str, memory=None, default_year=None, available_years=None) -> TaskPlan:
    """调用 OpenAI 兼容接口做规划；任何失败都抛异常，由调用方回退到规则规划器。"""
    base = os.environ.get("AQ_LLM_BASE_URL", "").rstrip("/")
    key = os.environ.get("AQ_LLM_API_KEY", "")
    model = os.environ.get("AQ_LLM_MODEL", "gpt-4o-mini")
    if not base or not key:
        raise RuntimeError("未配置 AQ_LLM_BASE_URL / AQ_LLM_API_KEY")

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": _llm_sys_prompt(default_year, available_years)},
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
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    content = data["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", content, re.S)
    if not m:
        raise ValueError("LLM 输出中未找到 JSON 计划")
    plan_data = json.loads(m.group(0))

    p = TaskPlan(question=question, source="llm")
    p.pollutant = _valid_pollutant(plan_data.get("pollutant")) or "PM2.5"
    p.cities = normalize_cities(plan_data.get("cities") if isinstance(plan_data.get("cities"), list) else [])
    p.year = _valid_year(plan_data.get("year"), default_year)
    p.steps = _sanitize_llm_steps(plan_data, question)
    for t in (plan_data.get("thoughts") or [])[:5]:
        if isinstance(t, str) and t.strip():
            p.thoughts.append(t.strip()[:200])
    p.thoughts.append(f"[规划] 由 LLM 生成本问题专属的 {len(p.steps)} 步工具调用链")
    return p


def plan(question: str, memory=None, default_year=None, use_llm=None,
         available_years=None) -> TaskPlan:
    """规划入口：配置了 LLM 则优先 LLM（任意问法），任何失败自动回退规则规划器。"""
    if use_llm is None:
        use_llm = bool(os.environ.get("AQ_LLM_BASE_URL") and os.environ.get("AQ_LLM_API_KEY"))
    if use_llm:
        try:
            return llm_plan(question, memory, default_year, available_years)
        except Exception as e:
            p = rule_plan(question, memory, default_year)
            p.thoughts.append(f"[规划] LLM 规划失败({type(e).__name__}: {e})，"
                              f"已回退本地规则规划器")
            return p
    return rule_plan(question, memory, default_year)
