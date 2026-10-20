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

# 省份名（按长度降序，避免「新疆」截断「新疆生产建设兵团」式误配）
PROVINCE_NAMES = sorted(set(CITY_TO_PROVINCE.values()), key=len, reverse=True)

# 常见城市别称 → 数据集标准名（口语问法：「俺们首都的 pm2.5…」）
CITY_ALIASES = {
    "首都": "北京", "魔都": "上海", "羊城": "广州", "鹏城": "深圳",
    "金陵": "南京", "石头城": "南京", "蓉城": "成都", "春城": "昆明",
    "泉城": "济南", "江城": "武汉", "星城": "长沙", "冰城": "哈尔滨",
    "山城": "重庆", "日光城": "拉萨",
}

POLLUTANTS = ["PM2.5", "PM10", "O3", "SO2", "NO2", "CO", "AQI"]

# 注意：Python re 的 \b 在中英文交界处失效（中文也算 \w），
# 因此 CO/AQI 这类纯字母词用 ASCII 边界断言 (?<![a-zA-Z])…(?![a-zA-Z0-9])。
POLLUTANT_PATTERNS = [
    ("PM2.5", [r"pm2\.?5", r"细颗粒物"]),
    ("PM10", [r"pm10", r"pm\s*10", r"可吸入颗粒物"]),
    ("O3", [r"o₃", r"o3", r"臭氧"]),
    ("SO2", [r"so₂", r"so2", r"二氧化硫"]),
    ("NO2", [r"no₂", r"no2", r"二氧化氮"]),
    ("AQI", [r"(?<![a-zA-Z])aqi(?![a-zA-Z0-9])", r"空气质量指数", r"空气质量"]),
    ("CO", [r"(?<![a-zA-Z])co(?![a-zA-Z0-9])", r"一氧化碳"]),
]

INTENT_PATTERNS = [
    # 分组切片优先于普通排名：「南北各自超标最严重的3个省份」不是全量排名
    ("group_top", [r"各自.{0,14}(最|前\s*\d+|top)", r"(南方|北方|南北方).{0,10}(省|省份).{0,14}(最|前\s*\d+)",
                   r"各省.{0,8}(最|前\s*\d+)"]),
    # 季节之最：「臭氧污染最严重的是哪个季节」
    ("season_worst", [r"哪个季节", r"什么季节"]),
    ("correlation", [r"相关", r"吻合", r"同步", r"联动", r"走势一致"]),
    ("source", [r"来源", r"哪来", r"来自.{0,3}哪", r"成因", r"为什么.{0,8}(严重|高|污染|差)", r"什么导致"]),
    ("monthly_trend", [r"月度", r"变化曲线", r"趋势", r"折线"]),
    ("exceed_share", [r"占比", r"比例", r"分布"]),
    ("city_season", [r"春季|夏季|秋季|冬季", r"总天数", r"多少天"]),
    ("anomaly", [r"异常", r"跳变", r"突发", r"告警"]),
    ("compare", [r"对比", r"比较", r"南北方", r"差异"]),
    # 省份排名：「CO浓度最低的5个省份」「新疆和河北哪个省份更严重」
    ("province_rank", [r"省份.{0,6}(排名|最|前\s*\d+|浓度)", r"哪个省", r"哪几个省",
                       r"省.{0,4}排名"]),
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
    provinces: list = field(default_factory=list)
    year: int = None
    season: str = None
    top_n: int = 5
    chart: str = "auto"
    audience: str = "simple"
    dim: str = "季节"
    source: str = "rule"           # rule / llm
    thoughts: list = field(default_factory=list)
    out_of_scope: bool = False     # True=问题超出数据能力范围（诚实拒答，不硬答）
    scope_reason: str = ""

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
        if "city_correlation" in tools:
            return "correlation"
        if "pollution_source" in tools:
            return "source_profile"
        for s in self.steps:
            if s.tool == "aggregate":
                gb, m = s.args.get("group_by"), s.args.get("metric")
                if gb == "province" and s.args.get("split_by") == "north_south":
                    return "group_top"
                if gb == "province" and m == "exceed_days":
                    return "exceed_share"
                if gb == "province" and m == "mean":
                    return "province_rank"
                if gb == "season" and m == "exceed_days":
                    return "season_worst"
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
            "pollutant": "PM2.5 / PM10 / O3 / SO2 / NO2 / CO / AQI",
            "cities": "可选，限定城市列表，如 ['北京','上海']",
            "provinces": "可选，限定省份列表（如 ['新疆','河北']），仅 group_by=province 时有效",
            "year": "可选，年份，如 2024",
            "top_n": "可选，返回前 N 行，默认 5；需要全量排名时用 500（数据集共 375 个城市）",
            "order": "可选，asc=浓度最低在前 | desc=最高在前（默认）。问「最好/最清洁/最低」用 asc，问「最高/最重/最差」用 desc",
            "split_by": "可选，仅 group_by=province 时有效：north_south=把省份按南北方分两组，各组内分别取 top_n（回答「南北各自最严重的N个省份」）",
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
    "city_correlation": {
        "desc": "两个城市某污染物的日浓度 Pearson 相关性，判断污染是否同步、是否受同一天气过程/区域传输影响。",
        "args": {"cities": "两个城市，如 ['北京','上海']", "pollutant": "污染物", "year": "可选，年份"},
    },
    "pollution_source": {
        "desc": "单个城市的污染来源特征分析（扬尘/交通燃烧/采暖/二次生成等浓度代理指标，非源谱解析）。",
        "args": {"city": "城市名", "pollutant": "污染物", "year": "可选，年份"},
    },
    "check_standard": {
        "desc": "查询限值口径（六项污染物为 GB 3095-2012 二级标准，AQI 为 HJ 633-2012），用于超标研判。",
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
- 来源：中国环境监测总站公开数据（CN-AIR），覆盖全国 375 个城市，逐日六项污染物浓度 + AQI；
- 污染物与判定口径：PM2.5(24h 平均)、PM10(24h)、SO2(24h)、NO2(24h)、O3(日最大8h 平均)、CO(24h)、AQI(日值，>100 为超标日)；
- 可用年份：{years}（未提年份用默认年份 {default_year}）。

能力边界（重要）：
- 数据里只有上述污染物浓度与 AQI，没有气温、降水、湿度、风速、天气现象等气象要素；
- 只覆盖中国城市，没有其他国家/地区；
- 若用户问题超出该范围（问气象要素、外国、闲聊问候、与空气质量数据分析无关的请求），
  输出 {{"out_of_scope": true, "reason": "一句话说明为什么无法回答，并提示本智能体能分析什么"}}，不要强行规划 steps；
- 注意区分：雾霾、沙尘、空气、污染、蓝天指数、呼吸健康相关的问题属于本智能体范围（对应污染物分析）。

可用工具（function calling 风格）：
{tools}

规划要点：
1. 多部分问题（一个问句里有多个子问题）必须为每个可回答的子问题都安排对应的 aggregate/trend/…步骤，全部结果都会保留并展示，不要只答最后一个；
2. 「最高/最重/最差/污染最严重」用 order=desc（默认）；「最低/最好/最清洁/空气质量最优」用 order=asc；
3. 需要全部城市参与排名、或「某城市排全国第几」这类定位问题时，用 top_n=500 拿全量表；
4. 「达标城市有几个」= 全量 top_n=500 后由结果表的「是否超标」列统计；
5. 「南北各自 top N 个省份」「北方/南方各自最严重的 N 个省」= aggregate(group_by=province, metric=exceed_days, split_by=north_south, top_n=N)，引擎会输出北方、南方各自的组内 TOP N 两张切片，不要让用户自己从 31 省全量表归组；
6. 问「两个城市污染是否相关/同步/是不是受同一天气过程影响」用 city_correlation；问「某城市污染哪来的/主要来源/为什么这么严重」用 pollution_source（代理指标推断，非源谱）；
7. 城市名写常见名即可（如 北京、和田、首都→北京），系统会自动校正为标准名；
8. 若用户问的年份不在「可用年份」里，在 thoughts 第一条明确说明「数据集无 XXXX 年，已改用 YYYY 年」，再用可用年份规划——不要静默替换。

输出要求：
1. 只输出一个 JSON 对象，不要输出任何其他文字；
2. 格式：{{"thoughts": ["规划思路…"], "pollutant": "PM2.5", "steps": [{{"tool": "工具名", "args": {{...}}, "thought": "这一步为什么这样做"}}, ...]}}；out_of_scope 时输出 {{"out_of_scope": true, "reason": "…"}}；
3. steps 中第一个工具必须是 load_data，最后一个建议是 validate；plot_chart 可选，放在产出统计结果之后；
4. 每步都要有 thought 字段，写清楚这一步在解决用户问题的哪个部分。
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
    for alias, std in CITY_ALIASES.items():   # 别称优先（「首都」不在标准名表里）
        if alias in text and std not in found:
            found.append(std)
    for c in CITY_NAMES:  # 长的优先，避免「北京」截断「北京市」
        if c in text and c not in found:
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
    if m:
        return int(m.group(1))
    # 「达标城市有几个」「排全国第几」需要全量表才能回答 → 500（375 城全量）
    if re.search(r"达标|二级标准|第几|排名|多少个|多少座|一共", text):
        return 500
    return 5


def _find_season(text: str):
    for w, s in SEASON_WORDS.items():
        if w + "季" in text:
            return s
    return None


def _find_provinces(text: str):
    """识别问题中提到的省份（支持「新疆和河北哪个省份更严重」式对比）。"""
    found = []
    for p in PROVINCE_NAMES:
        if p in text and p not in found:
            found.append(p)
    return found


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
    """把 LLM/用户写的城市名校正为数据集标准名（北京→北京市、首都→北京、和田→和田地区）。"""
    if not name:
        return None
    n = str(name).strip()
    if n in CITY_TO_PROVINCE:
        return n
    if n in CITY_ALIASES:
        return CITY_ALIASES[n]
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


def _valid_top_n(v, default=5):
    try:
        v = int(v)
    except (TypeError, ValueError):
        return default
    return max(1, min(500, v))


# 规则兜底用的域外问题关键词（LLM 不可用时的最后防线）
_OUT_OF_SCOPE_RE = re.compile(
    r"气温|温度|降雨|降水量|湿度|风速|风力|大风|台风|天气(?!污染)|下雪|降雪|日照|日出|日落|"
    r"你好|您好|在吗|谢谢|再见|hello|hi\b|help me|你是谁|自我介绍", re.I)


# ---------------- 规则规划器：意图 → 同构 steps ----------------

def build_steps(intent, pollutant, cities, year, season, top_n, dim, chart, order="desc",
                memory=None, season_metric="exceed_days", provinces=None):
    """按识别出的槽位生成标准工具调用链（与 LLM 路径同构）。"""
    S = Step
    load = S("load_data", {},
             "先分片读取本地数据集并清洗，确保后续统计基于真实监测数据")
    ck = S("check_standard", {"pollutant": pollutant},
           f"对照限值口径，明确 {pollutant} 的判定标准")
    val = S("validate", {}, "对统计结果做合理性校验，降低规划错误风险")
    plot = lambda c: S("plot_chart", {"chart": c}, f"将统计结果可视化为{c}图")

    if intent == "group_top":
        return [load,
                S("aggregate", {"group_by": "province", "metric": "exceed_days",
                                "split_by": "north_south", "pollutant": pollutant,
                                "year": year, "top_n": top_n, "order": order},
                  f"按南北方分组切片，取各组内 {pollutant} 超标最严重的前 {top_n} 个省份"),
                ck, plot(chart or "bar"), val]
    if intent == "province_rank":
        return [load,
                S("aggregate", {"group_by": "province", "metric": "mean",
                                "pollutant": pollutant, "provinces": provinces,
                                "year": year, "top_n": top_n, "order": order},
                  f"按省份分组计算 {pollutant} 平均浓度排名"
                  f"（限定省份: {provinces}）" if provinces else
                  f"按省份分组计算 {pollutant} 平均浓度排名"),
                ck, plot(chart or "bar"), val]
    if intent == "season_worst":
        metric = season_metric
        return [load,
                S("aggregate", {"group_by": "season", "metric": metric,
                                "pollutant": pollutant, "year": year},
                  f"按季节统计 {pollutant} {'平均浓度' if metric == 'mean' else '超标天数'}，找出最严重的季节"),
                ck, plot(chart or "bar"), val]
    if intent == "correlation":
        return [load,
                S("city_correlation", {"cities": cities, "pollutant": pollutant,
                                       "year": year},
                  f"计算 {('、'.join(cities)) if cities else '指定城市'} 的 {pollutant} "
                  f"日浓度相关性，判断污染是否同步"),
                ck, plot(chart or "bar"), val]
    if intent == "source":
        city = cities[0] if cities else (memory.last_cities[0] if memory and memory.last_cities else "北京")
        return [load,
                S("pollution_source", {"city": city, "pollutant": pollutant,
                                       "year": year},
                  f"从扬尘/交通燃烧/采暖/二次生成等代理指标分析 {city} 的 {pollutant} 来源特征"),
                ck, plot(chart or "bar"), val]
    if intent == "annual_rank":
        return [load,
                S("aggregate", {"group_by": "city", "metric": "mean", "pollutant": pollutant,
                                "year": year, "top_n": top_n, "order": order},
                  f"按城市分组计算 {pollutant} 年均浓度，取{'浓度最低' if order == 'asc' else '浓度最高'}前 {top_n} 个"),
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
                            "year": year, "top_n": top_n, "order": order},
              f"按城市分组计算 {pollutant} 年均浓度，取前 {top_n} 个"),
            ck, plot(chart or "bar"), val]


def rule_plan(question: str, memory=None, default_year=None, available_years=None) -> TaskPlan:
    """本地规则规划器（无网无 Key 兜底）。"""
    p = TaskPlan(question=question, source="rule")

    # 域外问题兜底：LLM 不可用时至少要诚实拒答，不能硬答空气质量
    if _OUT_OF_SCOPE_RE.search(question):
        p.out_of_scope = True
        p.scope_reason = ("该问题涉及气象要素或日常对话，不在本智能体的数据范围内"
                          "（数据集仅含六项污染物浓度与 AQI 的日均时序）。")
        p.thoughts.append(f"[规划] 规则模式判定问题超出数据能力范围，已拒答")
        return p

    p.pollutant = _find_pollutant(question)
    intent = _find_intent(question)
    p.year = _find_year(question, default_year)
    p.top_n = _find_top_n(question)
    p.season = _find_season(question)
    p.audience = _find_audience(question)
    p.cities = _find_cities(question)
    p.provinces = _find_provinces(question)
    p.dim = "north_south" if ("南方" in question or "北方" in question) else "season"
    order = "asc" if re.search(r"最低|最好|最优|最清洁|空气好", question) else "desc"

    # 年份越界：数据集不含该年份时改用默认年份，并在思考中明确说明（不静默替换）
    if p.year is not None and available_years and p.year not in available_years:
        ys = "、".join(str(y) for y in available_years)
        p.thoughts.append(f"[规划] 数据集无 {p.year} 年（仅覆盖 {ys} 年），已改用 {default_year} 年")
        p.year = default_year

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
    # 「哪个季节最严重」：问平均浓度用 mean，否则按超标天数
    season_metric = "mean" if re.search(r"平均", question) else "exceed_days"
    # 相关性/来源分析至少分别需要 2 个城市 / 1 个城市，槽位不足时给出明确降级
    if intent == "correlation" and len(p.cities) < 2:
        p.thoughts.append(f"[规划] 相关性分析需要两个城市（识别到 {p.cities or '无'}），"
                          f"将在执行步给出提示")
    if intent == "source" and not p.cities:
        p.cities = list(memory.last_cities) if memory and memory.last_cities else ["北京"]
        p.thoughts.append(f"[规划] 来源分析未指明城市，使用: {p.cities}")

    chart = _find_chart(question, intent)
    p.chart = chart
    p.steps = build_steps(intent, p.pollutant, p.cities, p.year, p.season,
                          p.top_n, p.dim, chart, order, memory=memory,
                          season_metric=season_metric, provinces=p.provinces)
    p.thoughts.append(f"[规划] 识别意图={intent}, 污染物={p.pollutant}, 方向={order}")
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
        if a.get("split_by") not in (None, "north_south"):
            a.pop("split_by")
        if a.get("split_by") and a["group_by"] != "province":
            a.pop("split_by")   # split_by 只在按省份分组时有意义
    if "order" in a:
        if a.get("order") not in ("asc", "desc"):
            a["order"] = "desc"
    if "pollutant" in a:
        a["pollutant"] = _valid_pollutant(a.get("pollutant")) or "PM2.5"
    if "year" in a:
        a["year"] = _valid_year(a.get("year"))
        if a["year"] is None:
            a.pop("year")
    if "top_n" in a:
        try:
            a["top_n"] = max(1, min(500, int(a["top_n"])))
        except (TypeError, ValueError):
            a.pop("top_n")
    if "cities" in a:
        a["cities"] = normalize_cities(a.get("cities") if isinstance(a.get("cities"), list) else [])
        if not a["cities"]:
            a.pop("cities")
    if "provinces" in a:
        provs = [str(x).strip() for x in (a.get("provinces") or [])
                 if str(x).strip() in PROVINCE_NAMES] if isinstance(a.get("provinces"), list) else []
        if provs:
            a["provinces"] = provs
        else:
            a.pop("provinces")
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


def _step_key(s):
    return (s.tool, tuple(sorted((k, str(v)) for k, v in s.args.items())))


def _prune_steps(steps, max_steps=12):
    """剪枝 LLM 偶发冗余步骤（不足⑤：偶发多规划一步，白跑约 5 秒）。

    规则：
    1. 相邻的完全重复步骤（同工具同参数）合并为一个；
    2. 非相邻的完全重复计算步骤（aggregate/trend/city_correlation/pollution_source/
       season_exceed/anomaly/check_standard/load_data）只保留第一次——
       同样的参数再算一遍只会得到同一张表，纯属浪费；
       plot_chart 豁免：多部分问题里每次画的是「当时的最新结果」，各自有意义；
    3. 总步数超过 max_steps 时，优先丢弃多余的 check_standard/plot_chart/validate。
    """
    if len(steps) <= 2:
        return steps
    dedup_tools = {"load_data", "check_standard", "aggregate", "trend",
                   "season_exceed", "anomaly", "city_correlation", "pollution_source"}
    seen, out = set(), []
    for s in steps:
        key = _step_key(s)
        adjacent_dup = out and _step_key(out[-1]) == key
        repeat_calc = key in seen and s.tool in dedup_tools
        if adjacent_dup or repeat_calc:
            continue
        seen.add(key)
        out.append(s)
    if len(out) > max_steps:
        optional = {"check_standard", "plot_chart", "validate"}
        keep_head = [s for s in out if s.tool not in optional][:max_steps]
        # 保留最后一步 validate（若有），保证链路完整性
        tail_val = [s for s in out if s.tool == "validate"][-1:]
        if tail_val and (not keep_head or keep_head[-1].tool != "validate"):
            keep_head = keep_head[:max_steps - 1] + tail_val
        out = keep_head
    return out


def _sanitize_llm_steps(data, question):
    """校验 LLM 输出的调用链：工具白名单 + 参数清洗 + 冗余剪枝 + 结构补齐。"""
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
    n_before = len(steps)
    steps = _prune_steps(steps)
    pruned = n_before - len(steps)
    # load_data 必须位于链首：已存在则移到最前，缺失则插入
    load_idx = next((i for i, s in enumerate(steps) if s.tool == "load_data"), None)
    if load_idx is None:
        steps.insert(0, Step("load_data", {},
                             "先分片读取本地数据集并清洗，确保统计基于真实监测数据"))
    elif load_idx > 0:
        steps.insert(0, steps.pop(load_idx))
    if pruned:
        steps[0].thought = (steps[0].thought +
                            f"（已自动剪枝 {pruned} 个冗余步骤）").strip()
    return steps


def _memory_context(memory) -> str:
    """把多轮记忆整理为注入 LLM 的上下文文本（LLM 路径的指代继承）。

    注意保守注入：只有本问出现指代词或明显缺少主语时才继承；
    本问已明确范围（出现「全国/各省」或具体城市名）时以本问为准，
    避免把上一轮的城市错误地套到独立的新问题上。
    """
    if memory is None:
        return ""
    parts = []
    if memory.history:
        parts.append(f"上一轮问题：{memory.history[-1]['question']}")
    if memory.last_cities:
        parts.append(f"上一轮分析城市：{'、'.join(memory.last_cities)}")
    if memory.last_pollutant:
        parts.append(f"上一轮污染物：{memory.last_pollutant}")
    if memory.last_year:
        parts.append(f"上一轮年份：{memory.last_year}")
    if not parts:
        return ""
    return ("\n\n[对话上下文] 仅当本问包含「那/这/上述/刚才/继续/呢」等指代词、"
            "或明显缺少主语/范围时，才继承下列上下文补全省略要素；"
            "若本问已明确范围（出现「全国」「各省」或具体城市名），一律以本问为准，不得继承：\n"
            + "\n".join(parts))


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
            {"role": "user", "content": question + _memory_context(memory)},
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

    # 能力边界：LLM 判定问题超出数据范围 → 诚实拒答，不硬答
    if plan_data.get("out_of_scope"):
        p.out_of_scope = True
        p.scope_reason = str(plan_data.get("reason", "")).strip()[:300]
        p.thoughts.append(f"[规划] LLM 判定问题超出数据能力范围：{p.scope_reason}")
        return p

    p.pollutant = _valid_pollutant(plan_data.get("pollutant")) or "PM2.5"
    p.cities = normalize_cities(plan_data.get("cities") if isinstance(plan_data.get("cities"), list) else [])
    p.provinces = [str(x).strip() for x in (plan_data.get("provinces") or [])
                   if str(x).strip() in PROVINCE_NAMES] \
        if isinstance(plan_data.get("provinces"), list) else []
    p.year = _valid_year(plan_data.get("year"), default_year)
    p.top_n = _valid_top_n(plan_data.get("top_n"), 5)
    if plan_data.get("season") in ("春季", "夏季", "秋季", "冬季", "全年"):
        p.season = plan_data["season"]
    if plan_data.get("chart") in ("auto", "line", "bar", "pie"):
        p.chart = plan_data["chart"]
    if plan_data.get("dim") in ("season", "north_south"):
        p.dim = plan_data["dim"]
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
            p = rule_plan(question, memory, default_year, available_years)
            p.thoughts.append(f"[规划] LLM 规划失败({type(e).__name__}: {e})，"
                              f"已回退本地规则规划器")
            return p
    return rule_plan(question, memory, default_year, available_years)
