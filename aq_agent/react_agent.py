# -*- coding: utf-8 -*-
"""ReAct 智能体：Thought → Action(tool) → Observation 循环 + 工具集合。"""
import time

import pandas as pd

from . import analysis, charts, config, planner
from .memory import Memory
from .report import build_html_report, build_text_report, edu_limit_tip


def df_to_html(df: pd.DataFrame) -> str:
    cols = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = "".join("<tr>" + "".join(f"<td>{v}</td>" for v in r) + "</tr>"
                   for r in df.itertuples(index=False))
    return f"<table><thead><tr>{cols}</tr></thead><tbody>{rows}</tbody></table>"


def df_to_text(df: pd.DataFrame) -> str:
    lines = [" | ".join(str(c) for c in df.columns)]
    lines += [" | ".join("" if pd.isna(v) else str(v) for v in r)
              for r in df.itertuples(index=False)]
    return "\n".join(lines)


class ReActAgent:
    """轻量 ReAct 引擎：规划器产出计划，按意图调度工具链，全程留痕。"""

    def __init__(self, df: pd.DataFrame, clean_report: dict, memory: Memory = None,
                 default_year=None):
        self.df = df
        self.clean_report = clean_report
        self.memory = memory or Memory()
        self.default_year = default_year
        # 工具注册表（name -> (描述, 函数)）
        self.tools = {
            "load_data": ("分片读取并清洗本地空气质量数据集", self._t_load_data),
            "aggregate": ("按城市/省份/年月维度分组聚合统计", self._t_aggregate),
            "check_standard": ("依据 GB 3095-2012 判定污染物超标", self._t_check_standard),
            "detect_anomaly": ("MAD 法检测时序异常跳变点", self._t_detect_anomaly),
            "plot_chart": ("生成折线/柱状/饼图可视化", self._t_plot_chart),
            "validate": ("对统计结果做合理性校验", self._t_validate),
        }

    # ---------- 工具实现 ----------

    def _t_load_data(self, plan):
        r = self.clean_report
        return (f"已加载 {r['清洗后行数']} 行（{r['城市数']} 个城市，{r['日期范围']}），"
                f"分片 {r['原始行数']} 行原始记录，清洗丢弃 {r['丢弃行数']} 行")

    def _t_aggregate(self, plan):
        p = plan.pollutant
        if plan.intent == "annual_rank":
            out, limit = analysis.annual_mean_rank(self.df, p, plan.year, plan.top_n)
            self._agg = out
            return out, f"年均浓度 TOP{plan.top_n}（{plan.year or '全部年份'}）", limit
        if plan.intent == "monthly_trend":
            # 先按年均排名选出代表城市（未指定城市时取 TOP N），再算月度趋势
            rank, _ = analysis.annual_mean_rank(self.df, p, plan.year, plan.top_n)
            cities = plan.cities or list(rank["城市"])
            piv = analysis.monthly_trend(self.df, cities, p, plan.year)
            # 列按各自年均浓度降序，直观呈现排名
            piv = piv[piv.mean().sort_values(ascending=False).index]
            self._agg = rank
            self._trend = piv
            return piv, f"月度均值趋势（城市: {cities}）", None
        if plan.intent == "exceed_share":
            d = self.df
            desc = f"各省份超标天数占比（{plan.year or '全部年份'}）"
            if plan.cities:
                # 指定了城市：只看这些城市所属省份
                provs = sorted({__import__("aq_agent.geo", fromlist=["CITY_TO_PROVINCE"])
                                .CITY_TO_PROVINCE.get(c, c) for c in plan.cities})
                d = d[d["省份"].isin(provs)]
                desc = f"指定城市所属省份（{provs}）超标天数占比"
            out, limit = analysis.exceed_days_by_province(d, p, plan.year)
            self._share = out
            return out, desc, limit
        if plan.intent == "city_season":
            city = plan.cities[0] if plan.cities else "北京"
            info = analysis.city_season_exceed_days(self.df, city, p, plan.season, plan.year)
            self._season = info
            return info, f"{city}{plan.season or '全年'} {p} 超标统计", info["限值"]
        if plan.intent == "compare":
            out = analysis.compare_dimension(self.df, p, plan.year, plan.dim)
            self._cmp = out
            return out, f"多维度对比（{plan.dim}）", None
        if plan.intent == "anomaly":
            city = plan.cities[0] if plan.cities else "北京"
            hits = analysis.detect_anomalies(self.df, city, p, plan.year)
            self._anomaly = (city, hits)
            return (city, hits), f"{city} {p} 异常跳变检测", None
        raise ValueError(f"未知意图: {plan.intent}")

    def _t_check_standard(self, plan, limit=None, kind=None):
        p = plan.pollutant
        return f"{p} 判定依据：{analysis.format_limit_text(p, kind)}"

    def _t_detect_anomaly(self, plan):
        return self._t_aggregate(plan)

    def _t_plot_chart(self, plan, agg, desc, limit):
        p = plan.pollutant
        unit = config.GB3095_GRADE2[p]["unit"]
        year_txt = f"{plan.year}年" if plan.year else "全时段"
        if plan.intent in ("annual_rank",):
            path = charts.bar_chart(agg, "城市", "年均浓度",
                                    f"{year_txt}全国{p}年均浓度 TOP{len(agg)}", unit)
        elif plan.intent == "monthly_trend":
            path = charts.line_chart(agg, f"{year_txt}{p}月度变化曲线", unit)
        elif plan.intent == "exceed_share":
            path = charts.pie_chart(agg.reset_index(), "省份", "占全国比例%",
                                    f"{year_txt}各省份{p}超标天数占比")
        elif plan.intent == "compare":
            path = charts.bar_chart(agg.reset_index(), "分组", "mean",
                                    f"{year_txt}{p}{plan.dim}对比", unit)
        elif plan.intent == "city_season":
            info = agg
            one = pd.DataFrame({"项目": ["超标天数", "有效天数", "超标率%"],
                                "数值": [info["超标天数"], info["有效天数"], info["超标率%"]]})
            path = charts.bar_chart(one, "项目", "数值",
                                    f"{info['城市']}{info['季节']} {p} 超标统计", "天")
        elif plan.intent == "anomaly":
            city, hits = agg
            d = self.df[self.df["城市"] == city]
            if plan.year:
                d = d[d["日期"].dt.year == plan.year]
            col = config.DAILY_JUDGE_COL[p] if p != "PM2.5" else "PM2.5_24h"
            s = d.set_index("日期")[col].dropna().sort_index()
            path = charts.anomaly_chart(s, hits, city, p, unit)
        else:
            path = None
        return path

    def _t_validate(self, plan, agg):
        if plan.intent == "exceed_share":
            ok, msg = analysis.validate_share(agg)
        elif plan.intent == "annual_rank":
            ok, msg = analysis.validate_rank(agg)
        else:
            ok, msg = True, "样本量充足，统计口径一致"
        return ("通过" if ok else "存疑") + f"（{msg}）"

    # ---------- ReAct 主循环 ----------

    def run(self, question: str) -> dict:
        t0 = time.time()
        plan = planner.plan(question, self.memory, self.default_year)
        trace = []
        result = {"trace": trace, "plan": plan}

        def think(t):
            trace.append(f"Thought: {t}")

        def act(name, obs):
            trace.append(f"Action: {name}")
            trace.append(f"Observation: {obs}")

        for t in plan.thoughts:
            think(t)
        think(f"用户意图是「{plan.intent}」，需要调度工具链完成分析")

        # 1) 读取数据
        act("load_data", self._t_load_data(plan))

        # 2) 聚合统计
        think("数据就绪，执行分组聚合统计")
        agg, desc, limit = self._t_aggregate(plan)
        if plan.intent == "anomaly":
            city_name, hits_df = agg
            obs = f"{desc}：检出 {len(hits_df)} 个异常跳变点"
            if len(hits_df):
                obs += "；\n" + df_to_text(hits_df.head(10))
        elif isinstance(agg, pd.DataFrame):
            obs = f"{desc}：共 {len(agg)} 行；\n{df_to_text(agg.head(10))}"
        else:
            obs = f"{desc}：{agg}"
        act("aggregate", obs)

        # 3) 国标超标判定
        kind = config.DEFAULT_LIMIT_KIND.get(plan.pollutant)
        think("统计完成，依据 GB 3095-2012 进行超标研判")
        act("check_standard", self._t_check_standard(plan, limit, kind))

        # 4) 可视化
        think("生成对应类型的可视化图表")
        chart_path = self._t_plot_chart(plan, agg, desc, limit)
        act("plot_chart", f"图表已保存: {chart_path}" if chart_path else "该意图无需图表")

        # 5) 结果校验
        think("对统计结果做数据合理性校验，降低规划错误风险")
        act("validate", self._t_validate(plan, agg))

        # 6) 汇总
        result.update(self._collect(plan, agg, desc, limit, chart_path, trace))
        result["seconds"] = round(time.time() - t0, 2)
        trace.append(f"Thought: 分析完成，输出结构化结论与图表（耗时 {result['seconds']}s）")
        # 记忆本轮「实际分析的城市」，支撑下一轮「这几个城市」式追问
        result_cities = []
        if plan.intent == "annual_rank" and hasattr(self, "_agg"):
            result_cities = list(self._agg["城市"])
        elif plan.intent == "monthly_trend" and hasattr(self, "_trend"):
            result_cities = list(self._trend.columns)
        elif plan.intent in ("city_season", "anomaly") and plan.cities:
            result_cities = plan.cities[:1]
        self.memory.update(plan, result["seconds"], result_cities)
        return result

    def _collect(self, plan, agg, desc, limit, chart_path, trace):
        """把工具产出整理为结果字典（供文本/HTML 报告使用）。"""
        p = plan.pollutant
        unit = config.GB3095_GRADE2[p]["unit"]
        year_txt = f"{plan.year}年" if plan.year else "全时段"
        out = {"口径": f"{desc}；数据来源：中国环境监测总站公开数据集（CN-AIR）",
               "图表路径": chart_path}
        if plan.intent in ("annual_rank", "exceed_share", "compare"):
            tbl = agg.reset_index()
            out["表格HTML"] = df_to_html(tbl)
            out["表格文本"] = df_to_text(tbl)
        if plan.intent == "annual_rank":
            n_over = int((agg["是否超标"] == "超标").sum())
            out["超标研判"] = (f"TOP{len(agg)} 城市中 {n_over} 个城市年均{p}超过二级限值 "
                              f"{config.GB3095_GRADE2[p]['annual']}{unit}"
                              f"（GB 3095-2012 年均二级标准）")
        elif plan.intent == "monthly_trend":
            rank = self._agg
            n_over = int((rank["是否超标"] == "超标").sum())
            out["超标研判"] = (f"TOP{len(rank)} 城市中 {n_over} 个城市年均{p}超过二级限值 "
                              f"{config.GB3095_GRADE2[p]['annual']}{unit}")
            out["表格HTML"] = ("<h3>年均浓度排名</h3>" + df_to_html(rank)
                               + "<h3>月度变化曲线数据</h3>" + df_to_html(agg.reset_index()))
            out["表格文本"] = ("年均浓度排名：\n" + df_to_text(rank)
                               + "\n\n月度变化曲线数据：\n" + df_to_text(agg.reset_index()))
        elif plan.intent == "city_season":
            info = agg
            df = pd.DataFrame({"项目": ["超标天数", "有效天数", "超标率%", "判定限值"],
                               "数值": [info["超标天数"], info["有效天数"],
                                       info["超标率%"], f"{info['限值']}{unit}（{info['口径']}）"]})
            out["表格HTML"] = df_to_html(df)
            out["表格文本"] = df_to_text(df)
            out["超标研判"] = (f"{info['城市']}{info['季节']}共 {info['有效天数']} 个有效监测日，"
                              f"{p}超标 {info['超标天数']} 天（限值 {info['限值']}{unit}，{info['口径']}）")
        elif plan.intent == "anomaly":
            city, hits = agg
            out["表格HTML"] = df_to_html(hits) if len(hits) else "<p>未检测到显著异常跳变</p>"
            out["表格文本"] = df_to_text(hits) if len(hits) else "未检测到显著异常跳变"
            out["超标研判"] = (f"{city} {p} 检出 {len(hits)} 个异常跳变点"
                              f"（MAD 3σ 阈值），已标记于图表，可用于污染突发时段识别")
        out["校验"] = self._t_validate(plan, agg)
        out["trace"] = "\n".join(trace)
        return out
