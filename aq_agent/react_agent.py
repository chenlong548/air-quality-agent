# -*- coding: utf-8 -*-
"""ReAct 智能体：按规划器的「工具调用链」逐步执行，全程事件化。

每一步（Thought → Action → Observation）都通过 on_event 回调实时发出，
网页版用 SSE 推送给前端逐条渲染；命令行版不传回调，事件自动收集为 trace。
"""
import os
import time

import pandas as pd

from . import analysis, charts, config, planner
from .memory import Memory
from .report import build_html_report, build_text_report


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


# 聚合结果类型 → 图表类型 自动映射
_AUTO_CHART = {
    "annual_rank": "bar", "city_exceed": "bar", "province_mean": "bar",
    "province_share": "pie", "group_share": "pie", "compare": "bar", "trend": "line",
    "season_exceed": "bar", "anomaly": "line",
}


class ReActAgent:
    """轻量 ReAct 引擎：按 steps 调度工具链，每步一个事件，全程留痕。"""

    def __init__(self, df: pd.DataFrame, clean_report: dict, memory: Memory = None,
                 default_year=None):
        self.df = df
        self.clean_report = clean_report
        self.memory = memory or Memory()
        self.default_year = default_year
        self._on_event = None
        self._trace_lines = []
        self._ctx = {}

    # ---------- 事件与留痕 ----------

    def _emit(self, kind: str, **kw):
        """发一个事件：通知回调（流式推送用），同时写入 trace（终端报告用）。"""
        ev = {"kind": kind, "ts": round(time.time(), 3)}
        ev.update(kw)
        if self._on_event:
            try:
                self._on_event(ev)
            except Exception:
                pass
        self._trace_event(ev)

    def _trace_event(self, ev):
        """把事件同步成终端 trace 文本（保持旧版 CLI 输出风格）。"""
        k = ev["kind"]
        if k == "plan":
            self._trace_lines.append(
                f"[规划] 规划来源={ev.get('source')}，共 {ev.get('total')} 步工具调用链")
            for i, s in enumerate(ev.get("steps", []), 1):
                self._trace_lines.append(f"  {i}. {s['tool']}  {s.get('thought', '')}")
        elif k == "thought":
            self._trace_lines.append(f"Thought: {ev.get('text', '')}")
        elif k == "action":
            args = ev.get("args") or {}
            arg_txt = ", ".join(f"{k}={v}" for k, v in args.items()) or "无参数"
            self._trace_lines.append(f"Action: {ev.get('tool')}({arg_txt})")
        elif k == "observation":
            self._trace_lines.append(f"Observation: {ev.get('text', '')}")
        elif k == "chart":
            self._trace_lines.append(f"[图表] {ev.get('name')}")
        elif k == "error":
            self._trace_lines.append(f"[错误] {ev.get('text', '')}")

    # ---------- 主循环 ----------

    def run(self, question: str, on_event=None) -> dict:
        """执行一次分析。on_event 为 None 时退化为收集 trace（CLI 模式）。"""
        self._on_event = on_event
        self._trace_lines = []
        # results 保存全部聚合结果块（多部分问题的每个子问题各一块，互不覆盖）
        self._ctx = {"plan": None, "results": []}
        t0 = time.time()

        # 1) 规划：生成本问题专属的工具调用链
        plan = planner.plan(question, self.memory, self.default_year,
                            available_years=self._years())
        self._ctx["plan"] = plan
        for t in plan.thoughts:
            self._emit("thought", text=t, step=0, total=len(plan.steps))
        self._emit("plan", source=plan.source, total=len(plan.steps),
                   steps=plan.step_summary(), pollutant=plan.pollutant)

        # 1.5) 能力边界：问题超出数据范围 → 诚实说明，不硬答
        if plan.out_of_scope:
            seconds = round(time.time() - t0, 2)
            result = self._collect_out_of_scope(plan, seconds)
            result["plan"] = plan
            self._emit("done", seconds=seconds, intent="out_of_scope",
                       source=plan.source, concl=result.get("_concl", ""))
            result["seconds"] = seconds
            result["trace"] = "\n".join(self._trace_lines)
            self.memory.note_out_of_scope(plan)
            return result

        # 2) 按链逐步执行
        total = len(plan.steps)
        for i, step in enumerate(plan.steps, 1):
            self._emit("thought", text=step.thought, step=i, total=total)
            self._emit("action", tool=step.tool, args=step.args, step=i, total=total)
            ok, obs, extra = self._dispatch(step.tool, step.args, plan)
            self._emit("observation", text=obs, ok=ok, step=i, total=total)
            if not ok:
                # 单步失败不中断：记录错误，继续后续步骤
                self._emit("error", text=f"步骤 {i}（{step.tool}）失败: {obs}", step=i)

        # 3) 汇总
        seconds = round(time.time() - t0, 2)
        result = self._collect(plan, seconds)
        result["plan"] = plan
        if result.get("表格HTML"):
            self._emit("table", html=result["表格HTML"])
        self._emit("done", seconds=seconds, intent=plan.intent, source=plan.source,
                   concl=result.get("_concl", ""))
        result["seconds"] = seconds
        self._trace_lines.append(
            f"Thought: 分析完成，输出结构化结论与图表（耗时 {seconds}s）")
        result["trace"] = "\n".join(self._trace_lines)

        # 4) 记忆本轮实际分析的城市，支撑下一轮「这几个城市」式追问
        self.memory.update(plan, seconds, self._result_cities(plan))
        return result

    def _years(self):
        try:
            from .data_layer import discover_years
            return discover_years()
        except Exception:
            return [self.default_year] if self.default_year else []

    def _check_year(self, year):
        """年份越界拦截：数据集不含该年份时给出明确指引（返回 None 表示通过）。"""
        if year is None:
            return None
        years = self._years()
        if years and year not in years:
            ys = "、".join(str(y) for y in years)
            return f"数据集仅覆盖 {ys} 年，不包含 {year} 年。请改问 {ys} 年的数据"
        return None

    def _save_block(self, kind, table, desc, limit, pollutant, year, obs, cities=None):
        """把一次聚合结果作为独立块保存（多部分问题的每个子问题各一块）。"""
        self._ctx.update(table=table, table_kind=kind, desc=desc, limit=limit,
                         pollutant=pollutant, year=year)
        self._ctx["results"].append(
            {"kind": kind, "table": table, "desc": desc, "limit": limit,
             "pollutant": pollutant, "year": year, "obs": obs,
             "cities": cities or [],
             "order": self._ctx.get("_order", "desc")})

    def _result_cities(self, plan):
        tbl = self._ctx.get("table")
        kind = self._ctx.get("table_kind")
        if kind in ("annual_rank", "city_exceed") and tbl is not None and "城市" in tbl.columns:
            return tbl["城市"].tolist()
        if kind == "trend" and tbl is not None:
            return [str(c) for c in tbl.columns]
        if kind in ("season_exceed", "anomaly"):
            city = self._ctx.get("city")
            return [city] if city else []
        return list(plan.cities)

    # ---------- 工具分派 ----------

    def _dispatch(self, tool, args, plan):
        """执行一个工具，返回 (ok, observation_text, extra)。"""
        fn = {
            "load_data": self._tool_load_data,
            "aggregate": self._tool_aggregate,
            "trend": self._tool_trend,
            "season_exceed": self._tool_season_exceed,
            "anomaly": self._tool_anomaly,
            "check_standard": self._tool_check_standard,
            "plot_chart": self._tool_plot_chart,
            "validate": self._tool_validate,
        }.get(tool)
        if fn is None:
            return False, f"未知工具: {tool}", None
        try:
            return fn(args, plan)
        except Exception as e:
            return False, f"工具 {tool} 执行失败: {type(e).__name__}: {e}", None

    def _tool_load_data(self, args, plan):
        r = self.clean_report
        return True, (f"已加载 {r['清洗后行数']} 行（{r['城市数']} 个城市，{r['日期范围']}），"
                      f"分片 {r['原始行数']} 行原始记录，清洗丢弃 {r['丢弃行数']} 行"), None

    def _tool_aggregate(self, args, plan):
        gb = args.get("group_by", "city")
        metric = args.get("metric", "mean")
        p = planner._valid_pollutant(args.get("pollutant")) or plan.pollutant
        year = planner._valid_year(args.get("year"), plan.year)
        top_n = args.get("top_n") or plan.top_n
        cities = planner.normalize_cities(args.get("cities")) or list(plan.cities)
        order = args.get("order", "desc")
        self._ctx["_order"] = order
        asc = order == "asc"
        year_txt = f"{year}年" if year else "全时段"

        err = self._check_year(year)
        if err:
            return False, err, None

        d = self.df
        if cities:
            d = d[d["城市"].isin(cities)]
        if year:
            d = d[d["日期"].dt.year == year]
        if d.empty:
            hint = f"（年份={year or '全部'}，城市={cities or '全国'}，污染物={p}）"
            return False, f"过滤后无任何数据行 {hint}。请检查城市名/年份是否在数据范围内", None

        dir_txt = "最低在前" if asc else "最高在前"
        if gb == "city" and metric == "mean":
            out, limit = analysis.annual_mean_rank(d, p, None, top_n, ascending=asc)
            kind = "annual_rank"
            desc = (f"{year_txt}指定城市{p}年均浓度（{dir_txt}）" if cities
                    else f"{year_txt}城市{p}年均浓度 TOP{top_n}（{dir_txt}）")
        elif gb == "city" and metric == "exceed_days":
            out, limit = analysis.city_exceed_days(d, p, top_n, ascending=asc)
            kind, desc = "city_exceed", f"{year_txt}城市{p}超标天数 TOP{top_n}（{dir_txt}）"
        elif gb == "province" and metric == "mean":
            out, limit = analysis.province_mean(d, p, ascending=asc)
            kind, desc = "province_mean", f"{year_txt}各省{p}平均浓度（{dir_txt}）"
        elif gb == "province" and metric == "exceed_days":
            out, limit = analysis.exceed_days_by_province(d, p)
            kind, desc = "province_share", f"{year_txt}各省{p}超标天数占比"
        elif gb == "season" and metric == "mean":
            out, limit = analysis.compare_dimension(d, p, None, "季节"), None
            kind, desc = "compare", f"{year_txt}{p}季节对比"
        elif gb == "north_south" and metric == "mean":
            out, limit = analysis.compare_dimension(d, p, None, "南北方"), None
            kind, desc = "compare", f"{year_txt}{p}南北方对比"
        elif gb == "season" and metric == "exceed_days":
            d = d.assign(_分组=d["日期"].dt.month.map(analysis.month_to_season))
            out, limit = analysis.group_exceed_days(d, p, "_分组", "季节")
            kind, desc = "group_share", f"{year_txt}{p}各季节超标天数占比"
        elif gb == "north_south" and metric == "exceed_days":
            d = d.assign(_分组=d["省份"].apply(
                lambda x: "北方" if x in config.NORTH_PROVINCES else "南方"))
            out, limit = analysis.group_exceed_days(d, p, "_分组", "区域")
            kind, desc = "group_share", f"{year_txt}{p}南北方超标天数占比"
        else:
            return False, f"不支持的聚合组合: group_by={gb}, metric={metric}", None

        obs = f"{desc}：共 {len(out)} 行\n{df_to_text(out.head(10))}"
        self._save_block(kind, out, desc, limit, p, year, obs, cities=cities)
        return True, obs, None

    def _tool_trend(self, args, plan):
        p = planner._valid_pollutant(args.get("pollutant")) or plan.pollutant
        year = planner._valid_year(args.get("year"), plan.year)
        cities = planner.normalize_cities(args.get("cities")) or list(plan.cities)
        err = self._check_year(year)
        if err:
            return False, err, None
        if not cities:
            rank, _ = analysis.annual_mean_rank(self.df, p, year, plan.top_n)
            cities = rank["城市"].tolist()
            if not cities:
                return False, f"{year or '全时段'}无 {p} 有效数据，无法确定趋势城市", None
        piv = analysis.monthly_trend(self.df, cities, p, year)
        if piv.empty:
            return False, (f"月度趋势无数据（城市={cities}，年份={year or '全部'}，"
                           f"污染物={p}）。请检查城市名/年份是否在数据范围内"), None
        piv = piv[piv.mean().sort_values(ascending=False).index]
        desc = f"月度浓度趋势（{p}，城市: {cities}）"
        obs = f"{desc}：{len(cities)} 个城市 × 12 个月\n{df_to_text(piv.head(12))}"
        self._save_block("trend", piv, desc, None, p, year, obs)
        return True, obs, None

    def _tool_season_exceed(self, args, plan):
        p = planner._valid_pollutant(args.get("pollutant")) or plan.pollutant
        year = planner._valid_year(args.get("year"), plan.year)
        city = planner.normalize_city(args.get("city")) or \
            (plan.cities[0] if plan.cities else None) or \
            (self.memory.last_cities[0] if self.memory.last_cities else None) or "北京"
        season = args.get("season") or plan.season
        err = self._check_year(year)
        if err:
            return False, err, None
        info = analysis.city_season_exceed_days(self.df, city, p, season, year)
        if not info["有效天数"]:
            return False, (f"{city} {season or '全年'}无 {p} 有效监测数据"
                           f"（年份={year or '全部'}）。请检查城市名/季节是否正确"), None
        self._ctx["city"] = city
        desc = f"{city}{season or '全年'} {p} 超标统计"
        obs = (f"{city}{season or '全年'} {p}：超标 {info['超标天数']} 天 / "
               f"有效 {info['有效天数']} 天，超标率 {info['超标率%']}%"
               f"（限值 {info['限值']}，{info['口径']}）")
        self._save_block("season_exceed", info, desc, info["限值"], p, year, obs)
        return True, obs, None

    def _tool_anomaly(self, args, plan):
        p = planner._valid_pollutant(args.get("pollutant")) or plan.pollutant
        year = planner._valid_year(args.get("year"), plan.year)
        city = planner.normalize_city(args.get("city")) or \
            (plan.cities[0] if plan.cities else None) or \
            (self.memory.last_cities[0] if self.memory.last_cities else None) or "北京"
        err = self._check_year(year)
        if err:
            return False, err, None
        hits = analysis.detect_anomalies(self.df, city, p, year)
        if hits.empty and self.df[self.df["城市"] == city].empty:
            return False, f"数据集中没有城市「{city}」的记录，请检查城市名", None
        self._ctx["city"] = city
        desc = f"{city} {p} 异常跳变检测"
        obs = f"{city} {p} 异常检测：检出 {len(hits)} 个跳变点（MAD 3σ）"
        if len(hits):
            obs += "\n" + df_to_text(hits.head(10))
        self._save_block("anomaly", (city, hits), desc, None, p, year, obs)
        return True, obs, None

    def _tool_check_standard(self, args, plan):
        p = planner._valid_pollutant(args.get("pollutant")) or plan.pollutant
        kind = args.get("kind") or config.DEFAULT_LIMIT_KIND.get(p)
        text = analysis.format_limit_text(p, kind)
        self._ctx["limit_text"] = text
        return True, f"{p} 判定依据：{text}", None

    def _tool_plot_chart(self, args, plan):
        kind = self._ctx.get("table_kind")
        if kind is None:
            return False, "没有可绘制的统计结果（plot_chart 须放在聚合类工具之后）", None
        tbl_now = self._ctx.get("table")
        if kind in ("annual_rank", "city_exceed", "province_mean") \
                and tbl_now is not None and len(tbl_now) > 40:
            return True, (f"统计结果共 {len(tbl_now)} 行，超出图表可读上限（40 行），"
                          f"已跳过绘图——完整排名见结果表格"), None
        chart = args.get("chart") or "auto"
        if chart == "auto":
            chart = _AUTO_CHART.get(kind, "bar")
        p = self._ctx.get("pollutant") or plan.pollutant
        unit = config.GB3095_GRADE2[p]["unit"]
        year = self._ctx.get("year")
        year_txt = f"{year}年" if year else "全时段"
        tbl = self._ctx["table"]

        path = None
        actual = chart
        if kind == "trend":
            path = charts.line_chart(tbl, f"{year_txt}{p}月度变化曲线", unit)
        elif kind == "anomaly":
            city, hits = tbl
            d = self.df[self.df["城市"] == city]
            if year:
                d = d[d["日期"].dt.year == year]
            col = config.DAILY_JUDGE_COL[p]
            s = d.set_index("日期")[col].dropna().sort_index()
            path = charts.anomaly_chart(s, hits, city, p, unit)
            actual = "line"
        elif kind in ("province_share", "group_share"):
            label_col = tbl.columns[0]
            if chart == "bar":
                path = charts.bar_chart(tbl, label_col, "超标天数",
                                        f"{year_txt}{p}超标天数（按{label_col}）", "天")
                actual = "bar"
            else:
                path = charts.pie_chart(tbl, label_col, "占全国比例%",
                                        f"{year_txt}{p}超标天数占比")
                actual = "pie"
        elif kind == "season_exceed":
            one = pd.DataFrame({"项目": ["超标天数", "有效天数", "超标率%"],
                                "数值": [tbl["超标天数"], tbl["有效天数"], tbl["超标率%"]]})
            path = charts.bar_chart(one, "项目", "数值",
                                    f"{tbl['城市']}{tbl['季节']} {p} 超标统计", "天")
            actual = "bar"
        elif kind == "compare":
            path = charts.bar_chart(tbl.reset_index(), "分组", "mean",
                                    f"{year_txt}{p}{self._ctx['desc'].split('对比')[0].split(p)[-1]}对比",
                                    unit)
            actual = "bar"
        elif kind == "annual_rank":
            path = charts.bar_chart(tbl, "城市", "年均浓度",
                                    f"{year_txt}{p}年均浓度 TOP{len(tbl)}", unit)
            actual = "bar"
        elif kind == "city_exceed":
            path = charts.bar_chart(tbl, "城市", "超标天数",
                                    f"{year_txt}{p}超标天数 TOP{len(tbl)}", "天")
            actual = "bar"
        elif kind == "province_mean":
            path = charts.bar_chart(tbl, "省份", "平均浓度",
                                    f"{year_txt}各省{p}平均浓度", unit)
            actual = "bar"

        if not path:
            return False, f"结果类型 {kind} 暂不支持绘图", None
        self._ctx["chart_path"] = path
        self._emit("chart", url="", name=os.path.basename(path))
        return True, f"图表已生成（{actual} 图）：{os.path.basename(path)}", None

    def _tool_validate(self, args, plan):
        blocks = self._ctx.get("results") or []
        if not blocks:
            ok, msg = True, "没有可校验的统计结果"
        else:
            parts = []
            all_ok = True
            for i, b in enumerate(blocks, 1):
                kind, tbl = b["kind"], b["table"]
                if kind in ("province_share", "group_share"):
                    ok, msg = analysis.validate_share(tbl)
                elif kind == "annual_rank":
                    ok, msg = analysis.validate_rank(tbl)
                elif kind == "city_exceed":
                    ok, msg = True, f"样本覆盖 {len(tbl)} 个城市，超标天数已排序"
                elif kind == "province_mean":
                    ok, msg = True, f"样本覆盖 {len(tbl)} 个省份，平均浓度已排序"
                elif kind == "compare":
                    ok, msg = True, f"分组均值对比，共 {len(tbl)} 组"
                elif kind == "trend":
                    ok, msg = True, f"{len(tbl.columns)} 个城市 × {len(tbl)} 个月，序列完整"
                elif kind == "season_exceed":
                    ok, msg = True, f"有效监测日 {tbl['有效天数']} 天，口径 {tbl['口径']}"
                elif kind == "anomaly":
                    ok, msg = True, f"MAD 3σ 检出 {len(tbl[1])} 个跳变点"
                else:
                    ok, msg = True, "样本量充足，统计口径一致"
                all_ok = all_ok and ok
                tag = f"结果{i}" if len(blocks) > 1 else "校验"
                parts.append(f"{tag}:{'通过' if ok else '存疑'}（{msg}）")
            ok, msg = all_ok, "；".join(parts)
        text = ("通过" if ok else "存疑") + f"（{msg}）" if len(blocks) <= 1 else msg
        self._ctx["validate_text"] = text
        return True, text, None

    # ---------- 结果汇总 ----------

    @staticmethod
    def _unit(p):
        return config.GB3095_GRADE2[p]["unit"]

    def _collect_out_of_scope(self, plan, seconds):
        """能力边界说明：诚实拒答 + 明确告诉用户能分析什么。"""
        years = self._years()
        ys = "、".join(str(y) for y in years) or str(self.default_year)
        n_city = self.clean_report.get("城市数", 375)
        reason = plan.scope_reason or "该问题超出本智能体的数据能力范围"
        concl = (f"【能力边界说明】{reason}\n"
                 f"本智能体基于中国环境监测总站 CN-AIR 公开数据集（全国 {n_city} 个城市，"
                 f"{ys} 年），可分析：PM2.5 / PM10 / O3 / SO2 / NO2 / CO / AQI 的日均时序——"
                 "浓度排名（最高/最低）、月度变化趋势、超标天数（GB 3095-2012 与 HJ 633-2012 "
                 "口径）、时序异常检测、省份/季节/南北方对比。欢迎换个空气质量角度提问。")
        return {"口径": "能力边界说明（未执行统计）", "表格HTML": "", "表格文本": "",
                "超标研判": "", "校验": "未执行统计", "图表路径": None,
                "intent": "out_of_scope", "source": plan.source, "_concl": concl}

    def _block_tables(self, b):
        """单个结果块 → (表格HTML, 表格文本)。"""
        kind, tbl = b["kind"], b["table"]
        if kind in ("annual_rank", "city_exceed", "province_mean", "province_share",
                    "group_share"):
            return df_to_html(tbl), df_to_text(tbl)
        if kind == "compare":
            return df_to_html(tbl.reset_index()), df_to_text(tbl.reset_index())
        if kind == "trend":
            return df_to_html(tbl.reset_index()), df_to_text(tbl.reset_index())
        if kind == "season_exceed":
            unit = self._unit(b["pollutant"])
            df = pd.DataFrame({"项目": ["超标天数", "有效天数", "超标率%", "判定限值"],
                               "数值": [tbl["超标天数"], tbl["有效天数"],
                                       tbl["超标率%"], f"{tbl['限值']}{unit}（{tbl['口径']}）"]})
            return df_to_html(df), df_to_text(df)
        if kind == "anomaly":
            city, hits = tbl
            if len(hits):
                return df_to_html(hits), df_to_text(hits)
            return ("<p>未检测到显著异常跳变</p>", "未检测到显著异常跳变")
        return ("<p>该结果类型暂无表格</p>", "（无）")

    def _block_research(self, idx, b, plan, multi):
        """单个结果块的研判文案（含差值/排名定位/方向措辞等增强）。"""
        kind, tbl, p, year = b["kind"], b["table"], b["pollutant"], b["year"]
        unit = self._unit(p)
        year_txt = f"{year}年" if year else "全时段"
        asc = b.get("order", "desc") == "asc"
        prefix = f"结果{idx}：" if multi else ""

        if kind == "annual_rank":
            if tbl.empty:
                return prefix + "无数据行（年份/城市/污染物可能超出数据范围）"
            n = int((tbl["是否超标"] == "超标").sum())
            annual_limit = config.GB3095_GRADE2[p].get("annual")
            val_col = tbl.columns[1]
            top_n = len(tbl)
            if asc:
                head = (f"{p}浓度最低的 {top_n} 个城市（数值越小空气越好）："
                        f"最清洁的是 {tbl.iloc[0]['城市']}（{tbl.iloc[0][val_col]}{unit}）")
            else:
                head = f"{p}浓度最高的 {top_n} 个城市：最重的是 {tbl.iloc[0]['城市']}（{tbl.iloc[0][val_col]}{unit}）"
            extra = []
            # 全量表（≥300 城）→ 直接回答「多少个达标」「排全国第几」
            if top_n >= 300:
                extra.append(f"在 {top_n} 个有效统计城市中，{top_n - n} 个达标、{n} 个超标")
            if annual_limit and top_n < 300:
                tail = f"其中 {n} 个城市年均值超过二级限值 {annual_limit}{unit}"
            elif annual_limit:
                tail = f"判定限值 {annual_limit}{unit}（年均二级标准）"
            else:
                tail = (f"该污染物无年均限值，排名仅供对比"
                        f"（依据 {analysis.format_limit_text(p)}）")
            # 两城市对比 → 直接给出差值，回答「重多少/差多少」
            if len(tbl) == 2 and pd.api.types.is_numeric_dtype(tbl[val_col]):
                hi, lo = tbl.iloc[0], tbl.iloc[1]
                diff = round(float(hi[val_col]) - float(lo[val_col]), 1)
                extra.append(f"{hi['城市']} 比 {lo['城市']} 高 {diff}{unit}"
                             f"（{hi[val_col]} vs {lo[val_col]}）")
            # 全量表 + 指定城市 → 给出「排全国第几」
            # 目标城市来源：结果块参数 → 计划顶层 → 问题原文提取（LLM 偶尔漏传 cities 时的兜底）
            targets = b.get("cities") or plan.cities or planner._find_cities(plan.question)
            for c in targets:
                hit = tbl.index[tbl["城市"] == c]
                if len(hit):
                    r = int(hit[0]) + 1
                    extra.append(f"{c} 的{p}年均浓度 "
                                 f"{tbl.loc[hit[0], val_col]}{unit}，"
                                 f"全国排名 第 {r} 位（共 {top_n} 城）")
            return prefix + head + "；" + tail + ("；" + "；".join(extra) if extra else "")

        if kind == "city_exceed":
            limit = b.get("limit")
            if tbl.empty:
                return prefix + "无数据行"
            direction = "最少" if asc else "最多"
            return (prefix + f"{p}超标天数{direction}的 {len(tbl)} 个城市："
                    f"{tbl.iloc[0]['城市']}（{tbl.iloc[0]['超标天数']} 天，"
                    f"超标率 {tbl.iloc[0]['超标率%']}%）；日均限值口径 {limit}{unit}")

        if kind == "province_mean":
            if tbl.empty:
                return prefix + "无数据行"
            n = int((tbl["是否超标"] == "超标").sum())
            annual_limit = config.GB3095_GRADE2[p].get("annual")
            direction = "最低" if asc else "最高"
            if annual_limit:
                return (prefix + f"{len(tbl)} 个省份中{p}平均浓度{direction}的是 "
                        f"{tbl.iloc[0]['省份']}（{tbl.iloc[0]['平均浓度']}{unit}）；"
                        f"{n} 个省份年均值超过二级限值 {annual_limit}{unit}")
            return (prefix + f"{len(tbl)} 个省份{p}平均浓度{direction}的是 "
                    f"{tbl.iloc[0]['省份']}（{tbl.iloc[0]['平均浓度']}{unit}）；"
                    f"该污染物无年均限值（依据 {analysis.format_limit_text(p)}）")

        if kind in ("province_share", "group_share"):
            label_col = tbl.columns[0]
            if tbl.empty:
                return prefix + "无数据行"
            return (prefix + f"{p}超标天数最多的{label_col}：{tbl.iloc[0][label_col]}"
                    f"（占全国 {tbl.iloc[0]['占全国比例%']}%，"
                    f"组内超标率 {tbl.iloc[0]['组内超标率%']}%）")

        if kind == "season_exceed":
            return (prefix + f"{tbl['城市']}{tbl['季节']}共 {tbl['有效天数']} 个有效监测日，"
                    f"{p}超标 {tbl['超标天数']} 天"
                    f"（限值 {tbl['限值']}{unit}，{tbl['口径']}）")

        if kind == "anomaly":
            city, hits = tbl
            return (prefix + f"{city} {p} 检出 {len(hits)} 个异常跳变点（MAD 3σ 阈值），"
                    f"可用于污染突发时段识别")

        if kind == "compare":
            g = tbl.reset_index()
            best, worst = g.iloc[0]["分组"], g.iloc[-1]["分组"]
            return (prefix + f"{p} 平均浓度最高分组：{best}（{g.iloc[0]['mean']}{unit}），"
                    f"最低：{worst}（{g.iloc[-1]['mean']}{unit}）")

        if kind == "trend":
            if tbl.empty:
                return prefix + "趋势无数据"
            return (prefix + f"{p} 月度浓度曲线已生成，各城市平均的峰值月份为 "
                    f"{tbl.mean(axis=1).idxmax()} 月")

        return prefix + "该结果类型暂无研判"

    def _collect(self, plan, seconds):
        """把全部结果块整理为结果字典（供文本/HTML 报告与网页端使用）。"""
        ctx = self._ctx
        blocks = ctx.get("results") or []

        out = {"口径": "", "图表路径": ctx.get("chart_path"),
               "校验": ctx.get("validate_text", ""),
               "intent": plan.intent, "source": plan.source}

        if not blocks:
            out["口径"] = "本次调度链未产出统计结果（相关步骤失败或被拦截）"
            out["表格HTML"] = "<p>无统计结果——请检查问题要素（年份/城市/污染物）是否在数据范围内</p>"
            out["表格文本"] = "（无）"
            out["超标研判"] = ""
            out["_concl"] = build_text_report(plan, out)
            return out

        html_parts, text_parts, research = [], [], []
        multi = len(blocks) > 1
        for i, b in enumerate(blocks, 1):
            bhtml, btext = self._block_tables(b)
            research.append(self._block_research(i, b, plan, multi))
            if multi:
                head = f"结果{i}：{b['desc']}"
                html_parts.append(
                    f'<div class="rblock"><div class="rblock-title">◈ {head}</div>{bhtml}</div>')
                text_parts.append(f"◆ {head}\n{btext}")
            else:
                html_parts.append(bhtml)
                text_parts.append(btext)

        src = "；数据来源：中国环境监测总站公开数据集（CN-AIR）"
        out["口径"] = (blocks[-1]["desc"] + src) if not multi else \
            ("；".join(b["desc"] for b in blocks) + src)
        out["表格HTML"] = "".join(html_parts)
        out["表格文本"] = "\n\n".join(text_parts)
        out["超标研判"] = "\n".join(research)
        out["_concl"] = build_text_report(plan, out)
        return out
