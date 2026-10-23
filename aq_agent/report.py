# -*- coding: utf-8 -*-
"""输出层：结构化文本结论、HTML 报告、健康科普与数据说明。"""
import datetime
import html
import os

from . import config
from . import analysis

# ---------------- 健康科普与标准提示（随业务输出） ----------------

GB_TIP = ("我国《环境空气质量标准》GB 3095-2012 规定：{limit}。")

DATA_NOTE = ("数据说明：以上统计基于中国环境监测总站公开数据集 CN-AIR（日值口径），"
             "仅供研究与学习参考，不作为环境监管或执法依据。")

BUSINESS_NOTE = ("业务解读：时序监测大数据可支撑污染溯源与治理评估——"
                 "持续追踪高值时段与区域分布，是精准治霾、科学治污的重要依据。")

POLLUTION_CARD = {
    "PM2.5": "PM2.5 可深入肺部并进入血液，长期暴露增加呼吸道与心血管疾病风险。",
    "PM10": "PM10 可刺激呼吸道，诱发咳嗽与哮喘，影响能见度。",
    "O3": "臭氧(O₃)是二次污染物，强氧化性，刺激呼吸道、降低肺功能。",
    "SO2": "二氧化硫刺激呼吸道，是酸雨的重要前体物。",
    "NO2": "二氧化氮刺激肺部，是臭氧与二次颗粒物的重要前体物。",
    "CO": "一氧化碳与血红蛋白结合能力强，高浓度时影响人体供氧。",
    "AQI": "AQI 综合六项污染物换算成健康风险指数：>100 为轻度污染及以上，数值越大空气越差。",
}


def edu_limit_tip(pollutant, kind=None):
    limit_text = analysis.format_limit_text(pollutant, kind)
    return GB_TIP.format(limit=limit_text)


def build_text_report(plan, result) -> str:
    """结构化文本结论（终端展示用）。"""
    lines = []
    lines.append(f"【分析任务】{plan.raw}")
    lines.append(f"【统计口径】{result.get('口径', '')}")
    lines.append("【统计结果】")
    lines.append(result.get("表格文本", "（无）"))
    if result.get("超标研判"):
        lines.append(f"【超标研判】{result['超标研判']}")
    if result.get("图表路径"):
        lines.append(f"【图表输出】{result['图表路径']}")
    if result.get("校验"):
        lines.append(f"【结果校验】{result['校验']}")
    # 健康科普卡片 + 数据说明，随结果一并输出
    if plan.pollutant in POLLUTION_CARD:
        lines.append(f"【科普卡片】{POLLUTION_CARD[plan.pollutant]}")
    lines.append(DATA_NOTE)
    return "\n".join(lines)


def build_html_report(plan, result) -> str:
    """生成 HTML 报告（深空遥测主题，图表相对路径引用）。"""
    config.setup_chinese_font() if hasattr(config, "setup_chinese_font") else None
    chart_rel = ""
    if result.get("图表路径"):
        chart_rel = os.path.relpath(result["图表路径"], config.REPORT_DIR).replace("\\", "/")

    def sec(title, body_html):
        return f'<section><h2>{html.escape(title)}</h2>{body_html}</section>'

    table_html = result.get("表格HTML", "")
    info_html = ""
    if plan.pollutant in POLLUTION_CARD:
        info_html += f'<div class="tip"><b>科普卡片</b>：{html.escape(POLLUTION_CARD[plan.pollutant])}</div>'
    info_html += f'<div class="tip"><b>国标提示</b>：{html.escape(edu_limit_tip(plan.pollutant))}</div>'
    info_html += f'<div class="soul">{html.escape(DATA_NOTE)}</div>'

    page = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>空气质量时序分析报告</title>
<style>
:root{{--panel:#0d1526;--line:#1c2a4a;--line-hi:#2b4166;--txt:#d7e3f4;--dim:#8ba0bf;--cyan:#22d3ee;--cyan-hi:#67e8f9;--green:#34d399;--mono:"Cascadia Code","JetBrains Mono","Cascadia Mono",Consolas,"Liberation Mono",monospace;--sans:"Microsoft YaHei","PingFang SC","Noto Sans SC",sans-serif}}
*{{box-sizing:border-box}}
body{{font-family:var(--sans);color:var(--txt);margin:0;padding:28px 20px;background:radial-gradient(1100px 480px at 50% -200px,rgba(34,211,238,.10),transparent 60%),linear-gradient(rgba(56,189,248,.04) 1px,transparent 1px),linear-gradient(90deg,rgba(56,189,248,.04) 1px,transparent 1px),#05080f;background-size:auto,44px 44px,44px 44px,auto}}
.wrap{{max-width:920px;margin:0 auto;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:28px 34px;box-shadow:0 12px 36px rgba(2,6,15,.55)}}
h1{{font-size:22px;margin:0 0 4px;letter-spacing:.04em}}
.meta{{color:var(--dim);font-family:var(--mono);font-size:12px;margin-bottom:20px;letter-spacing:.02em}}
section{{margin:20px 0}}
h2{{font-size:15px;border-left:3px solid var(--cyan);padding-left:10px;margin:16px 0 10px;letter-spacing:.08em;color:var(--cyan-hi);font-family:var(--mono)}}
p{{line-height:1.8}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0}}
th,td{{border:1px solid var(--line);padding:7px 11px;text-align:left}}
th{{background:#0f1b31;color:var(--cyan-hi);font-family:var(--mono);font-size:12px;letter-spacing:.05em}}
tr:nth-child(even) td{{background:rgba(34,211,238,.03)}}
.tip{{background:rgba(34,211,238,.06);border:1px solid rgba(34,211,238,.25);border-radius:6px;padding:11px 15px;font-size:13px;margin:10px 0;line-height:1.75}}
.soul{{background:rgba(52,211,153,.07);border:1px solid rgba(52,211,153,.3);border-radius:6px;padding:11px 15px;font-size:13px;margin:10px 0;line-height:1.75}}
img{{max-width:100%;border:1px solid var(--line-hi);border-radius:8px;box-shadow:0 0 26px rgba(34,211,238,.10)}}
pre{{background:#080d18;border:1px solid var(--line);border-radius:6px;padding:12px;font-size:12px;white-space:pre-wrap;font-family:var(--mono);color:#b9cbe4;line-height:1.6}}
</style></head><body><div class="wrap">
<h1>空气质量时序分析报告</h1>
<div class="meta">生成时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ｜ 数据来源：中国环境监测总站公开数据（CN-AIR）</div>
{sec("分析任务", f"<p>{html.escape(plan.raw)}</p>")}
{sec("统计口径", f"<p>{html.escape(result.get('口径', ''))}</p>")}
{sec("统计结果", table_html)}
{sec("超标研判", f"<p>{html.escape(result.get('超标研判', ''))}</p>") if result.get("超标研判") else ""}
{sec("可视化图表", f'<img src="{chart_rel}" alt="chart">') if chart_rel else ""}
{sec("健康科普与数据说明", info_html)}
{sec("Agent 执行轨迹", f"<pre>{html.escape(result.get('trace', ''))}</pre>")}
</div></body></html>"""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(config.REPORT_DIR, f"report_{ts}.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    return path
