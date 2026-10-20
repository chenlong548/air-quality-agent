# -*- coding: utf-8 -*-
"""泛化能力审计：用一批全新问法检验 Agent 是否真的"理解问题"而非查表。

覆盖维度：多部分问题 / 语义方向（最好vs最差） / AQI 维度 / 年份边界 /
域外问题 / 口语错别字 / 多轮指代 / 深组合链 / 鲁棒性。
结果写入 outputs/logs/generalization_audit.txt。
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aq_agent import config
from aq_agent.data_layer import discover_years, load_dataset
from aq_agent.memory import Memory
from aq_agent.react_agent import ReActAgent

OUT = os.path.join(config.LOG_DIR, "generalization_audit.txt")
lines = []


def log(s=""):
    print(s)
    lines.append(s)


# ---------------- 单轮问题集 ----------------
SINGLE = [
    # --- 多部分问题（疑点：单结果上下文，后一步覆盖前一步）---
    ("M1", "对比一下2024年南方和北方的PM2.5污染差异，并列出各自超标最严重的3个省份"),
    ("M2", "2024年PM2.5年均浓度最高和最低的5个城市分别是哪些"),
    ("M3", "北京和上海2024年谁的PM2.5污染更重，重多少"),
    # --- 语义方向（疑点：只有降序，没有"最好/最低"）---
    ("D1", "2024年全国空气质量最好的10个城市是哪些"),
    ("D2", "2024年CO浓度最低的5个省份排名"),
    # --- AQI 维度（疑点：HELP 宣称支持 AQI，实际会静默按 PM2.5 处理）---
    ("A1", "2024年全国AQI年均值最高的5个城市"),
    # --- 年份/数据边界（疑点：问 2023 得 0 行空表，无明确提示）---
    ("Y1", "统计2023年全国PM2.5年均浓度排名"),
    ("Y2", "2025年北京PM2.5月度变化趋势"),
    # --- 域外问题（疑点：数据无气温/降雨/风力，会硬答空气质量）---
    ("O1", "2024年北京的平均气温和降雨量是多少"),
    ("O2", "上海2024年刮大风的天气多吗"),
    ("O3", "你好"),
    # --- 口语/错别字/别名 ---
    ("S1", "2024年俺们首都的pm2.5年均浓度是多少"),
    ("S2", "2024年广州的二氧化硫年均浓度排全国第几"),
    # --- 组合深度 ---
    ("C1", "臭氧污染最严重的是哪个季节，其次呢"),
    ("C2", "2024年全国有多少个城市的PM2.5年均浓度达到二级标准"),
    ("C3", "各省PM2.5超标天数占比，再按南北方拆开对比"),
]

# ---------------- 多轮问题对（疑点：LLM 路径不用 memory，指代失效）---
MULTI = [
    ("T1a", "2024年北京市PM2.5年均浓度是多少", "T1b", "那上海呢"),
    ("T2a", "找出2024年PM2.5年均浓度最高的5个城市", "T2b", "这几个城市里哪个臭氧污染最严重"),
]


def brief(result):
    tbl = result.get("表格文本", "")
    n_rows = max(0, len([l for l in tbl.splitlines() if l.strip()]) - 1)
    return (f"表格行数={n_rows} | 研判={result.get('超标研判', '-')[:80]} | "
            f"校验={result.get('校验', '-')[:40]}")


def obs_lines(result, limit=6):
    out = []
    for l in result.get("trace", "").splitlines():
        if l.startswith("Observation:"):
            out.append(l[:150])
    return out[-limit:]


def run_one(agent, tag, q):
    log(f"\n===== {tag}: {q} =====")
    try:
        r = agent.run(q)
    except Exception as e:
        log(f"!! 崩溃: {type(e).__name__}: {e}")
        return None
    plan = r["plan"]
    chain = " → ".join(f"{s.tool}({','.join(f'{k}={v}' for k, v in s.args.items()) or '无参'})"
                       for s in plan.steps)
    log(f"[规划] source={plan.source} pollutant={plan.pollutant} year={plan.year} "
        f"cities={plan.cities} top_n={plan.top_n}")
    log(f"[调用链] {chain}")
    for l in obs_lines(r):
        log(f"  {l}")
    log(f"[结论] {brief(r)}")
    concl = r.get("_concl", "")
    log(f"[结论文案前120字] {concl[:120].replace(chr(10), ' / ')}")
    return r


def main():
    config.load_env_file()   # 加载 .env 中的 AQ_LLM_*，启用 LLM 规划路径
    years = discover_years()
    df, report = load_dataset(config.data_file(max(years)))
    agent = ReActAgent(df, report, Memory(), max(years))
    log(f"# 泛化审计  data={report['文件']} rows={report['清洗后行数']} "
        f"cities={report['城市数']} range={report['日期范围']}")
    log(f"# LLM: base={os.environ.get('AQ_LLM_BASE_URL', '')} model={os.environ.get('AQ_LLM_MODEL', '')}")

    for tag, q in SINGLE:
        run_one(agent, tag, q)
    for t1, q1, t2, q2 in MULTI:
        run_one(agent, t1, q1)
        run_one(agent, t2, q2)   # 同一 agent，记忆延续

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n# 已写入 {OUT}")


if __name__ == "__main__":
    main()
