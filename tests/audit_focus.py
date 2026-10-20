# -*- coding: utf-8 -*-
"""修复后的针对性复验：S2 排名定位 / C2 达标数直答 / Y1 年份替换告知。"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aq_agent import config
config.load_env_file()
from aq_agent.data_layer import discover_years, load_dataset
from aq_agent.memory import Memory
from aq_agent.react_agent import ReActAgent

years = discover_years()
df, rep = load_dataset(config.data_file(max(years)))
agent = ReActAgent(df, rep, Memory(), max(years))

QS = [
    "2024年广州的二氧化硫年均浓度排全国第几",
    "2024年全国有多少个城市的PM2.5年均浓度达到二级标准",
    "统计2023年全国PM2.5年均浓度排名",
]
for q in QS:
    r = agent.run(q)
    print("=" * 70)
    print("Q:", q)
    print("[source]", r["plan"].source)
    for t in r["plan"].thoughts:
        print("[thought]", t[:120])
    print("[研判]", (r.get("超标研判") or "").replace("\n", " ⏎ "))
