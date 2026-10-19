# -*- coding: utf-8 -*-
"""真实 LLM 联调：用 DeepSeek 实测规划器对全新问法的调用链生成。"""
import sys
import os

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aq_agent import config

loaded = config.load_env_file()
print("已加载 .env:", loaded)

from aq_agent import planner
from aq_agent.memory import Memory

# 三个规则规划器没见过的新问法
questions = [
    "广东和江苏哪个省份臭氧污染更严重？说出一二名",
    "春天的时候全国二氧化氮情况怎么样",
    "帮我看下呼和浩特去年一氧化碳有没有哪天突然飙升",
]
for q in questions:
    print("=" * 60)
    print("Q:", q)
    try:
        p = planner.plan(q, Memory(), 2024)
    except Exception as e:
        print("!! 规划异常:", type(e).__name__, e)
        continue
    print("source:", p.source, "| pollutant:", p.pollutant)
    print("调用链:")
    for i, s in enumerate(p.steps, 1):
        print(f"  {i}. {s.tool}  args={s.args}")
        print(f"     thought: {s.thought}")
