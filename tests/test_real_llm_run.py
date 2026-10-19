# -*- coding: utf-8 -*-
"""真实 LLM 全链路：规划 + 执行 + 图表 + 结论。"""
import sys
import os

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aq_agent import config
config.load_env_file()

from aq_agent.data_layer import discover_years, load_dataset
from aq_agent.memory import Memory
from aq_agent.react_agent import ReActAgent

year = max(discover_years())
df, report = load_dataset(config.data_file(year))
agent = ReActAgent(df, report, Memory(), year)

questions = [
    "广东和江苏哪个省份臭氧污染更严重？说出一二名",
    "春天的时候全国二氧化氮情况怎么样",
    "帮我看下呼和浩特去年一氧化碳有没有哪天突然飙升",
]
for q in questions:
    print("=" * 66)
    print("Q:", q)
    result = agent.run(q)
    print("source:", result["plan"].source, "| steps:", len(result["plan"].steps),
          "| seconds:", result["seconds"])
    for line in result["trace"].splitlines():
        if line.startswith(("Action:", "Observation:", "[错误]", "[图表]")):
            print("   ", line[:110])
    print("   超标研判:", result.get("超标研判", "")[:100])
    print("   校验:", result["校验"])
    print("   图表:", os.path.basename(result["图表路径"]) if result["图表路径"] else "无")
