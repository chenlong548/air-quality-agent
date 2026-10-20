# -*- coding: utf-8 -*-
"""补齐不足后的新能力冒烟测试：分组切片/相关性/来源分析/规则模式/剪枝。"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aq_agent import config, planner
from aq_agent.data_layer import discover_years, load_dataset
from aq_agent.memory import Memory
from aq_agent.react_agent import ReActAgent


def main():
    years = discover_years()
    df, report = load_dataset(config.data_file(max(years)))
    print(f"# 数据: {report['文件']} rows={report['清洗后行数']} cities={report['城市数']}")

    # 1) 分组切片：南北各自超标最严重3省（规则模式，不走 LLM）
    agent = ReActAgent(df, report, Memory(), max(years))
    for q in [
        "2024年南北各自超标最严重的3个省份是哪些",
        "北京和上海的PM2.5污染是否相关",
        "北京的PM2.5污染主要来自哪里",
        "臭氧污染最严重的是哪个季节",
        "2024年全国有多少个城市的PM2.5年均浓度达到二级标准",
        "2024年俺们首都的pm2.5年均浓度是多少",
    ]:
        print(f"\n===== {q} =====")
        r = agent.run(q)
        plan = r["plan"]
        print(f"[规划] source={plan.source} intent={plan.intent} "
              f"pollutant={plan.pollutant} top_n={plan.top_n} cities={plan.cities}")
        print(f"[调用链] {' → '.join(s.tool for s in plan.steps)}")
        print(f"[研判] {r.get('超标研判', '')[:220]}")
        print(f"[校验] {r.get('校验', '')[:120]}")
        tbl_lines = [l for l in r.get('表格文本', '').splitlines() if l.strip()]
        print(f"[表格] {len(tbl_lines) - 1} 行; 表头: {tbl_lines[0][:100] if tbl_lines else '无'}")
        if len(tbl_lines) > 1:
            print(f"  首行: {tbl_lines[1][:100]}")

    # 2) 剪枝单测
    print("\n===== 剪枝单测 =====")
    steps = [
        planner.Step("load_data", {}, "load"),
        planner.Step("check_standard", {"pollutant": "PM2.5"}, "ck"),
        planner.Step("aggregate", {"group_by": "city", "metric": "mean"}, "agg"),
        planner.Step("aggregate", {"group_by": "city", "metric": "mean"}, "agg重复"),
        planner.Step("check_standard", {"pollutant": "PM2.5"}, "ck重复"),
        planner.Step("validate", {}, "val"),
    ]
    pruned = planner._prune_steps(steps)
    print(f"剪枝: {len(steps)} → {len(pruned)} 步: {[s.tool for s in pruned]}")

    # 3) split_by 清洗
    print("\n===== _clean_args split_by =====")
    print(planner._clean_args("aggregate", {"group_by": "province", "metric": "exceed_days",
                                            "split_by": "north_south", "top_n": 3}))
    print(planner._clean_args("aggregate", {"group_by": "city", "split_by": "north_south"}))


if __name__ == "__main__":
    main()
