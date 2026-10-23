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

    # 4) 多部分问题（双向问法 / 区域拆解）——规则模式语义完整性回归
    print("\n===== 多部分问题：双向问法 + 区域拆解 =====")
    failures = []

    def check(name, cond, detail):
        print(f"  [{'✅' if cond else '❌'}] {name}: {detail}")
        if not cond:
            failures.append(name)

    # 4.1 单元级：_dual_direction 判定
    check("_dual_direction 双向问法", planner._dual_direction(
        "2024年PM2.5年均浓度最高和最低的5个城市分别是哪些") is True, "最高+最低 → True")
    check("_dual_direction 单向不误判", planner._dual_direction(
        "2024年全国空气质量最好的10个城市是哪些") is False, "仅「最好」 → False")
    check("_dual_direction 反向单向", planner._dual_direction(
        "2024年CO浓度最低的5个省份排名") is False, "仅「最低」 → False")

    # 4.2 端到端：M2 双向问法 → 最高/最低两块都出
    q_m2 = "2024年PM2.5年均浓度最高和最低的5个城市分别是哪些"
    r = agent.run(q_m2)
    research = r.get("超标研判", "")
    n_blocks = research.count("结果")
    check("M2 双向出两块", "结果1" in research and "结果2" in research,
          f"研判含 结果1/结果2（{n_blocks} 处标记）")
    check("M2 最高块", "最高" in research and "和田地区" in research,
          "最高块含和田地区")
    check("M2 最低块", "最低" in research and "阿里地区" in research,
          "最低块含阿里地区")
    orders = [s.args.get("order") for s in r["plan"].steps if s.tool == "aggregate"]
    check("M2 双方向参数", orders == ["desc", "asc"], f"aggregate order={orders}")

    # 4.3 端到端：C3 省份占比 + 南北方拆开
    q_c3 = "各省PM2.5超标天数占比，再按南北方拆开对比"
    r3 = agent.run(q_c3)
    research3 = r3.get("超标研判", "")
    check("C3 省份占比块", "省份" in research3 and "河南" in research3,
          "省份占比块含河南")
    check("C3 区域拆解块", "区域对比" in research3 and "北方" in research3 and "南方" in research3,
          "区域对比块含北方/南方")
    gbs = [s.args.get("group_by") for s in r3["plan"].steps if s.tool == "aggregate"]
    check("C3 双聚合维度", gbs == ["province", "north_south"], f"aggregate group_by={gbs}")

    print(f"\n# 多部分问题回归: {'全部通过' if not failures else '失败: ' + ', '.join(failures)}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
