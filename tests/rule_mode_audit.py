# -*- coding: utf-8 -*-
"""规则兜底模式泛化审计：强制不走 LLM，检验本地规则规划器的覆盖度。

与 generalization_audit.py 同一问题集，但屏蔽 AQ_LLM_* 环境变量。
LLM 不可用（无网/无 Key/接口故障）时，规则模式是最后防线——
它决定智能体「断网时还能不能用」。
结果写入 outputs/logs/rule_mode_audit.txt。
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aq_agent import config
from aq_agent.data_layer import discover_years, load_dataset
from aq_agent.memory import Memory
from aq_agent.react_agent import ReActAgent

OUT = os.path.join(config.LOG_DIR, "rule_mode_audit.txt")
lines = []

# 复用同一问题集
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generalization_audit import SINGLE, MULTI  # noqa: E402


def log(s=""):
    print(s)
    lines.append(s)


def brief(result):
    tbl = result.get("表格文本", "")
    n_rows = max(0, len([l for l in tbl.splitlines() if l.strip()]) - 1)
    return (f"表格行数={n_rows} | 研判={result.get('超标研判', '-')[:100]} | "
            f"校验={result.get('校验', '-')[:40]}")


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
    log(f"[规划] source={plan.source} intent={plan.intent} pollutant={plan.pollutant} "
        f"year={plan.year} cities={plan.cities} top_n={plan.top_n} "
        f"out_of_scope={plan.out_of_scope}")
    log(f"[调用链] {chain}")
    log(f"[结论] {brief(r)}")
    return r


def main():
    # 关键：加载 .env 后立即拔掉 LLM 配置 → 强制规则模式
    config.load_env_file()
    os.environ.pop("AQ_LLM_BASE_URL", None)
    os.environ.pop("AQ_LLM_API_KEY", None)

    years = discover_years()
    df, report = load_dataset(config.data_file(max(years)))
    agent = ReActAgent(df, report, Memory(), max(years))
    log(f"# 规则模式审计  data={report['文件']} rows={report['清洗后行数']} "
        f"cities={report['城市数']} range={report['日期范围']}")
    log("# LLM 已强制屏蔽（模拟无网/无 Key 的兜底场景）")

    ok = 0
    total = 0
    for tag, q in SINGLE:
        r = run_one(agent, tag, q)
        total += 1
        if r and (r.get("表格文本", "（无）") != "（无）" or r["plan"].out_of_scope):
            ok += 1
    for t1, q1, t2, q2 in MULTI:
        r1 = run_one(agent, t1, q1)
        r2 = run_one(agent, t2, q2)
        total += 2
        ok += sum(1 for r in (r1, r2)
                  if r and (r.get("表格文本", "（无）") != "（无）" or r["plan"].out_of_scope))

    log(f"\n# 汇总：{ok}/{total} 题给出了有效结果（表格或诚实拒答）")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n# 已写入 {OUT}")


if __name__ == "__main__":
    main()
