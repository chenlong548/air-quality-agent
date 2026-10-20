# -*- coding: utf-8 -*-
"""命令行交互入口。"""
import os
import sys
import webbrowser

from . import config
from .data_layer import discover_years, load_dataset
from .memory import Memory
from .react_agent import ReActAgent
from .report import build_html_report, build_text_report

BANNER = """====================================================
 空气质量时序分析 Agent（环境 + 大数据）
 ReAct 智能体 + 本地真实数据集 + GB 3095-2012 国标研判
====================================================
输入自然语言提问即可分析，例如：
  · 找出2024年PM2.5年均浓度最高的5个城市，画出月度变化曲线
  · 筛选PM2.5超标记录，统计各省份超标天数占比
  · 统计北京市夏季臭氧O3超标总天数
  · 北京2024年O3有没有异常跳变
命令：help 帮助 | board 任务看板 | mode 专业/简易 | quit 退出
"""

HELP = """支持的提问方式（自然语言，要素可自由组合）：
  污染物：PM2.5 / PM10 / 臭氧O3 / SO2 / NO2 / CO / AQI
  时间： 2024年、2023年…；春季/夏季/秋季/冬季
  地域： 全国 / 某省份 / 某城市（如 北京、和田地区）
  意图： 年均浓度TOP5、月度变化曲线、各省份超标占比、
         城市季节超标天数、异常跳变检测、南北方对比
  模板： 结尾加「专业版」输出专业统计口径，「简易版」输出通俗解读
"""


class CLI:
    def __init__(self, year=None):
        self.memory = Memory()
        self.audience_mode = None
        years = discover_years()
        if not years:
            raise SystemExit("未找到数据文件，请先运行 scripts/prepare_data.py 或查看 data/DATASET.md")
        self.year = year or max(years)
        loaded = config.load_env_file()
        llm_on = bool(os.environ.get("AQ_LLM_BASE_URL") and os.environ.get("AQ_LLM_API_KEY"))
        if loaded:
            print(f"[配置] 已从 .env 加载: {', '.join(loaded)}")
        print(f"[启动] 规划器：{'LLM function calling' if llm_on else '本地规则（未配置 Key，无网可跑）'}")
        print(f"[启动] 加载数据集 air_quality_{self.year}.csv（分片读取）...")
        df, report = load_dataset(config.data_file(self.year))
        self.agent = ReActAgent(df, report, self.memory, self.year)
        print(f"[启动] 完成：{report['清洗后行数']} 行 / {report['城市数']} 城市 / {report['日期范围']}")

    def handle(self, question: str) -> bool:
        q = question.strip()
        if not q:
            return True
        low = q.lower()
        if low in ("quit", "exit", "退出"):
            return False
        if low in ("help", "帮助", "?"):
            print(HELP)
            return True
        if low in ("board", "看板"):
            st = self.memory.stats()
            print(f"[任务看板] 总任务 {st['总任务数']}，平均耗时 {st['平均耗时秒']}s，"
                  f"高频任务 {st['高频任务']}")
            return True
        if low.startswith("mode") or low.startswith("模板"):
            self.audience_mode = "pro" if ("专业" in q or "pro" in low) else "simple"
            print(f"[模板] 已切换为{'专业统计版' if self.audience_mode=='pro' else '简易解读版'}")
            return True

        result = self.agent.run(q)
        if self.audience_mode:
            result["plan"].audience = self.audience_mode
        print("\n----- Agent 执行轨迹（ReAct） -----")
        print(result["trace"])
        print("\n----- 分析结论 -----")
        print(build_text_report(result["plan"], result))
        html_path = build_html_report(result["plan"], result)
        print(f"\n[HTML 报告] {html_path}")
        try:
            webbrowser.open("file://" + os.path.abspath(html_path))
        except Exception:
            pass
        return True

    def loop(self):
        print(BANNER)
        while True:
            try:
                q = input("\n请输入分析问题 > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break
            if not self.handle(q):
                print("再见！")
                break


def main():
    year = None
    if len(sys.argv) > 1:
        # 支持: run.py "问题" 或 run.py --year 2023 "问题"
        args = sys.argv[1:]
        if args[0] == "--year" and len(args) > 1:
            year = int(args[1])
            args = args[2:]
            if not os.path.exists(config.data_file(year)):
                available = discover_years()
                print(f"[错误] 数据集中没有 {year} 年的文件（可用年份：{available}）。"
                      f"如需准备 {year} 年数据，请运行 scripts/fetch_data.py 与 scripts/prepare_data.py")
                return
        if args:
            cli = CLI(year)
            cli.handle(" ".join(args))
            return
    CLI(year).loop()


if __name__ == "__main__":
    main()
