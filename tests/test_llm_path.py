# -*- coding: utf-8 -*-
"""LLM 规划路径联调：本地模拟 OpenAI 兼容接口，验证 function calling 全链路。"""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["AQ_LLM_BASE_URL"] = "http://127.0.0.1:8617/v1"
os.environ["AQ_LLM_API_KEY"] = "mock-key"
os.environ["AQ_LLM_MODEL"] = "mock-model"

# 模拟 LLM 对一个「规则规划器从未见过的新问法」生成的调用链
MOCK_REPLY = {
    "thoughts": [
        "用户问广东和江苏哪个省臭氧更严重，需要按省份聚合 O3 平均浓度并对比",
        "还需要判断哪个省超过国标年均/8h 限值",
    ],
    "pollutant": "O3",
    "steps": [
        {"tool": "load_data", "args": {},
         "thought": "先分片读取本地数据集并清洗"},
        {"tool": "aggregate", "args": {"group_by": "province", "metric": "mean",
                                       "pollutant": "O3", "year": 2024},
         "thought": "按省份分组计算 O3 平均浓度，覆盖广东与江苏"},
        {"tool": "check_standard", "args": {"pollutant": "O3"},
         "thought": "对照 GB 3095-2012 明确 O3 判定口径"},
        {"tool": "plot_chart", "args": {"chart": "bar"},
         "thought": "将两省浓度对比画成柱状图"},
        {"tool": "validate", "args": {},
         "thought": "校验样本量与排序"},
    ],
}

received = {}


class MockLLM(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        received["messages"] = body.get("messages", [])
        received["model"] = body.get("model")
        payload = {"choices": [{"message": {"role": "assistant",
                                            "content": json.dumps(MOCK_REPLY, ensure_ascii=False)}}]}
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", 8617), MockLLM)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    from aq_agent import planner
    from aq_agent.data_layer import discover_years, load_dataset
    from aq_agent.memory import Memory
    from aq_agent.react_agent import ReActAgent
    from aq_agent import config

    # 1) 规划层：验证 LLM 输出的调用链被正确解析与清洗
    q = "广东和江苏哪个省份臭氧污染更严重？说出一二名"
    p = planner.plan(q, Memory(), 2024)
    print("== 规划结果 ==")
    print("source:", p.source, "| pollutant:", p.pollutant, "| steps:",
          [s.tool for s in p.steps])
    assert p.source == "llm", "应走 LLM 规划路径"
    assert [s.tool for s in p.steps] == ["load_data", "aggregate", "check_standard",
                                         "plot_chart", "validate"]
    assert p.steps[1].args["group_by"] == "province"
    print("mock LLM 收到的 system prompt 长度:", len(received["messages"][0]["content"]))
    print("mock LLM 收到的 user 内容:", received["messages"][1]["content"])

    # 2) 执行层：验证引擎按 LLM 链跑完并产出真实结果
    df, report = load_dataset(config.data_file(max(discover_years())))
    agent = ReActAgent(df, report, Memory(), max(discover_years()))
    result = agent.run(q)
    print("\n== 执行结果 ==")
    print("intent:", result["plan"].intent, "| seconds:", result["seconds"])
    print("口径:", result["口径"])
    print("超标研判:", result.get("超标研判", ""))
    print("校验:", result["校验"])
    print("图表:", result["图表路径"])
    assert result["plan"].source == "llm"
    assert result["图表路径"] and os.path.isfile(result["图表路径"])
    print("\nLLM 规划路径全链路 OK")

    # 3) 容错：LLM 返回垃圾时应自动回退规则规划器
    MOCK_REPLY.clear()
    MOCK_REPLY.update({"steps": [{"tool": "不存在的工具", "args": {}}]})
    p2 = planner.plan("统计北京市夏季臭氧O3超标总天数", Memory(), 2024)
    print("\n== 垃圾输出容错 ==")
    print("回退后 source:", p2.source, "| steps:", [s.tool for s in p2.steps])
    assert p2.source == "rule", "LLM 输出无效时应回退规则规划器"
    print("容错回退 OK")

    srv.shutdown()


if __name__ == "__main__":
    main()
