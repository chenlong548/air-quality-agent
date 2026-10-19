# -*- coding: utf-8 -*-
"""网页版交互界面（零第三方依赖，Python 标准库实现）。

启动：
    python web.py                 # 默认 8611 端口，自动打开浏览器
    python web.py --port 9000 --no-open

然后浏览器访问 http://127.0.0.1:8611
页面为聊天式交互：输入自然语言提问，实时展示 ReAct 执行轨迹、
统计表格、可视化图表与结构化结论。
"""
import argparse
import json
import os
import sys
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aq_agent import config
from aq_agent.data_layer import discover_years, load_dataset
from aq_agent.memory import Memory
from aq_agent.react_agent import ReActAgent
from aq_agent.report import build_text_report

INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>空气质量时序分析 Agent</title>
<style>
*{box-sizing:border-box}
body{font-family:"Microsoft YaHei","PingFang SC",sans-serif;background:#f5f7fa;color:#1f2937;margin:0;height:100vh;display:flex;flex-direction:column}
header{background:#fff;border-bottom:1px solid #e5e7eb;padding:10px 20px;display:flex;align-items:center;gap:12px}
header .logo{width:34px;height:34px;border-radius:8px;background:#3b7dd8;color:#fff;display:flex;align-items:center;justify-content:center;font-weight:700}
header h1{font-size:17px;margin:0}
header .meta{font-size:12px;color:#6b7280;margin-left:auto}
#chat{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:14px}
.msg{max-width:880px;width:100%;margin:0 auto}
.bubble{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:14px 16px;box-shadow:0 1px 3px rgba(0,0,0,.04)}
.msg.user .bubble{background:#3b7dd8;color:#fff;border-color:#3b7dd8}
.role{font-size:12px;color:#6b7280;margin-bottom:6px}
.msg.user .role{color:#e8f1ff}
pre.trace{background:#f8fafc;border:1px solid #e5e7eb;border-radius:8px;padding:10px;font-size:12px;white-space:pre-wrap;margin:8px 0;max-height:260px;overflow:auto}
table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}
th,td{border:1px solid #e5e7eb;padding:5px 9px;text-align:left}
th{background:#eef2f7}
img.chart{max-width:100%;border:1px solid #e5e7eb;border-radius:8px;margin-top:8px}
.concl{font-size:13.5px;white-space:pre-wrap;line-height:1.7}
.soul{margin-top:10px;background:#f0faf3;border:1px solid #c9e8d4;border-radius:8px;padding:8px 12px;font-size:12.5px}
details summary{cursor:pointer;font-size:12.5px;color:#3b7dd8}
#bar{background:#fff;border-top:1px solid #e5e7eb;padding:12px 20px}
#bar .inner{max-width:880px;margin:0 auto;display:flex;gap:10px}
#q{flex:1;border:1px solid #d1d5db;border-radius:10px;padding:10px 14px;font-size:14px;outline:none}
#q:focus{border-color:#3b7dd8}
button{background:#3b7dd8;color:#fff;border:none;border-radius:10px;padding:10px 20px;font-size:14px;cursor:pointer}
button:disabled{background:#9db9dd}
.chips{max-width:880px;margin:0 auto 10px;display:flex;gap:8px;flex-wrap:wrap}
.chip{font-size:12px;border:1px solid #d1d5db;border-radius:999px;padding:5px 12px;cursor:pointer;background:#fff;color:#374151}
.chip:hover{border-color:#3b7dd8;color:#3b7dd8}
.loading{color:#6b7280;font-size:13px}
</style></head><body>
<header>
  <div class="logo">AQ</div>
  <h1>空气质量时序分析 Agent</h1>
  <span class="meta" id="meta">ReAct 智能体 ｜ GB 3095-2012 国标研判 ｜ 真实监测数据</span>
</header>
<div id="chat"></div>
<div id="bar">
  <div class="chips">
    <span class="chip" onclick="fill(this)">找出2024年PM2.5年均浓度最高的5个城市，画出月度变化曲线</span>
    <span class="chip" onclick="fill(this)">筛选PM2.5超标记录，统计各省份超标天数占比</span>
    <span class="chip" onclick="fill(this)">统计北京市夏季臭氧O3超标总天数</span>
    <span class="chip" onclick="fill(this)">北京2024年O3有没有异常跳变</span>
    <span class="chip" onclick="fill(this)">对比2024年南北方城市PM2.5污染差异</span>
  </div>
  <div class="inner">
    <input id="q" placeholder="输入自然语言分析问题，回车发送…" onkeydown="if(event.key==='Enter')send()">
    <button id="btn" onclick="send()">分析</button>
  </div>
</div>
<script>
const chat = document.getElementById('chat');
function fill(el){ document.getElementById('q').value = el.textContent; document.getElementById('q').focus(); }
function addUser(text){
  const d = document.createElement('div');
  d.className = 'msg user';
  d.innerHTML = '<div class="role">你</div><div class="bubble"></div>';
  d.querySelector('.bubble').textContent = text;
  chat.appendChild(d); chat.scrollTop = chat.scrollHeight;
}
function addAgentShell(){
  const d = document.createElement('div');
  d.className = 'msg agent';
  d.innerHTML = '<div class="role">Agent（ReAct）</div><div class="bubble"><span class="loading">正在思考并调度工具…</span></div>';
  chat.appendChild(d); chat.scrollTop = chat.scrollHeight;
  return d.querySelector('.bubble');
}
async function send(){
  const inp = document.getElementById('q');
  const q = inp.value.trim();
  if(!q) return;
  inp.value = '';
  document.getElementById('btn').disabled = true;
  addUser(q);
  const shell = addAgentShell();
  try{
    const r = await fetch('/api/ask', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({question: q})});
    const data = await r.json();
    if(!data.ok){ shell.innerHTML = '<span class="loading">出错了：' + (data.error||'未知错误') + '</span>'; return; }
    let html = '<details open><summary>▸ Agent 执行轨迹（ReAct: Thought → Action → Observation）</summary>'
      + '<pre class="trace"></pre></details>'
      + '<div class="concl"></div>'
      + '<div class="tbl"></div>';
    if(data.chart_url) html += '<img class="chart" src="' + data.chart_url + '" alt="chart">';
    html += '<div class="soul"></div>';
    shell.innerHTML = html;
    shell.querySelector('pre.trace').textContent = data.trace;
    shell.querySelector('.concl').textContent = data.concl;
    if(data.table_html) shell.querySelector('.tbl').innerHTML = data.table_html;
    shell.querySelector('.soul').textContent = '耗时 ' + data.seconds + 's ｜ ' + data.intent;
  }catch(e){
    shell.innerHTML = '<span class="loading">请求失败：' + e + '</span>';
  }finally{
    document.getElementById('btn').disabled = false;
    chat.scrollTop = chat.scrollHeight;
  }
}
</script>
</body></html>"""


class AppState:
    """全局单例：数据集与 Agent 只加载一次，请求间共享。"""

    def __init__(self):
        years = discover_years()
        if not years:
            raise SystemExit("未找到数据文件，请先运行 scripts/prepare_data.py")
        self.year = max(years)
        df, report = load_dataset(config.data_file(self.year))
        self.clean_report = report
        self.agent = ReActAgent(df, report, Memory(), self.year)
        self.lock = threading.Lock()
        print(f"[web] 数据就绪：{report['清洗后行数']} 行 / {report['城市数']} 城市 / {report['日期范围']}")


STATE = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # 静默请求日志
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, text):
        body = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        if parsed.path in ("/", "/index.html"):
            self._html(INDEX_HTML)
        elif parsed.path == "/chart":
            name = os.path.basename(query.get("name", [""])[0])
            fp = os.path.join(config.CHART_DIR, name)
            if not name or not os.path.isfile(fp):
                self.send_error(404)
                return
            with open(fp, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/api/ask":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            question = (payload.get("question") or "").strip()
            if not question:
                self._json({"ok": False, "error": "问题不能为空"})
                return
            with STATE.lock:
                result = STATE.agent.run(question)
            concl = build_text_report(result["plan"], result)
            chart_url = ""
            if result.get("图表路径"):
                chart_url = "/chart?name=" + os.path.basename(result["图表路径"])
            self._json({
                "ok": True,
                "trace": result["trace"],
                "concl": concl,
                "table_html": result.get("表格HTML", ""),
                "chart_url": chart_url,
                "seconds": result["seconds"],
                "intent": result["plan"].intent,
            })
        except Exception as e:
            self._json({"ok": False, "error": str(e)})


def urllib_parse(raw):
    import urllib.parse as up
    p = up.urlparse(raw)
    q = up.parse_qs(p.query)
    return type("P", (), {"path": p.path, "query": q})()


def main():
    global STATE
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8611)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    STATE = AppState()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"[web] 网页版已启动：{url} （Ctrl+C 停止）")
    if not args.no_open:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    server.serve_forever()


if __name__ == "__main__":
    main()
