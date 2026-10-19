# -*- coding: utf-8 -*-
"""网页版交互界面（零第三方依赖，Python 标准库实现）。

启动：
    python web.py                 # 默认 8611 端口，自动打开浏览器
    python web.py --port 9000 --no-open
    python web.py --interval 1.5  # 调整前端步进展示间隔（秒，默认 1.0）

然后浏览器访问 http://127.0.0.1:8611

流式架构（SSE + 消息队列节流）：
    前端 EventSource 连接 GET /api/stream?q=...
    → Agent 在后台线程运行，每一步（Thought/Action/Observation）事件写入
      queue.Queue（消息队列缓冲）；
    → SSE 生成器按单调时钟节流：相邻两个事件间隔不足 --interval 秒时
      sleep 补齐，再依次推送，保证前端能看清每一步调度；
    → 前端逐条渲染调用链，图表、结论最后到位。
"""
import argparse
import json
import os
import queue
import sys
import threading
import time
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
.plan-card{background:#f0f7ff;border:1px solid #cfe3ff;border-radius:8px;padding:10px 14px;font-size:12.5px;margin-bottom:10px}
.plan-card ol{margin:6px 0 0;padding-left:20px}
.plan-card li{margin:3px 0;color:#374151}
.plan-card .src{font-weight:600;color:#1d4ed8}
.step{border:1px solid #e5e7eb;border-radius:10px;padding:10px 12px;margin:8px 0;background:#fcfdfe;animation:fadeUp .35s ease both}
@keyframes fadeUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
.step-head{display:flex;align-items:center;gap:8px;margin-bottom:6px}
.badge{font-size:11.5px;font-weight:600;border-radius:6px;padding:2px 8px;color:#fff}
.b-load_data{background:#6b7280}.b-aggregate{background:#3b7dd8}.b-trend{background:#6366f1}
.b-season_exceed{background:#0d9488}.b-anomaly{background:#dc2626}.b-check_standard{background:#7c3aed}
.b-plot_chart{background:#d97706}.b-validate{background:#059669}
.step-no{font-size:11px;color:#9ca3af;margin-left:auto}
.args{font-size:11px;color:#6b7280;margin:2px 0 4px}
.args span{background:#eef2f7;border-radius:4px;padding:1px 6px;margin-right:4px}
.thought{font-size:12.5px;color:#6b7280;margin:2px 0}
.obs{font-size:12px;white-space:pre-wrap;background:#f8fafc;border:1px solid #e5e7eb;border-radius:6px;padding:8px 10px;margin:4px 0 0;max-height:220px;overflow:auto}
.obs.err{border-color:#fecaca;background:#fef2f2}
img.chart{max-width:100%;border:1px solid #e5e7eb;border-radius:8px;margin-top:8px;animation:fadeUp .4s ease both}
table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}
th,td{border:1px solid #e5e7eb;padding:5px 9px;text-align:left}
th{background:#eef2f7}
.concl{font-size:13.5px;white-space:pre-wrap;line-height:1.7;margin-top:10px}
.soul{margin-top:10px;background:#f0faf3;border:1px solid #c9e8d4;border-radius:8px;padding:8px 12px;font-size:12.5px}
.final{margin-top:10px;border-top:1px dashed #e5e7eb;padding-top:8px}
.loading{color:#6b7280;font-size:13px}
#bar{background:#fff;border-top:1px solid #e5e7eb;padding:12px 20px}
#bar .inner{max-width:880px;margin:0 auto;display:flex;gap:10px}
#q{flex:1;border:1px solid #d1d5db;border-radius:10px;padding:10px 14px;font-size:14px;outline:none}
#q:focus{border-color:#3b7dd8}
button{background:#3b7dd8;color:#fff;border:none;border-radius:10px;padding:10px 20px;font-size:14px;cursor:pointer}
button:disabled{background:#9db9dd}
.chips{max-width:880px;margin:0 auto 10px;display:flex;gap:8px;flex-wrap:wrap}
.chip{font-size:12px;border:1px solid #d1d5db;border-radius:999px;padding:5px 12px;cursor:pointer;background:#fff;color:#374151}
.chip:hover{border-color:#3b7dd8;color:#3b7dd8}
</style></head><body>
<header>
  <div class="logo">AQ</div>
  <h1>空气质量时序分析 Agent</h1>
  <span class="meta" id="meta">ReAct 智能体 ｜ GB 3095-2012 国标研判 ｜ 真实监测数据 ｜ 流式调度展示</span>
</header>
<div id="chat"></div>
<div id="bar">
  <div class="chips">
    <span class="chip" onclick="fill(this)">找出2024年PM2.5年均浓度最高的5个城市，画出月度变化曲线</span>
    <span class="chip" onclick="fill(this)">筛选PM2.5超标记录，统计各省份超标天数占比</span>
    <span class="chip" onclick="fill(this)">统计北京市夏季臭氧O3超标总天数</span>
    <span class="chip" onclick="fill(this)">北京2024年O3有没有异常跳变</span>
    <span class="chip" onclick="fill(this)">对比2024年南北方城市PM2.5污染差异</span>
    <span class="chip" onclick="fill(this)">新疆和河北哪个省份PM10污染更严重</span>
  </div>
  <div class="inner">
    <input id="q" placeholder="输入自然语言分析问题（任意问法），回车发送…" onkeydown="if(event.key==='Enter')send()">
    <button id="btn" onclick="send()">分析</button>
  </div>
</div>
<script>
const chat = document.getElementById('chat');
const TOOL_ICON = {load_data:'📂',aggregate:'📊',trend:'📈',season_exceed:'🗓️',
  anomaly:'🚨',check_standard:'📏',plot_chart:'🎨',validate:'✅'};
fetch('/api/status').then(r=>r.json()).then(s=>{
  document.getElementById('meta').textContent =
    'ReAct 智能体 ｜ GB 3095-2012 国标研判 ｜ 真实监测数据 ｜ 流式调度展示 ｜ 规划：'
    + (s.llm ? ('LLM（' + s.model + '）') : '本地规则');
}).catch(()=>{});
function fill(el){ document.getElementById('q').value = el.textContent; document.getElementById('q').focus(); }
function esc(s){ const d=document.createElement('div'); d.textContent=s==null?'':String(s); return d.innerHTML; }
function bottom(){ chat.scrollTop = chat.scrollHeight; }
function addUser(text){
  const d = document.createElement('div');
  d.className = 'msg user';
  d.innerHTML = '<div class="role">你</div><div class="bubble"></div>';
  d.querySelector('.bubble').textContent = text;
  chat.appendChild(d); bottom();
}
function addAgentShell(){
  const d = document.createElement('div');
  d.className = 'msg agent';
  d.innerHTML = '<div class="role">Agent（ReAct 流式调度）</div><div class="bubble">'
    + '<span class="loading" id="planning">🧠 Agent 正在理解问题并生成工具调用链…</span>'
    + '<div class="steps"></div><div class="final"></div></div>';
  chat.appendChild(d); bottom();
  return d.querySelector('.bubble');
}
let es = null;
function send(){
  const inp = document.getElementById('q');
  const q = inp.value.trim();
  if(!q) return;
  inp.value = '';
  document.getElementById('btn').disabled = true;
  addUser(q);
  const shell = addAgentShell();
  const stepsEl = shell.querySelector('.steps');
  const finalEl = shell.querySelector('.final');
  let cur = null, curNo = 0, chartCount = 0, tablePending = '';

  function newStep(no, total){
    const s = document.createElement('div');
    s.className = 'step';
    s.innerHTML = '<div class="step-head"><span class="badge">⏳</span>'
      + '<span class="step-no">步骤 ' + no + '/' + total + '</span></div>'
      + '<div class="args"></div><div class="thought"></div><pre class="obs"></pre>';
    stepsEl.appendChild(s); bottom();
    cur = s; curNo = no;
    return s;
  }
  es = new EventSource('/api/stream?q=' + encodeURIComponent(q));
  es.onmessage = (e) => {
    let ev; try { ev = JSON.parse(e.data); } catch(err){ return; }
    const planning = document.getElementById('planning');
    if(planning) planning.remove();
    if(ev.kind === 'plan'){
      const pc = document.createElement('div');
      pc.className = 'plan-card';
      let li = ev.steps.map((s,i) =>
        '<li><b>' + esc(s.tool) + '</b> — ' + esc(s.thought) + '</li>').join('');
      pc.innerHTML = '🧠 规划完成（来源：<span class="src">'
        + (ev.source === 'llm' ? 'LLM 实时生成' : '本地规则') + '</span>）'
        + '，共 <b>' + ev.total + '</b> 步工具调用链：<ol>' + li + '</ol>';
      stepsEl.appendChild(pc); bottom();
    }
    else if(ev.kind === 'thought'){
      if(!cur || curNo !== ev.step) newStep(ev.step, ev.total);
      cur.querySelector('.thought').textContent = '💭 ' + ev.text;
      bottom();
    }
    else if(ev.kind === 'action'){
      if(!cur || curNo !== ev.step) newStep(ev.step, ev.total);
      const b = cur.querySelector('.badge');
      b.className = 'badge b-' + ev.tool;
      b.textContent = (TOOL_ICON[ev.tool]||'🔧') + ' ' + ev.tool;
      const args = ev.args || {};
      cur.querySelector('.args').innerHTML = Object.keys(args).map(k =>
        '<span>' + esc(k) + '=' + esc(args[k]) + '</span>').join('');
      bottom();
    }
    else if(ev.kind === 'observation'){
      if(!cur || curNo !== ev.step) newStep(ev.step, ev.total);
      const o = cur.querySelector('.obs');
      o.textContent = '📥 ' + ev.text;
      if(ev.ok === false) o.classList.add('err');
      bottom();
    }
    else if(ev.kind === 'chart'){
      const img = document.createElement('img');
      img.className = 'chart';
      img.src = ev.url + '&t=' + Date.now();
      img.alt = ev.name;
      (cur || finalEl).appendChild(img);
      chartCount++; bottom();
    }
    else if(ev.kind === 'done'){
      finalEl.innerHTML = '<div class="concl"></div><div class="tbl"></div>'
        + '<div class="soul"></div>';
      finalEl.querySelector('.concl').textContent = ev.concl || '';
      if(tablePending) finalEl.querySelector('.tbl').innerHTML = tablePending;
      finalEl.querySelector('.soul').textContent =
        '耗时 ' + ev.seconds + 's ｜ 规划来源：' + (ev.source === 'llm' ? 'LLM' : '本地规则')
        + ' ｜ 意图：' + ev.intent;
      bottom();
    }
    else if(ev.kind === 'table'){
      const tbl = finalEl.querySelector('.tbl');
      if(tbl) tbl.innerHTML = ev.html; else tablePending = ev.html;
      bottom();
    }
    else if(ev.kind === 'error'){
      const s = document.createElement('div');
      s.className = 'step';
      s.innerHTML = '<div class="obs err"></div>';
      s.querySelector('.obs').textContent = '⚠️ ' + ev.text;
      stepsEl.appendChild(s); bottom();
    }
    if(ev.kind === 'done' || ev.kind === 'closed' || ev.kind === 'error'){
      if(es){ es.close(); es = null; }
      document.getElementById('btn').disabled = false;
      bottom();
    }
  };
  es.onerror = () => {
    if(es){ es.close(); es = null; }
    document.getElementById('btn').disabled = false;
    const planning = document.getElementById('planning');
    if(planning) planning.textContent = '连接中断，请重试';
  };
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
MIN_GAP = 1.0   # 相邻事件最小推送间隔（秒），可用 --interval 调整


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

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

    # ---------- 页面与图表 ----------

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
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        elif parsed.path == "/api/stream":
            self._sse_stream(query.get("q", [""])[0])
        elif parsed.path == "/api/status":
            llm = bool(os.environ.get("AQ_LLM_BASE_URL") and os.environ.get("AQ_LLM_API_KEY"))
            self._json({"llm": llm, "model": os.environ.get("AQ_LLM_MODEL", "")})
        else:
            self.send_error(404)

    # ---------- SSE 流式端点 ----------

    def _sse_stream(self, question):
        """Agent 后台运行，事件入消息队列，按 ≥MIN_GAP 秒节流依次推送。"""
        if not question.strip():
            self._json({"ok": False, "error": "问题不能为空"})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        evq = queue.Queue()

        def worker():
            try:
                with STATE.lock:
                    STATE.agent.run(question, on_event=evq.put)
            except Exception as e:
                evq.put({"kind": "error", "text": f"{type(e).__name__}: {e}"})
            finally:
                evq.put(None)   # 结束哨兵

        threading.Thread(target=worker, daemon=True).start()

        def send_ev(ev):
            if ev.get("kind") == "chart" and not ev.get("url"):
                ev["url"] = "/chart?name=" + str(ev.get("name", ""))
            self.wfile.write(("data: " + json.dumps(ev, ensure_ascii=False,
                                                    default=str) + "\n\n").encode("utf-8"))
            self.wfile.flush()

        try:
            self.wfile.write(b": connected\n\n")   # SSE 注释，保活
            self.wfile.flush()
            last = None
            while True:
                ev = evq.get()
                done = ev is None
                if done:
                    ev = {"kind": "closed"}
                if last is not None:               # 节流：相邻事件至少间隔 MIN_GAP 秒
                    gap = time.monotonic() - last
                    if gap < MIN_GAP:
                        time.sleep(MIN_GAP - gap)
                last = time.monotonic()
                send_ev(ev)
                if done:
                    break
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass   # 客户端断开

    # ---------- 一次性接口（兼容命令行/测试） ----------

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
                chart_url = ("/chart?name=" + os.path.basename(result["图表路径"])
                             + "&t=" + str(int(time.time() * 1000)))
            self._json({
                "ok": True,
                "trace": result["trace"],
                "concl": concl,
                "table_html": result.get("表格HTML", ""),
                "chart_url": chart_url,
                "seconds": result["seconds"],
                "intent": result["plan"].intent,
                "source": result.get("source", "rule"),
            })
        except Exception as e:
            self._json({"ok": False, "error": str(e)})


def main():
    global STATE, MIN_GAP
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8611)
    ap.add_argument("--interval", type=float, default=1.0,
                    help="前端流式展示的相邻事件最小间隔（秒）")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()
    MIN_GAP = max(0.0, args.interval)

    loaded = config.load_env_file()
    llm_on = bool(os.environ.get("AQ_LLM_BASE_URL") and os.environ.get("AQ_LLM_API_KEY"))
    if loaded:
        print(f"[web] 已从 .env 加载: {', '.join(loaded)}")
    print(f"[web] 规划器：{'LLM function calling（' + os.environ.get('AQ_LLM_MODEL', '') + '）' if llm_on else '本地规则（未配置 Key，无网可跑）'}")

    STATE = AppState()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"[web] 网页版已启动：{url} （流式间隔 {MIN_GAP}s，Ctrl+C 停止）")
    if not args.no_open:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    server.serve_forever()


if __name__ == "__main__":
    main()
