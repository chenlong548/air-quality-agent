# -*- coding: utf-8 -*-
"""网页版交互界面（零第三方依赖，Python 标准库实现）。

启动：
    python web.py                 # 默认 8611 端口，自动打开浏览器
    python web.py --port 9000 --no-open
    python web.py --interval 1.5  # 调整前端步进展示间隔（秒，默认 1.0）

然后浏览器访问 http://127.0.0.1:8611

流式架构（SSE + 消息队列节流）：
    前端 EventSource 连接 GET /api/stream?q=...&sid=...
    → Agent 在后台线程运行，每一步（Thought/Action/Observation）事件写入
      queue.Queue（消息队列缓冲）；
    → SSE 生成器按单调时钟节流：相邻两个事件间隔不足 --interval 秒时
      sleep 补齐，再依次推送，保证前端能看清每一步调度；
    → 前端逐条渲染调用链，图表、结论最后到位。

并发模型（每个请求独立 Agent，无全局串行锁）：
    数据集在 AppState 中只加载一次且全程只读；每次提问现场创建独立的
    ReActAgent 实例（中间态 _ctx 不共享），多人同时提问并行执行；
    多轮记忆按 sid（sessionStorage 生成）隔离，支持「那上海呢」式追问，
    且不同会话的记忆互不污染。
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
:root{--bg:#05080f;--panel:#0d1526;--panel2:#101b31;--inset:#080d18;--line:#1c2a4a;--line-hi:#2b4166;--txt:#d7e3f4;--dim:#8ba0bf;--faint:#526685;--cyan:#22d3ee;--cyan-hi:#67e8f9;--teal:#2dd4bf;--violet:#818cf8;--amber:#fbbf24;--rose:#fb7185;--green:#34d399;--slate:#94a3b8;--mono:"Cascadia Code","JetBrains Mono","Cascadia Mono",Consolas,"Liberation Mono",monospace;--sans:"Microsoft YaHei","PingFang SC","Noto Sans SC",sans-serif}
*{box-sizing:border-box}
html,body{height:100%}
body{font-family:var(--sans);color:var(--txt);margin:0;height:100vh;display:flex;flex-direction:column;background:radial-gradient(1200px 520px at 50% -220px,rgba(34,211,238,.10),transparent 60%),radial-gradient(900px 420px at 92% -120px,rgba(129,140,248,.07),transparent 60%),linear-gradient(rgba(56,189,248,.042) 1px,transparent 1px),linear-gradient(90deg,rgba(56,189,248,.042) 1px,transparent 1px),var(--bg);background-size:auto,auto,44px 44px,44px 44px,auto}
body::after{content:"";position:fixed;inset:0;pointer-events:none;z-index:0;background:repeating-linear-gradient(0deg,rgba(148,163,184,.026) 0 1px,transparent 1px 3px)}
header{position:relative;z-index:1;display:flex;align-items:center;gap:14px;padding:12px 22px;background:linear-gradient(180deg,rgba(13,21,38,.94),rgba(13,21,38,.74));border-bottom:1px solid var(--line);box-shadow:0 1px 0 rgba(34,211,238,.16),0 10px 28px rgba(2,6,15,.55)}
header .logo{width:36px;height:36px;flex:none;display:flex;align-items:center;justify-content:center;font-family:var(--mono);font-weight:700;font-size:13px;letter-spacing:.04em;color:#031a24;background:linear-gradient(135deg,var(--cyan-hi),var(--cyan));clip-path:polygon(50% 0,100% 25%,100% 75%,50% 100%,0 75%,0 25%);box-shadow:0 0 18px rgba(34,211,238,.55);animation:logoPulse 2.8s ease-in-out infinite}
@keyframes logoPulse{0%,100%{box-shadow:0 0 12px rgba(34,211,238,.40)}50%{box-shadow:0 0 28px rgba(34,211,238,.85)}}
header .titles{display:flex;flex-direction:column;gap:3px;min-width:0}
header h1{font-size:16px;margin:0;letter-spacing:.05em;font-weight:600}
header .sub{font-family:var(--mono);font-size:10px;letter-spacing:.3em;color:var(--cyan);text-transform:uppercase;opacity:.85}
header .meta{margin-left:auto;font-family:var(--mono);font-size:11px;color:var(--dim);letter-spacing:.03em;text-align:right;line-height:1.5}
#meta::before{content:"";display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 9px var(--green);margin-right:8px;vertical-align:1px;animation:blink 1.8s ease-in-out infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.25}}
.strip{position:relative;z-index:1;height:3px;overflow:hidden;background:var(--line)}
.strip::before{content:"";position:absolute;top:0;bottom:0;left:0;width:38%;background:linear-gradient(90deg,transparent,rgba(34,211,238,.7),transparent);animation:scan 3.4s linear infinite}
@keyframes scan{from{transform:translateX(-100%)}to{transform:translateX(300%)}}
#chat{position:relative;z-index:1;flex:1;overflow-y:auto;padding:22px 20px 26px;display:flex;flex-direction:column;gap:14px;scrollbar-width:thin;scrollbar-color:#1e2f52 transparent}
#chat::-webkit-scrollbar{width:10px}
#chat::-webkit-scrollbar-thumb{background:#1e2f52;border-radius:6px}
#chat::-webkit-scrollbar-track{background:transparent}
.msg{max-width:900px;width:100%;margin:0 auto}
.msg.user{display:flex;flex-direction:column;align-items:flex-end}
.bubble{position:relative;background-color:var(--panel);background-image:linear-gradient(180deg,rgba(34,211,238,.045),transparent 45%);border:1px solid var(--line);border-radius:6px;padding:14px 16px;box-shadow:0 8px 26px rgba(2,6,15,.45)}
.msg.user .bubble{max-width:82%;background-color:#0e2233;background-image:linear-gradient(180deg,rgba(34,211,238,.16),rgba(34,211,238,.05));border-color:rgba(34,211,238,.42);box-shadow:0 0 20px rgba(34,211,238,.12)}
.role{font-family:var(--mono);font-size:10px;letter-spacing:.22em;color:var(--faint);margin-bottom:7px;text-transform:uppercase}
.msg.user .role{color:var(--cyan)}
.plan-card{position:relative;background:rgba(34,211,238,.05);border:1px solid rgba(34,211,238,.26);border-left:3px solid var(--cyan);border-radius:4px;padding:10px 14px 12px;font-size:12.5px;margin-bottom:12px;animation:fadeUp .35s ease both}
.plan-card ol{margin:8px 0 0;padding-left:0;list-style:none;counter-reset:step}
.plan-card li{position:relative;margin:6px 0;color:var(--dim);padding-left:36px;counter-increment:step;animation:fadeUp .3s ease both}
.plan-card li::before{content:counter(step,decimal-leading-zero);position:absolute;left:0;top:2px;font-family:var(--mono);font-size:10px;color:var(--cyan);border:1px solid rgba(34,211,238,.4);border-radius:2px;padding:1px 4px;letter-spacing:.05em}
.plan-card li b{color:var(--cyan-hi);font-family:var(--mono);font-weight:600;font-size:12px}
.plan-card .src{font-family:var(--mono);color:var(--cyan);font-weight:600}
.step{position:relative;border:1px solid var(--line);border-radius:6px;padding:10px 14px 12px;margin:10px 0;background:rgba(16,27,49,.55);animation:stepIn .4s cubic-bezier(.2,.7,.3,1) both}
@keyframes stepIn{from{opacity:0;transform:translateY(10px);border-color:rgba(34,211,238,.55)}to{opacity:1;transform:none;border-color:var(--line)}}
@keyframes fadeUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
.step::before,.step::after{content:"";position:absolute;width:10px;height:10px;pointer-events:none;opacity:.85}
.step::before{top:-1px;left:-1px;border-top:2px solid var(--cyan);border-left:2px solid var(--cyan)}
.step::after{bottom:-1px;right:-1px;border-bottom:2px solid var(--cyan);border-right:2px solid var(--cyan)}
.step-head{display:flex;align-items:center;gap:10px;margin-bottom:8px}
.badge{font-family:var(--mono);font-size:11px;font-weight:600;letter-spacing:.05em;border-radius:3px;padding:3px 9px;color:var(--dim);background:#1c2a4a}
.b-load_data{background:#94a3b8;color:#0b1220;box-shadow:0 0 12px rgba(148,163,184,.45)}
.b-aggregate{background:#22d3ee;color:#031a24;box-shadow:0 0 12px rgba(34,211,238,.5)}
.b-trend{background:#818cf8;color:#0b1020;box-shadow:0 0 12px rgba(129,140,248,.5)}
.b-season_exceed{background:#2dd4bf;color:#032224;box-shadow:0 0 12px rgba(45,212,191,.5)}
.b-anomaly{background:#fb7185;color:#2b060f;box-shadow:0 0 12px rgba(251,113,133,.5)}
.b-check_standard{background:#a78bfa;color:#170a2e;box-shadow:0 0 12px rgba(167,139,250,.5)}
.b-plot_chart{background:#fbbf24;color:#2b1e03;box-shadow:0 0 12px rgba(251,191,36,.5)}
.b-validate{background:#34d399;color:#032418;box-shadow:0 0 12px rgba(52,211,153,.5)}
.step-no{margin-left:auto;font-family:var(--mono);font-size:10px;letter-spacing:.18em;color:var(--faint)}
.args{font-family:var(--mono);font-size:11px;margin:2px 0 6px;display:flex;flex-wrap:wrap;gap:5px}
.args span{background:rgba(34,211,238,.07);border:1px solid rgba(34,211,238,.18);border-radius:3px;padding:2px 7px;color:var(--dim)}
.thought{font-size:12.5px;color:var(--dim);margin:2px 0 5px;padding-left:2px}
.obs{font-family:var(--mono);font-size:11.5px;line-height:1.65;white-space:pre-wrap;background:var(--inset);border:1px solid var(--line);border-left:2px solid rgba(34,211,238,.35);border-radius:4px;padding:9px 12px;margin:6px 0 0;max-height:240px;overflow:auto;color:#b9cbe4}
.obs.err{border-color:rgba(251,113,133,.5);border-left-color:var(--rose);background:rgba(251,113,133,.07);color:#fecdd3}
img.chart{max-width:100%;display:block;border:1px solid var(--line-hi);border-radius:6px;margin-top:10px;box-shadow:0 0 26px rgba(34,211,238,.10);animation:fadeUp .45s ease both}
table{border-collapse:collapse;width:100%;font-size:12.5px;margin:10px 0}
th,td{border:1px solid var(--line);padding:6px 10px;text-align:left}
th{background:#0f1b31;color:var(--cyan-hi);font-family:var(--mono);font-size:11px;letter-spacing:.06em;font-weight:600}
tr:nth-child(even) td{background:rgba(34,211,238,.028)}
.concl{font-size:13.5px;white-space:pre-wrap;line-height:1.85;margin-top:12px}
.soul{position:relative;margin-top:12px;background:rgba(52,211,153,.07);border:1px solid rgba(52,211,153,.3);border-radius:4px;padding:9px 14px;font-family:var(--mono);font-size:11.5px;color:#a7f3d0;letter-spacing:.02em}
.soul::before{content:"▸";color:var(--green);margin-right:9px}
.final{margin-top:12px;border-top:1px dashed var(--line-hi);padding-top:10px}
.loading{font-family:var(--mono);color:var(--cyan);font-size:12.5px;letter-spacing:.04em;animation:pulse 1.2s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:.5}50%{opacity:1}}
#bar{position:relative;z-index:1;background:linear-gradient(0deg,rgba(13,21,38,.96),rgba(13,21,38,.8));border-top:1px solid var(--line);padding:12px 20px 16px;box-shadow:0 -10px 28px rgba(2,6,15,.55)}
#bar .inner{max-width:900px;margin:0 auto;display:flex;gap:10px}
#q{flex:1;background:var(--inset);border:1px solid var(--line-hi);border-radius:4px;padding:11px 14px;font-size:14px;font-family:var(--sans);color:var(--txt);outline:none;transition:border-color .2s,box-shadow .2s}
#q::placeholder{color:var(--faint)}
#q:focus{border-color:var(--cyan);box-shadow:0 0 0 1px rgba(34,211,238,.35),0 0 18px rgba(34,211,238,.18)}
button{background:linear-gradient(135deg,var(--cyan-hi),var(--cyan));color:#031a24;border:none;border-radius:4px;padding:11px 24px;font-size:14px;font-family:var(--mono);letter-spacing:.12em;cursor:pointer;box-shadow:0 0 16px rgba(34,211,238,.35);transition:box-shadow .2s,transform .1s}
button:hover{box-shadow:0 0 28px rgba(34,211,238,.6)}
button:active{transform:translateY(1px)}
button:disabled{background:#1c2a4a;color:var(--faint);box-shadow:none;cursor:not-allowed}
.chips{max-width:900px;margin:0 auto 12px;display:flex;gap:8px;flex-wrap:wrap}
.chip{font-family:var(--mono);font-size:11.5px;border:1px solid var(--line-hi);border-radius:2px;padding:6px 13px;cursor:pointer;background:rgba(16,27,49,.6);color:var(--dim);clip-path:polygon(8px 0,100% 0,100% calc(100% - 8px),calc(100% - 8px) 100%,0 100%,0 8px);transition:border-color .18s,color .18s,background .18s,transform .18s}
.chip:hover{border-color:var(--cyan);color:var(--cyan-hi);background:rgba(34,211,238,.08);transform:translateY(-1px)}
/* ---- 遥测终端精修：时钟 / 开机引导 / 入场编排 / 结果标签 / 按钮微光 ---- */
.clock{font-family:var(--mono);font-size:11px;color:var(--dim);letter-spacing:.06em;padding-left:14px;margin-left:2px;border-left:1px solid var(--line);white-space:nowrap;align-self:center}
.boot{margin:auto;text-align:center;max-width:580px;padding:30px;animation:fadeUp .6s ease both}
.boot-mark{width:54px;height:54px;margin:0 auto 18px;display:flex;align-items:center;justify-content:center;font-family:var(--mono);font-weight:700;font-size:17px;letter-spacing:.04em;color:#031a24;background:linear-gradient(135deg,var(--cyan-hi),var(--cyan));clip-path:polygon(50% 0,100% 25%,100% 75%,50% 100%,0 75%,0 25%);box-shadow:0 0 26px rgba(34,211,238,.5);animation:logoPulse 2.8s ease-in-out infinite}
.boot-title{font-family:var(--mono);font-size:15px;letter-spacing:.34em;text-indent:.34em;color:var(--cyan-hi);text-shadow:0 0 18px rgba(34,211,238,.45)}
.boot-sub{font-size:12.5px;color:var(--dim);margin-top:9px;letter-spacing:.1em}
.boot-log{margin:24px auto 0;text-align:left;display:inline-block;font-family:var(--mono);font-size:11.5px;line-height:2.15;color:var(--dim)}
.boot-log>div{opacity:0;animation:fadeUp .5s ease both}
.boot-log>div:nth-child(1){animation-delay:.15s}
.boot-log>div:nth-child(2){animation-delay:.32s}
.boot-log>div:nth-child(3){animation-delay:.49s}
.boot-log>div:nth-child(4){animation-delay:.66s}
.boot-log .ok{color:var(--green)}
.boot-log .eng{color:var(--cyan-hi)}
.boot-hint{margin-top:22px;font-size:12px;color:var(--faint);letter-spacing:.05em;opacity:0;animation:fadeUp .5s ease .85s both}
.msg.user{animation:msgInR .35s cubic-bezier(.2,.7,.3,1) both}
.msg.agent{animation:msgInL .35s cubic-bezier(.2,.7,.3,1) both}
@keyframes msgInR{from{opacity:0;transform:translateX(18px)}to{opacity:1;transform:none}}
@keyframes msgInL{from{opacity:0;transform:translateX(-18px)}to{opacity:1;transform:none}}
.loading::after{content:"▊";margin-left:3px;color:var(--cyan-hi);animation:blink 1s steps(1) infinite}
.final::before{content:"◈ ANALYSIS RESULT · 分析结果";display:block;font-family:var(--mono);font-size:10px;letter-spacing:.26em;color:var(--faint);margin-bottom:9px}
.rblock{margin:12px 0}
.rblock-title{font-family:var(--mono);font-size:11px;letter-spacing:.08em;color:var(--cyan);margin-bottom:6px}
button{position:relative;overflow:hidden}
button::after{content:"";position:absolute;inset:0;background:linear-gradient(105deg,transparent 40%,rgba(255,255,255,.42) 50%,transparent 60%);transform:translateX(-130%);transition:transform .65s ease;pointer-events:none}
button:hover::after{transform:translateX(130%)}
@media (max-width:680px){
  header .sub{display:none}
  header .meta{display:none}
  #chat{padding:16px 12px 20px}
  .boot-title{font-size:13px;letter-spacing:.18em;text-indent:.18em}
  .boot-log{font-size:10.5px;line-height:2}
  .step{padding:9px 11px 10px}
  .chip{font-size:11px;padding:5px 10px}
  #bar .inner{gap:8px}
  button{padding:11px 16px}
}
</style></head><body>
<header>
  <div class="logo">AQ</div>
  <div class="titles">
    <h1>空气质量时序分析 Agent</h1>
    <div class="sub">Atmospheric Telemetry Console</div>
  </div>
  <span class="meta" id="meta">ReAct 智能体 ｜ GB 3095-2012 国标研判 ｜ 真实监测数据 ｜ 流式调度展示</span><span class="clock" id="clock"></span>
</header>
<div class="strip"></div>
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
// ---- 会话标识：同一标签页共享多轮记忆，不同标签页/浏览器互不串扰 ----
let SID = sessionStorage.getItem('aq_sid');
if(!SID){ SID = 's' + Date.now().toString(36) + Math.random().toString(36).slice(2,8);
  sessionStorage.setItem('aq_sid', SID); }
// ---- 遥测时钟 ----
function tick(){ const d=new Date(), p=n=>String(n).padStart(2,'0');
  document.getElementById('clock').textContent =
    d.getFullYear()+'-'+p(d.getMonth()+1)+'-'+p(d.getDate())+' '+p(d.getHours())+':'+p(d.getMinutes())+':'+p(d.getSeconds()); }
tick(); setInterval(tick,1000);
// ---- 开机引导屏（首次发送时移除）----
(function(){ const d=document.createElement('div'); d.className='boot'; d.id='boot';
  d.innerHTML='<div class="boot-mark">AQ</div>'
    + '<div class="boot-title">ATMOSPHERIC TELEMETRY CONSOLE</div>'
    + '<div class="boot-sub">空气质量时序分析 Agent · 深空遥测终端</div>'
    + '<div class="boot-log">'
    + '<div>▸ 数据链路 CN-AIR 年度监测数据集 <span class="ok">就绪</span></div>'
    + '<div>▸ 判定标准 GB 3095-2012 二级 <span class="ok">已装载</span></div>'
    + '<div>▸ 规划引擎 <span class="eng" id="boot-engine">···</span> <span class="ok">在线</span></div>'
    + '<div>▸ 流式调度通道 SSE <span class="ok">已连接</span></div></div>'
    + '<div class="boot-hint">在下方输入自然语言问题，Agent 将现场生成工具调用链并流式执行</div>';
  chat.appendChild(d); })();
const TOOL_ICON = {load_data:'📂',aggregate:'📊',trend:'📈',season_exceed:'🗓️',
  anomaly:'🚨',check_standard:'📏',plot_chart:'🎨',validate:'✅'};
fetch('/api/status').then(r=>r.json()).then(s=>{
  document.getElementById('meta').textContent =
    'ReAct 智能体 ｜ GB 3095-2012 国标研判 ｜ 真实监测数据 ｜ 流式调度展示 ｜ 规划：'
    + (s.llm ? ('LLM（' + s.model + '）') : '本地规则');
  const eng=document.getElementById('boot-engine');
  if(eng) eng.textContent = s.llm ? s.model : '本地规则';
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
  const boot=document.getElementById('boot'); if(boot) boot.remove();
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
  es = new EventSource('/api/stream?q=' + encodeURIComponent(q) + '&sid=' + SID);
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
      pc.querySelectorAll('li').forEach((it,i)=>{it.style.animationDelay=(i*80)+'ms'});
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
    """全局共享态：数据集只加载一次（全程只读），会话记忆按 sid 隔离。

    并发设计（不足③）：
      - 数据集/清洗报告不可变，多线程只读groupby 安全；
      - 每个请求现场创建独立的 ReActAgent 实例（Agent 只在一次 run 内持有
        _ctx 中间态），因此多人同时提问并行执行、互不排队；
      - 多轮记忆（「那上海呢」式追问）按前端 sessionStorage 生成的 sid 归属，
        不同浏览器/标签页的记忆互不串扰；
      - 会话表有容量上限（LRU 淘汰），防止长期运行内存泄漏。
    """

    MAX_SESSIONS = 128

    def __init__(self):
        years = discover_years()
        if not years:
            raise SystemExit("未找到数据文件，请先运行 scripts/prepare_data.py")
        self.year = max(years)
        df, report = load_dataset(config.data_file(self.year))
        self.clean_report = report
        self.df = df
        self._sessions = {}          # sid -> Memory（dict 保序 = 天然 LRU）
        self._sessions_lock = threading.Lock()
        print(f"[web] 数据就绪：{report['清洗后行数']} 行 / {report['城市数']} 城市 / {report['日期范围']}")

    def memory_for(self, sid):
        """按会话取/建记忆（并发安全）；sid 为空时返回一次性记忆。"""
        if not sid:
            return Memory()
        with self._sessions_lock:
            mem = self._sessions.pop(sid, None)   # 取出后放回尾部 = LRU 刷新
            if mem is None:
                while len(self._sessions) >= self.MAX_SESSIONS:
                    self._sessions.pop(next(iter(self._sessions)))
                mem = Memory()
            self._sessions[sid] = mem
            return mem

    def new_agent(self, sid=""):
        """为单个请求创建独立 Agent（无全局锁，天然支持并发）。"""
        return ReActAgent(self.df, self.clean_report,
                          self.memory_for(sid), self.year)


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
            self._sse_stream(query.get("q", [""])[0], query.get("sid", [""])[0])
        elif parsed.path == "/api/status":
            llm = bool(os.environ.get("AQ_LLM_BASE_URL") and os.environ.get("AQ_LLM_API_KEY"))
            self._json({"llm": llm, "model": os.environ.get("AQ_LLM_MODEL", "")})
        else:
            self.send_error(404)

    # ---------- SSE 流式端点 ----------

    def _sse_stream(self, question, sid=""):
        """Agent 后台运行，事件入消息队列，按 ≥MIN_GAP 秒节流依次推送。

        每个连接独立 Agent 实例 + 按 sid 隔离的记忆：多人同时提问并行执行。
        """
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
                STATE.new_agent(sid).run(question, on_event=evq.put)
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
            result = STATE.new_agent(payload.get("sid") or "").run(question)
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
