# -*- coding: utf-8 -*-
"""网页版并发验证：3 个问题同时经 SSE 发出，验证并行执行 + 记忆隔离。"""
import json
import sys
import threading
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "http://127.0.0.1:8631"
QUESTIONS = [
    ("sessA", "2024年北京PM2.5年均浓度是多少"),
    ("sessB", "2024年上海PM2.5年均浓度是多少"),
    ("sessC", "2024年广州PM2.5年均浓度是多少"),
]


def run_sse(sid, q, out):
    url = f"{BASE}/api/stream?q={urllib.parse.quote(q)}&sid={sid}"
    t0 = time.time()
    done_at = None
    concl = ""
    events = []
    try:
        with urllib.request.urlopen(url, timeout=180) as resp:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                ev = json.loads(line[6:])
                events.append(ev["kind"])
                if ev["kind"] == "done":
                    concl = ev.get("concl", "")
                    done_at = time.time() - t0
                if ev["kind"] == "closed":
                    break
    except Exception as e:
        out.append((sid, False, f"{type(e).__name__}: {e}", 0, ""))
        return
    out.append((sid, True, concl, done_at or time.time() - t0, ",".join(events)))


import urllib.parse

out = []
threads = []
t_start = time.time()
for sid, q in QUESTIONS:
    t = threading.Thread(target=run_sse, args=(sid, q, out))
    t.start()
    threads.append(t)
for t in threads:
    t.join()
wall = time.time() - t_start

print(f"# 3 个并发 SSE 请求，总耗时(墙钟) = {wall:.1f}s")
for sid, ok, concl, secs, kinds in sorted(out, key=lambda x: x[3]):
    print(f"  {sid}: ok={ok} 单请求耗时={secs:.1f}s 事件数={len(kinds.split(','))}")
    print(f"    结论前80字: {concl[:80].replace(chr(10), ' / ')}")

# 判定：并发总耗时应明显小于单请求耗时之和（否则就是串行）
sum_secs = sum(s for _, _, _, s, _ in out)
print(f"# 单请求耗时之和 = {sum_secs:.1f}s；墙钟 = {wall:.1f}s；"
      f"{'✅ 并行生效' if wall < sum_secs * 0.75 else '⚠️ 疑似仍串行'}")

# 判定：三个结论应分别对应各自城市，无记忆串扰
cities = [c.split("：")[0][-2:] if "：" in c else "" for _, _, c, _, _ in out]
ok_isolated = all(("北京" in c and "上海" not in c) or ("上海" in c and "北京" not in c)
                  or ("广州" in c and "北京" not in c and "上海" not in c)
                  for _, _, c, _, _ in out)
print(f"# 记忆隔离: {'✅ 各会话结论未串扰' if ok_isolated else '⚠️ 存在串扰，需检查'}")
