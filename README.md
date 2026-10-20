# 空气质量时序分析 Agent（环境 + 大数据）

OHCP 一小时智能编程大赛作品。自然语言提问 → **LLM 针对每个问题现生成工具调用链**（function calling 风格，任意问法）→ ReAct 引擎按链逐步调度 → 本地真实数据集分片读取清洗 → 聚合统计 → **GB 3095-2012 国标超标研判** → 文本结论 + 可视化图表 + HTML 报告。课程思政内容随分析结果自然融入，不弹窗、不说教。

## 一键启动

**网页版（推荐，浏览器聊天界面，流式展示调度链）：**

```bash
pip install -r requirements.txt
python web.py            # 默认 http://127.0.0.1:8611，自动打开浏览器
python web.py --interval 1.5   # 调整流式步进间隔（秒，默认 1.0）
```

**命令行版：**

```bash
python run.py
```

启动后输入自然语言提问即可，例如：

- `找出2024年PM2.5年均浓度最高的5个城市，画出月度变化曲线`
- `筛选PM2.5超标记录，统计各省份超标天数占比`
- `统计北京市夏季臭氧O3超标总天数`
- `北京2024年O3有没有异常跳变`
- `对比2024年南北方城市PM2.5污染差异`
- `2024年南北各自超标最严重的3个省份是哪些`（分组切片）
- `北京和上海的PM2.5污染是否相关`（相关性分析）
- `北京的PM2.5污染主要来自哪里`（来源代理分析）
- `新疆和河北哪个省份PM10污染更严重`（省份对比）

也支持单条命令直接跑：`python run.py "统计北京市夏季臭氧O3超标总天数"`

交互命令：`help` 帮助 ｜ `board` 任务看板 ｜ `mode 专业/简易` 切换输出模板 ｜ `quit` 退出。

网页版与命令行版共用同一套 Agent 内核，额外提供：快捷提问 chips、规划卡片（调用链预览）、ReAct 执行轨迹逐条流式渲染、结果表格与图表内联展示。

## Agent 调度链（核心机制）

规划器把每个问题解析为一条**工具调用链**（steps），引擎逐步执行，每步一个事件：

```
用户提问
  → 规划器生成调用链（LLM function calling / 本地规则，自动容错回退）
     例：load_data → aggregate(province, exceed_days) → check_standard
          → plot_chart(pie) → validate
  → 引擎逐步执行，每步发出 Thought / Action / Observation 事件
  → 网页版：事件入消息队列，SSE 按 ≥1s 间隔依次推送，前端逐条渲染
```

可用工具（LLM 可自由组合，参数任意）：

| 工具 | 作用 | 关键参数 |
|---|---|---|
| load_data | 分片读取并清洗数据集 | — |
| aggregate | 分组聚合 | group_by=city/province/season/north_south，metric=mean/exceed_days，pollutant（六项+AQI），cities，provinces，year，top_n（全量排名用 500），order=asc/desc（最低/最高在前），split_by=north_south（省份按南北方分组切片） |
| trend | 城市月度趋势 | cities，pollutant，year |
| season_exceed | 单城市单季节超标天数 | city，season，pollutant，year |
| anomaly | MAD 3σ 异常检测 | city，pollutant，year |
| city_correlation | 两城市日浓度 Pearson 相关（污染是否同步/区域传输） | cities，pollutant，year |
| pollution_source | 污染来源代理分析（扬尘/交通燃烧/采暖/二次生成） | city，pollutant，year |
| check_standard | 限值口径（GB 3095-2012 / AQI 按 HJ 633-2012） | pollutant，kind |
| plot_chart | 可视化（auto 按结果选图型；>40 行自动跳过绘图） | chart=auto/line/bar/pie |
| validate | 结果合理性校验（多结果块逐块校验） | — |

## 泛化能力与诚实拒答

- **多部分问题**：每个子问题独立成结果块，全部保留展示（不再被最后一步覆盖）；
- **语义方向**：「最好/最低」→ order=asc，「最差/最高」→ order=desc，不再答反；
- **全量排名**：top_n 上限 500（375 城），「排全国第几」「多少个城市达标」直接在研判里给答案（含目标城市名次定位、达标/超标计数、两城差值）；
- **分组切片**：「南北各自超标最严重的 3 个省份」→ `aggregate(split_by=north_south, top_n=3)` 自动按组切片输出南北各自的组内 TOP N，不用用户自己从 31 省表归组；
- **AQI 一等公民**：问 AQI 不再静默替换成 PM2.5（超标判定依据 HJ 633-2012，>100 为超标日）；
- **年份越界**：问 2023/2025 → 明确提示数据覆盖范围并改用可用年份（trend 空表不再崩溃），LLM 与规则模式都会在思考中说明替换、不静默；
- **能力边界**：气象要素/外国/闲聊 → LLM 输出 `out_of_scope` 诚实拒答并引导可分析什么（规则兜底模式亦有关键词拒答）；
- **多轮记忆**：LLM 路径注入上一轮上下文（保守继承：本问已明确范围时以本问为准），「那上海呢」「这几个城市」式追问可正确继承；
- **冗余步骤剪枝**：LLM 偶发多规划步骤时自动去重（同工具同参数、重复 validate），不浪费执行时间；
- **规则兜底模式**：无网无 Key 时覆盖分组切片、省份排名、季节之最、城市别名（首都→北京）、定位问法等问法，同一问题集 20 题中 18 题给出有效结果；
- **回归工具**：`tests/generalization_audit.py`（20 条全新问法 LLM 模式审计）、`tests/rule_mode_audit.py`（同一问题集强制规则模式审计）、`tests/audit_focus.py`（针对性复验）、`tests/smoke_new_caps.py`（新能力冒烟），审计日志落盘 `outputs/logs/`。

## 流式推送架构（网页版）

```
浏览器 EventSource ──GET /api/stream?q=...──▶ 服务端
                                                │
   Agent 后台线程运行 ──每步事件──▶ queue.Queue（消息队列缓冲）
                                                │
   SSE 生成器按单调时钟节流：相邻事件间隔不足 --interval 秒时 sleep 补齐，
   再依次推送 ──▶ 前端逐条渲染（规划卡片 → 步骤卡片 → 图表 → 结论表格）
```

事件序列示例（共 23 个）：`thought×3 → plan → (thought → action → observation)×5 步 → chart → table → done → closed`。
`POST /api/ask` 保留为一次性接口（兼容命令行与测试）。

**并发模型**：数据集只加载一次且全程只读；每次提问现场创建独立的 ReActAgent
实例（无全局串行锁），多人同时提问并行执行；多轮记忆按 sid（前端
sessionStorage 生成）隔离，不同标签页/浏览器的「那上海呢」式追问互不串扰。
3 路并发实测：墙钟 14s vs 串行总和 36s。验证脚本：`tests/web_concurrency_check.py`。

## 目录结构

```
air-quality-agent/
├── run.py                     # 命令行一键启动入口
├── web.py                     # 网页版启动入口（零第三方 Web 框架）
├── requirements.txt
├── aq_agent/                  # 主包
│   ├── config.py              # GB 3095-2012 限值、路径、季节常量
│   ├── geo.py                 # 城市→省份映射（375 城）
│   ├── data_layer.py          # 分片(chunk)读取 + 清洗
│   ├── analysis.py            # 聚合统计 / 国标判定 / 异常检测 / 结果校验
│   ├── planner.py             # 自然语言→任务计划（规则 + 可选 LLM）
│   ├── react_agent.py         # ReAct 引擎 + 工具集合
│   ├── memory.py              # 多轮对话记忆 + 任务看板
│   ├── charts.py              # matplotlib 折线/柱状/饼图（中文字体自适应）
│   ├── report.py              # 结构化文本 + HTML 报告 + 思政文案
│   └── cli.py                 # 命令行交互
├── scripts/
│   ├── fetch_data.py          # 下载原始数据（CN-AIR，git LFS 稀疏检出）
│   └── prepare_data.py        # 日宽表 → tidy 日长表（分片转换）
├── data/
│   ├── processed/air_quality_2024.csv   # 分析用数据集（137,250 行）
│   └── DATASET.md             # 数据集说明与口径
└── outputs/                   # 运行产出：charts / reports / logs
```

## 数据集（真实公开数据，非模拟）

- 来源：**CN-AIR**（中国环境监测总站公开数据整理，MIT 许可），
  覆盖 375 个城市、2024 全年逐日六项污染物 + AQI；
- 原始为小时级宽表（每日一个 CSV），`scripts/prepare_data.py`
  分片转换为 tidy 长表；日值口径见 `data/DATASET.md`；
- 重新生成数据：`python scripts/fetch_data.py --year 2024` 后运行
  `python scripts/prepare_data.py --raw-dir <解压目录> --year 2024`。

## 国标口径（GB 3095-2012 二级标准）

| 污染物 | 判定指标 | 二级限值 |
|---|---|---|
| PM2.5 | 24h 平均 / 年平均 | 75 / 35 μg/m³ |
| PM10 | 24h 平均 / 年平均 | 150 / 70 μg/m³ |
| SO₂ | 24h 平均 / 年平均 | 150 / 60 μg/m³ |
| NO₂ | 24h 平均 / 年平均 | 80 / 40 μg/m³ |
| O₃ | 日最大 8h 平均 | 160 μg/m³ |
| CO | 24h 平均 | 4 mg/m³ |

## 创新点（加分项）

1. **多轮对话记忆**：追问「再看这几个城市的 PM10 情况」自动继承上下文；
2. **分组切片分析**：「南北各自最严重的 N 个省份」自动按组切片，无需人工归组；
3. **高阶归因工具**：城市间 Pearson 相关（区域传输/同步性）、污染来源代理分析（扬尘比值/交通同源/采暖贡献/二次生成，诚实标注非源谱解析）；
4. **异常告警**：MAD 3σ 检测时序跳变，标记污染突发时段；
5. **结果校验工具**：占比合计=100%、排名单调性、样本量校验，降低规划错误；
6. **任务日志看板**：查询历史、耗时统计、高频任务（`board` 命令）；
7. **双输出模板**：简易解读版 / 专业统计版（`mode` 命令切换）；
8. **网页版并发**：每请求独立 Agent + 按会话隔离记忆，多人同时提问不排队；
9. **大数据分片加载**：日文件 + chunksize 流式读取，不一次性载入内存。

## 接入大模型做任务规划（function calling）

默认使用本地规则规划器（无网、无 Key 也能完整演示）。接入 LLM 后，
**每个问题都由大模型现生成工具调用链**，任意问法都能规划：

```bash
set AQ_LLM_BASE_URL=https://api.openai.com/v1   # 或任何 OpenAI 兼容接口
set AQ_LLM_API_KEY=sk-...
set AQ_LLM_MODEL=gpt-4o-mini
```

也可以写入项目根目录的 `.env` 文件（`AQ_LLM_BASE_URL=...` 每行一项），
启动时自动加载。网页版顶部会显示当前规划来源（LLM / 本地规则）。

**容错设计**：LLM 超时、返回非 JSON、工具名不在白名单、参数越界等情况，
全部自动回退到本地规则规划器，演示不中断。核心计算（读取/清洗/统计/超标/绘图）
始终基于本地数据与自定义工具，大模型只负责任务拆解，不碰任何数字。

## 演示建议（60 分钟）

10min 审题 + 工具设计 → 40min 编码调试 → 10min 提交展示。
上台演示顺序：三个示例提问跑完整链路 → 追问一句验证多轮记忆 → `board` 看看板 → 打开 HTML 报告展示图表与思政融入。
