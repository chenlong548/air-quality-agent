# -*- coding: utf-8 -*-
"""可视化：matplotlib 折线 / 柱状 / 饼图，中文字体自动探测，深空遥测主题。"""
import datetime
import os
import uuid

import matplotlib
matplotlib.use("Agg")  # 无界面环境渲染，服务端/CI 不弹窗
import matplotlib.pyplot as plt
from matplotlib import font_manager

from . import config

# ---- 深空遥测主题（与网页端面板底色 #0d1526 一致，图表内嵌无割裂感） ----
BG = "#0d1526"      # 图表面板底色
FG = "#d7e3f4"      # 主文字
DIM = "#8ba0bf"     # 次要文字（轴标签）
GRID = "#22345c"    # 网格线
PALETTE = ["#22d3ee", "#2dd4bf", "#818cf8", "#fbbf24",
           "#fb7185", "#34d399", "#a78bfa", "#94a3b8"]  # 青→蓝→紫数据色系

_FONT_READY = False


def setup_chinese_font():
    """探测系统中文字体，避免中文乱码；同时应用深色遥测主题。"""
    global _FONT_READY
    if _FONT_READY:
        return
    candidates = ["Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC",
                  "Source Han Sans SC", "WenQuanYi Micro Hei", "Arial Unicode MS"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["figure.facecolor"] = BG
    plt.rcParams["axes.facecolor"] = BG
    plt.rcParams["savefig.facecolor"] = BG
    plt.rcParams["text.color"] = FG
    plt.rcParams["axes.labelcolor"] = FG
    plt.rcParams["axes.edgecolor"] = GRID
    plt.rcParams["xtick.color"] = DIM
    plt.rcParams["ytick.color"] = DIM
    _FONT_READY = True


def _style_ax(ax):
    """统一坐标轴样式：暗色网格、隐去上右边框。"""
    ax.set_axisbelow(True)
    ax.grid(color=GRID, linewidth=0.7, alpha=0.65)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=DIM, labelsize=9)


def _dark_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    leg = ax.legend(facecolor=BG, edgecolor=GRID, fontsize=9)
    for t in leg.get_texts():
        t.set_color(FG)
    return leg


def _save(fig, kind):
    """保存图表：文件名带毫秒时间戳 + uuid，保证每次提问的图都是当次真实计算结果，
    且并发渲染（网页版多人同时提问）时不会同名撞车。"""
    setup_chinese_font()
    fig.patch.set_facecolor(BG)
    for ax in fig.axes:
        ax.set_facecolor(BG)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    name = f"chart_{kind}_{ts}_{uuid.uuid4().hex[:6]}.png"
    path = os.path.join(config.CHART_DIR, name)
    fig.savefig(path, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return path


def line_chart(monthly_piv, title, ylabel):
    """月度变化折线图（多序列自动取遥测色系）。"""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, col in enumerate(monthly_piv.columns):
        c = PALETTE[i % len(PALETTE)]
        ax.plot(monthly_piv.index, monthly_piv[col], marker="o", linewidth=1.8,
                label=str(col), color=c, markersize=5, markeredgecolor=BG)
    ax.set_title(title, color=FG, fontsize=12, pad=12)
    ax.set_xlabel("月份", color=DIM, fontsize=10)
    ax.set_ylabel(ylabel, color=DIM, fontsize=10)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels([f"{m}月" for m in range(1, 13)])
    _style_ax(ax)
    _dark_legend(ax)
    return _save(fig, "line")


def bar_chart(df, x_col, y_col, title, ylabel, color="#22d3ee"):
    """排名柱状图。"""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(df[x_col].astype(str), df[y_col], color=color, width=0.62)
    ax.set_title(title, color=FG, fontsize=12, pad=12)
    ax.set_ylabel(ylabel, color=DIM, fontsize=10)
    _style_ax(ax)
    for i, v in enumerate(df[y_col]):
        ax.text(i, v, f"{v:g}", ha="center", va="bottom", fontsize=9, color=FG)
    return _save(fig, "bar")


def pie_chart(df, label_col, value_col, title, top=10):
    """占比饼图（超过 top 个分组时合并为「其他」）。"""
    d = df.copy()
    if len(d) > top:
        head = d.head(top)
        rest = d.tail(len(d) - top)[value_col].sum()
        head = head.copy()
        head.loc[len(head)] = {label_col: "其他", value_col: rest}
        d = head
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(d))]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.pie(d[value_col], labels=d[label_col].astype(str), autopct="%1.1f%%",
           startangle=90, colors=colors,
           wedgeprops={"edgecolor": BG, "linewidth": 1.5},
           textprops={"fontsize": 9, "color": FG})
    ax.set_title(title, color=FG, fontsize=12, pad=12)
    return _save(fig, "pie")


def anomaly_chart(series, hits, city, pollutant, unit):
    """异常检测折线图（标记跳变点）。"""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(series.index, series.values, linewidth=1.4, color="#22d3ee", label=pollutant)
    if len(hits):
        ax.scatter(hits["日期"], hits["浓度"], color="#fb7185", zorder=5,
                   s=28, edgecolors=BG, linewidths=0.8, label="异常跳变点")
    ax.set_title(f"{city} {pollutant} 时序异常检测（{unit}）",
                 color=FG, fontsize=12, pad=12)
    ax.set_ylabel(unit, color=DIM, fontsize=10)
    _style_ax(ax)
    _dark_legend(ax)
    fig.autofmt_xdate()
    return _save(fig, "anomaly")
