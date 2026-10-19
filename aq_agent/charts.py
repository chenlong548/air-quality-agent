# -*- coding: utf-8 -*-
"""可视化：matplotlib 折线 / 柱状 / 饼图，中文字体自动探测。"""
import os

import matplotlib
matplotlib.use("Agg")  # 无界面环境渲染，比赛现场不弹窗
import matplotlib.pyplot as plt
from matplotlib import font_manager

from . import config

_FONT_READY = False


def setup_chinese_font():
    """探测系统中文字体，避免中文乱码。"""
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
    _FONT_READY = True


def _save(fig, name):
    setup_chinese_font()
    path = os.path.join(config.CHART_DIR, name)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def line_chart(monthly_piv, title, ylabel):
    """月度变化折线图。"""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for col in monthly_piv.columns:
        ax.plot(monthly_piv.index, monthly_piv[col], marker="o", linewidth=1.8, label=str(col))
    ax.set_title(title)
    ax.set_xlabel("月份")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels([f"{m}月" for m in range(1, 13)])
    ax.grid(alpha=0.3)
    ax.legend()
    return _save(fig, "chart_line.png")


def bar_chart(df, x_col, y_col, title, ylabel, color="#3b7dd8"):
    """排名柱状图。"""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(df[x_col].astype(str), df[y_col], color=color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(df[y_col]):
        ax.text(i, v, f"{v:g}", ha="center", va="bottom", fontsize=9)
    return _save(fig, "chart_bar.png")


def pie_chart(df, label_col, value_col, title, top=10):
    """占比饼图（超过 top 个分组时合并为「其他」）。"""
    d = df.copy()
    if len(d) > top:
        head = d.head(top)
        rest = d.tail(len(d) - top)[value_col].sum()
        head = head.copy()
        head.loc[len(head)] = {label_col: "其他", value_col: rest}
        d = head
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.pie(d[value_col], labels=d[label_col].astype(str), autopct="%1.1f%%",
           startangle=90, textprops={"fontsize": 9})
    ax.set_title(title)
    return _save(fig, "chart_pie.png")


def anomaly_chart(series, hits, city, pollutant, unit):
    """异常检测折线图（标记跳变点）。"""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(series.index, series.values, linewidth=1.4, color="#3b7dd8", label=pollutant)
    if len(hits):
        ax.scatter(hits["日期"], hits["浓度"], color="#d84b3b", zorder=5,
                   s=28, label="异常跳变点")
    ax.set_title(f"{city} {pollutant} 时序异常检测（{unit}）")
    ax.set_ylabel(unit)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    return _save(fig, "chart_anomaly.png")
