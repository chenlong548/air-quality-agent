# -*- coding: utf-8 -*-
"""分析层：聚合统计、GB 3095-2012 超标判定、异常检测、结果校验。"""
import numpy as np
import pandas as pd

from . import config


# ---------- 聚合统计 ----------

def annual_mean_rank(df: pd.DataFrame, pollutant: str, year=None, top=5, ascending=False):
    """城市年均浓度排名（默认取浓度最高的 top N）。"""
    col = "PM2.5_24h" if pollutant == "PM2.5" else config.DAILY_JUDGE_COL[pollutant]
    d = df
    if year is not None:
        d = d[d["日期"].dt.year == year]
    g = d.groupby("城市")[col].agg(["mean", "count"])
    g = g.dropna(subset=["mean"]).sort_values("mean", ascending=ascending)
    g = g.head(top)
    limit = _annual_limit(pollutant)
    out = g.reset_index()
    out.columns = ["城市", "年均浓度", "有效天数"]
    out["年均浓度"] = out["年均浓度"].round(1)
    out["是否超标"] = out["年均浓度"].apply(lambda v: "超标" if limit and v > limit else "达标")
    return out, limit


def monthly_trend(df: pd.DataFrame, cities, pollutant: str, year=None):
    """指定城市的月度均值序列（用于折线图）。"""
    col = config.DAILY_JUDGE_COL[pollutant] if pollutant != "PM2.5" else "PM2.5_24h"
    d = df[df["城市"].isin(cities)]
    if year is not None:
        d = d[d["日期"].dt.year == year]
    d = d.copy()
    d["月份"] = d["日期"].dt.month
    piv = d.pivot_table(index="月份", columns="城市", values=col, aggfunc="mean")
    piv = piv.sort_index()
    return piv.round(1)


def exceed_days_by_province(df: pd.DataFrame, pollutant: str, year=None, limit_kind=None):
    """各省份超标天数统计。

    返回列：有效天数、超标天数、占全国比例%（合计=100%，可校验）、省内超标率%。
    """
    kind = limit_kind or config.DEFAULT_LIMIT_KIND[pollutant]
    col = config.DAILY_JUDGE_COL[pollutant]
    limit = config.GB3095_GRADE2[pollutant][kind]
    d = df
    if year is not None:
        d = d[d["日期"].dt.year == year]
    valid = d[d[col].notna()]
    total = valid.groupby("省份").size()
    exceed = valid[valid[col] > limit].groupby("省份").size()
    res = pd.DataFrame({"有效天数": total, "超标天数": exceed}).fillna(0)
    res["超标天数"] = res["超标天数"].astype(int)
    res["占全国比例%"] = (res["超标天数"] / res["超标天数"].sum() * 100).round(1)
    res["省内超标率%"] = (res["超标天数"] / res["有效天数"] * 100).round(1)
    return res.sort_values("超标天数", ascending=False), limit


def city_season_exceed_days(df: pd.DataFrame, city: str, pollutant: str,
                            season=None, year=None):
    """某城市某季节的超标天数（season 为 None 时全年）。"""
    kind = config.DEFAULT_LIMIT_KIND[pollutant]
    col = config.DAILY_JUDGE_COL[pollutant]
    limit = config.GB3095_GRADE2[pollutant][kind]
    d = df[df["城市"] == city]
    if year is not None:
        d = d[d["日期"].dt.year == year]
    if season and season in config.SEASONS:
        m1, m2 = config.SEASONS[season]
        months = list(range(m1, m2 + 1)) if m1 <= m2 else list(range(m1, 13)) + list(range(1, m2 + 1))
        d = d[d["日期"].dt.month.isin(months)]
    valid = d[d[col].notna()]
    n_exceed = int((valid[col] > limit).sum())
    n_valid = int(len(valid))
    return {
        "城市": city, "季节": season or "全年", "污染物": pollutant,
        "超标天数": n_exceed, "有效天数": n_valid,
        "超标率%": round(n_exceed / n_valid * 100, 1) if n_valid else None,
        "限值": limit, "口径": _kind_desc(kind),
    }


def compare_dimension(df: pd.DataFrame, pollutant: str, year=None, dim="季节"):
    """多维度对比：按季节 / 南北方 对比平均浓度。"""
    col = config.DAILY_JUDGE_COL[pollutant] if pollutant != "PM2.5" else "PM2.5_24h"
    d = df
    if year is not None:
        d = d[d["日期"].dt.year == year]
    d = d.copy()
    if dim == "季节":
        d["分组"] = d["日期"].dt.month.map(
            lambda m: next(s for s, (a, b) in config.SEASONS.items()
                           if (a <= m <= b) or (a > b and (m >= a or m <= b))))
    else:
        d["分组"] = d["省份"].apply(lambda p: "北方" if p in config.NORTH_PROVINCES else "南方")
    g = d.groupby("分组")[col].agg(["mean", "count"]).round(1)
    return g.sort_values("mean", ascending=False)


# ---------- 异常检测 ----------

def detect_anomalies(df: pd.DataFrame, city: str, pollutant: str, year=None, k=3.0):
    """时序异常跳变检测：MAD（中位数绝对偏差）法，标记突发污染时段。"""
    col = config.DAILY_JUDGE_COL[pollutant] if pollutant != "PM2.5" else "PM2.5_24h"
    d = df[df["城市"] == city]
    if year is not None:
        d = d[d["日期"].dt.year == year]
    s = d.set_index("日期")[col].dropna().sort_index()
    if len(s) < 30:
        return pd.DataFrame(columns=["日期", "浓度", "偏离度"])
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0:
        return pd.DataFrame(columns=["日期", "浓度", "偏离度"])
    score = (s - med).abs() / (1.4826 * mad)
    hits = s[score > k]
    out = pd.DataFrame({"日期": hits.index, "浓度": hits.values.round(1),
                        "偏离度": score[hits.index].round(1)})
    return out.reset_index(drop=True)


# ---------- 结果校验 ----------

def validate_share(res: pd.DataFrame):
    """占比类结果校验：占全国比例之和应为 100%。"""
    s = res["占全国比例%"].sum()
    return abs(s - 100) < 1.0, f"各省份占全国比例合计 {s:.1f}%"


def validate_rank(res: pd.DataFrame):
    """排名类结果校验：单调递减且样本量充足。"""
    vals = res["年均浓度"].tolist()
    mono = all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))
    enough = bool((res["有效天数"] > 300).all())
    return mono and enough, f"单调性={mono}, 有效天数均>300={enough}"


# ---------- 工具函数 ----------

def _annual_limit(pollutant):
    return config.GB3095_GRADE2[pollutant].get("annual")


def _kind_desc(kind):
    return {"annual": "年平均", "24h": "24小时平均", "8h": "日最大8小时平均",
            "1h": "1小时平均"}.get(kind, kind)


def format_limit_text(pollutant, kind=None):
    """生成国标限值说明文案。"""
    std = config.GB3095_GRADE2[pollutant]
    kind = kind or config.DEFAULT_LIMIT_KIND[pollutant]
    parts = []
    for k, v in std.items():
        if k == "unit":
            continue
        parts.append(f"{_kind_desc(k)}限值 {v}{std['unit']}")
    return "、".join(parts) + "（GB 3095-2012 二级标准）"
