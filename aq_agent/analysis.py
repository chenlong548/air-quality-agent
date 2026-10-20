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
    out["是否超标"] = (out["年均浓度"].apply(lambda v: "超标" if v > limit else "达标")
                     if limit else "—（无年均限值）")
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

    返回列：省份、有效天数、超标天数、占全国比例%（合计=100%，可校验）、省内超标率%。
    """
    return group_exceed_days(df, pollutant, "省份", "省份", limit_kind)


def group_exceed_days(df: pd.DataFrame, pollutant: str, group_col: str,
                      label: str, limit_kind=None):
    """按任意分组（省份/季节/南北方…）统计超标天数。

    group_col: df 中已存在的分组列名；label: 输出表的分组列名。
    返回列：label、有效天数、超标天数、占全国比例%、组内超标率%。
    """
    kind = limit_kind or config.DEFAULT_LIMIT_KIND[pollutant]
    col = config.DAILY_JUDGE_COL[pollutant]
    limit = config.GB3095_GRADE2[pollutant][kind]
    d = df[df[col].notna()]
    total = d.groupby(group_col).size()
    exceed = d[d[col] > limit].groupby(group_col).size()
    res = pd.DataFrame({"有效天数": total, "超标天数": exceed}).fillna(0)
    res["超标天数"] = res["超标天数"].astype(int)
    res["占全国比例%"] = (res["超标天数"] / res["超标天数"].sum() * 100).round(1)
    res["组内超标率%"] = (res["超标天数"] / res["有效天数"] * 100).round(1)
    res = res.sort_values("超标天数", ascending=False)
    res.index.name = label
    return res.reset_index(), limit


def month_to_season(m: int) -> str:
    """月份 → 季节（与 config.SEASONS 一致，含跨年冬季 12-2 月）。"""
    for s, (a, b) in config.SEASONS.items():
        if (a <= m <= b) or (a > b and (m >= a or m <= b)):
            return s
    return "未知"


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


def province_mean(df: pd.DataFrame, pollutant: str, top=None, ascending=False):
    """各省平均浓度排名（年均浓度列口径）。"""
    col = "PM2.5_24h" if pollutant == "PM2.5" else config.DAILY_JUDGE_COL[pollutant]
    g = df.groupby("省份")[col].agg(["mean", "count"]).dropna(subset=["mean"])
    g = g.sort_values("mean", ascending=ascending)
    if top:
        g = g.head(top)
    out = g.reset_index()
    out.columns = ["省份", "平均浓度", "有效天数"]
    out["平均浓度"] = out["平均浓度"].round(1)
    limit = _annual_limit(pollutant)
    out["是否超标"] = (out["平均浓度"].apply(lambda v: "超标" if v > limit else "达标")
                     if limit else "—（无年均限值）")
    return out, limit


def city_exceed_days(df: pd.DataFrame, pollutant: str, top=5, ascending=False):
    """城市超标天数排名（默认取超标最多的 top N）。"""
    kind = config.DEFAULT_LIMIT_KIND[pollutant]
    col = config.DAILY_JUDGE_COL[pollutant]
    limit = config.GB3095_GRADE2[pollutant][kind]
    valid = df[df[col].notna()]
    exceed = valid[valid[col] > limit].groupby("城市").size()
    total = valid.groupby("城市").size()
    res = pd.DataFrame({"有效天数": total, "超标天数": exceed}).fillna(0)
    res["超标天数"] = res["超标天数"].astype(int)
    res["超标率%"] = (res["超标天数"] / res["有效天数"] * 100).round(1)
    res = res.sort_values("超标天数", ascending=ascending).head(top)
    return res.reset_index(), limit


# ---------- 分组切片（组内 TOP N：如「南北各自超标最严重的 3 个省份」） ----------

def region_of(province):
    """省份 → 南北方（与 config.NORTH_PROVINCES 一致）。"""
    return "北方" if province in config.NORTH_PROVINCES else "南方"


def region_province_top(df: pd.DataFrame, pollutant: str, top=3,
                        metric="exceed_days", ascending=False):
    """按南北方分组切片：各组内省份 TOP N。

    典型问法：「南北各自超标最严重的 3 个省份」——用户要的是按组切片，
    而不是自己从 31 省全量表里归组。
    metric=exceed_days 按超标天数；metric=mean 按平均浓度。
    返回列：区域、省份、有效天数、超标天数、组内超标率%（或 平均浓度）。
    """
    col = "PM2.5_24h" if pollutant == "PM2.5" else config.DAILY_JUDGE_COL[pollutant]
    d = df[df[col].notna()].copy()
    if d.empty:
        return pd.DataFrame(columns=["区域", "省份"]), None
    d["区域"] = d["省份"].map(region_of)
    if metric == "exceed_days":
        kind = config.DEFAULT_LIMIT_KIND[pollutant]
        limit = config.GB3095_GRADE2[pollutant][kind]
        exceed = d[d[col] > limit].groupby(["区域", "省份"]).size()
        total = d.groupby(["区域", "省份"]).size()
        res = pd.DataFrame({"有效天数": total, "超标天数": exceed}).fillna(0)
        res["超标天数"] = res["超标天数"].astype(int)
        res["组内超标率%"] = (res["超标天数"] / res["有效天数"] * 100).round(1)
        res = res.reset_index()
        res["_ord"] = res["区域"].map({"北方": 0, "南方": 1})
        # 组内排序：默认最严重（超标天数最多）在前；ascending=True 时最轻在前
        res = res.sort_values(["_ord", "超标天数"], ascending=[True, ascending])
        res = res.groupby("区域", group_keys=False).head(top).drop(columns="_ord")
        return res.reset_index(drop=True), limit
    # mean 口径
    g = d.groupby(["区域", "省份"])[col].agg(["mean", "count"]).dropna(subset=["mean"])
    g = g.reset_index().rename(columns={"mean": "平均浓度", "count": "有效天数"})
    g["平均浓度"] = g["平均浓度"].round(1)
    g["_ord"] = g["区域"].map({"北方": 0, "南方": 1})
    g = g.sort_values(["_ord", "平均浓度"], ascending=[True, ascending])
    g = g.groupby("区域", group_keys=False).head(top).drop(columns="_ord")
    return g.reset_index(drop=True), _annual_limit(pollutant)


# ---------- 高阶分析：城市间相关性 / 污染来源代理 ----------

def _corr_strength(r):
    """相关系数 → 业务解读（区域传输/共同暴露语境）。"""
    if r is None:
        return "无法判定（有效配对天数不足 30）"
    a = abs(r)
    if a >= 0.8:
        return "强相关——两地污染常同步起伏，多受同一天气过程/区域传输影响"
    if a >= 0.5:
        return "中等相关——存在共同影响因素，同时各有本地源贡献"
    return "弱相关——两地污染走势相对独立，本地源影响更突出"


def city_correlation(df: pd.DataFrame, cities, pollutant: str, year=None):
    """两个（或多个）城市某污染物的日浓度 Pearson 相关性。

    回答「A 和 B 的污染是否同步 / 是不是受同一天气过程影响」。
    返回 (rows_df[城市A,城市B,相关系数r,样本天数,相关强度], 口径说明)。
    """
    col = "PM2.5_24h" if pollutant == "PM2.5" else config.DAILY_JUDGE_COL[pollutant]
    d = df[df["城市"].isin(cities)]
    if year:
        d = d[d["日期"].dt.year == year]
    piv = d.pivot_table(index="日期", columns="城市", values=col, aggfunc="mean")
    cols = [c for c in cities if c in piv.columns]
    rows = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pair = piv[[cols[i], cols[j]]].dropna()
            r = (round(float(pair.iloc[:, 0].corr(pair.iloc[:, 1])), 3)
                 if len(pair) >= 30 else None)
            rows.append({"城市A": cols[i], "城市B": cols[j],
                         "相关系数r": r if r is not None else "样本不足",
                         "样本天数": int(len(pair)),
                         "相关强度": _corr_strength(r)})
    return pd.DataFrame(rows), f"{pollutant} 日浓度 Pearson 相关（{year or '全时段'}）"


# 采暖季月份（11 月—次年 3 月）
HEATING_MONTHS = {11, 12, 1, 2, 3}


def pollution_source_profile(df: pd.DataFrame, city: str, pollutant: str = "PM2.5",
                             year=None):
    """污染来源特征分析（浓度代理指标，非源谱解析）。

    数据只有六项污染物浓度，没有化学源谱，因此用五类代理指标刻画来源结构：
      1. PM10/PM2.5 比值      —— 扬尘/粗颗粒贡献代理；
      2. NO2–CO 日相关        —— 机动车/燃烧排放同源性；
      3. 工作日 vs 周末差值   —— 人类活动强度代理；
      4. 采暖季/非采暖季比值  —— 采暖贡献代理（PM2.5/PM10/SO2/NO2/CO）；
      5. 季节错位（夏 O3/冬 PM2.5）与周末臭氧效应 —— 二次生成代理（PM2.5/O3）。
    返回 (rows_df[指标,数值,解读], 结论)。无数据时 rows 为 None。
    """
    d = df[df["城市"] == city]
    if year:
        d = d[d["日期"].dt.year == year]
    d = d[d["日期"].notna()]
    if d.empty:
        return None, f"数据集中没有「{city}」的有效记录（年份={year or '全部'}）"
    rows = []

    def add(name, value, interp, num=None):
        rows.append({"指标": name, "数值": value, "数值n": num, "解读": interp})

    # 1) 扬尘代理：PM10/PM2.5（与 PM2.5、PM10 相关）
    if pollutant in ("PM2.5", "PM10"):
        pair = d[["PM2.5_24h", "PM10_24h"]].dropna()
        pair = pair[pair["PM2.5_24h"] > 5]
        if len(pair) > 30:
            ratio = float((pair["PM10_24h"] / pair["PM2.5_24h"]).median())
            if ratio >= 2.5:
                interp = "偏高——扬尘/粗颗粒贡献显著（道路扬尘、建筑、风沙）"
            elif ratio >= 1.8:
                interp = "中等——粗细颗粒混合，扬尘有一定贡献"
            else:
                interp = "偏低——以细颗粒物（燃烧/二次生成）为主"
            add("PM10/PM2.5 比值（扬尘代理）", round(ratio, 2), interp, round(ratio, 2))

    # 2) NO2–CO 同源相关（全污染物通用：机动车与燃烧排放的指纹）
    pair = d[["NO2_24h", "CO_24h"]].dropna()
    if len(pair) > 30:
        r = float(pair["NO2_24h"].corr(pair["CO_24h"]))
        interp = ("强同源——机动车/燃烧排放贡献突出" if r >= 0.6 else
                  "中等同源——燃烧源与交通源并存" if r >= 0.35 else
                  "同源性弱——两类排放来源结构差异较大")
        add("NO2–CO 日相关（燃烧/交通同源性）", round(r, 2), interp, round(r, 2))

    # 3) 人类活动代理：工作日 vs 周末（目标污染物）
    pcol = "PM2.5_24h" if pollutant == "PM2.5" else config.DAILY_JUDGE_COL[pollutant]
    wd = d[d["日期"].dt.weekday < 5]
    we = d[d["日期"].dt.weekday >= 5]
    a, b = wd[pcol].mean(), we[pcol].mean()
    if pd.notna(a) and pd.notna(b) and b > 0:
        gap = (float(a) - float(b)) / float(b) * 100
        if abs(gap) >= 8:
            unit = self_unit = config.GB3095_GRADE2[pollutant]["unit"]
            interp = (f"工作日比周末{'高' if gap > 0 else '低'} {abs(gap):.0f}%——"
                      + ("人类活动/交通排放影响明显" if gap > 0
                         else "周末反而更高，需关注非工业源与扩散条件"))
            add(f"{pollutant} 工作日/周末差异",
                f"{a:.2f} vs {b:.2f} {unit}", interp, round(gap, 1))

    # 4) 采暖贡献代理（11-3 月 vs 其余月份）
    if pollutant in ("PM2.5", "PM10", "SO2", "NO2", "CO"):
        heat = d[d["日期"].dt.month.isin(HEATING_MONTHS)][pcol].mean()
        rest = d[~d["日期"].dt.month.isin(HEATING_MONTHS)][pcol].mean()
        if pd.notna(heat) and pd.notna(rest) and rest > 0:
            ratio = float(heat) / float(rest)
            if ratio >= 1.5:
                interp = f"采暖季浓度约为非采暖季 {ratio:.1f} 倍——采暖贡献显著"
            elif ratio >= 1.2:
                interp = f"采暖季浓度约为非采暖季 {ratio:.1f} 倍——采暖有一定贡献"
            else:
                interp = f"采暖季/非采暖季比值 {ratio:.1f}——采暖贡献不突出"
            add(f"{pollutant} 采暖季/非采暖季比值", round(ratio, 2), interp, round(ratio, 2))

    # 5) 二次生成代理
    if pollutant in ("PM2.5", "O3"):
        o3m = d.groupby(d["日期"].dt.month)["O3_8h"].mean()
        p25m = d.groupby(d["日期"].dt.month)["PM2.5_24h"].mean()
        if len(o3m.dropna()) >= 6 and len(p25m.dropna()) >= 6:
            summer_o3 = o3m.loc[[m for m in (6, 7, 8) if m in o3m.index]].mean()
            winter_p25 = p25m.loc[[m for m in (12, 1, 2) if m in p25m.index]].mean()
            if pd.notna(summer_o3) and pd.notna(winter_p25):
                add("夏 O3 / 冬 PM2.5 季节错位（二次生成）",
                    f"{summer_o3:.0f} / {winter_p25:.0f} μg/m³",
                    "夏季臭氧高、冬季 PM2.5 高的典型季节错位——光化学二次生成与冬季"
                    "采暖/逆温叠加是两地共性机制", None)
    if pollutant == "O3":
        we_o3 = we["O3_8h"].mean()
        wd_o3 = wd["O3_8h"].mean()
        if pd.notna(we_o3) and pd.notna(wd_o3) and wd_o3 > 0:
            gap = (float(we_o3) - float(wd_o3)) / float(wd_o3) * 100
            if abs(gap) >= 8:
                add("O3 周末效应", f"{we_o3:.0f} vs {wd_o3:.0f} μg/m³",
                    f"周末 O3 比工作日{'高' if gap > 0 else '低'} {abs(gap):.0f}%——"
                    + ("周末 NOx 减排但 VOCs 仍足，光化学生成占比上升" if gap > 0
                       else "工作日臭氧更高，工业/交通 NOx 排放驱动明显"),
                    round(gap, 1))

    if not rows:
        return None, f"「{city}」有效监测样本不足，无法计算来源代理指标"

    # 结论：把超过阈值的强信号汇成一句话
    sig = []
    if any("扬尘代理" in r["指标"] and r["数值n"] and r["数值n"] >= 2.5 for r in rows):
        sig.append("扬尘/粗颗粒贡献显著")
    if any("同源" in r["指标"] and r["数值n"] and r["数值n"] >= 0.6 for r in rows):
        sig.append("机动车与燃烧排放同源突出")
    if any("工作日/周末差异" in r["指标"] and r["数值n"] and abs(r["数值n"]) >= 15
           for r in rows):
        sig.append("人类活动强度差异明显")
    if any("采暖季/非采暖季" in r["指标"] and r["数值n"] and r["数值n"] >= 1.5
           for r in rows):
        sig.append("采暖贡献显著")
    if any("季节错位" in r["指标"] for r in rows):
        sig.append("光化学二次生成与冬季逆温/采暖季节错位典型")
    concl = ("；".join(sig) if sig else
             "各代理指标均处温和区间，本地来源结构不突出，以区域传输复合污染为主")
    concl += "。（以上为浓度比值/相关性代理指标推断，非源谱成分解析，供方向性参考）"
    return pd.DataFrame(rows), concl


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
    if res is None or len(res) == 0:
        return False, "无数据行（先检查年份/城市/污染物是否在数据范围内）"
    s = res["占全国比例%"].sum()
    return abs(s - 100) < 1.0, f"占全国比例合计 {s:.1f}%"


def validate_rank(res: pd.DataFrame):
    """排名类结果校验：单调递减且样本量充足。"""
    if res is None or len(res) == 0:
        return False, "无数据行（先检查年份/城市/污染物是否在数据范围内）"
    vals = res[res.columns[1]].tolist()
    dec = all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))
    inc = all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
    mono = dec or inc   # 升序（最低在前）与降序（最高在前）都算合法排名
    enough = bool((res["有效天数"] > 300).all())
    return mono and enough, f"单调性={mono}, 有效天数均>300={enough}"


# ---------- 工具函数 ----------

def _annual_limit(pollutant):
    return config.GB3095_GRADE2[pollutant].get("annual")


def _kind_desc(kind):
    return {"annual": "年平均", "24h": "24小时平均", "8h": "日最大8小时平均",
            "1h": "1小时平均"}.get(kind, kind)


def format_limit_text(pollutant, kind=None):
    """生成限值说明文案（AQI 走 HJ 633-2012 口径，其余为 GB 3095-2012）。"""
    if pollutant == "AQI":
        return config.AQI_LIMIT_NOTE
    std = config.GB3095_GRADE2[pollutant]
    kind = kind or config.DEFAULT_LIMIT_KIND[pollutant]
    parts = []
    for k, v in std.items():
        if k == "unit":
            continue
        parts.append(f"{_kind_desc(k)}限值 {v}{std['unit']}")
    return "、".join(parts) + "（GB 3095-2012 二级标准）"
