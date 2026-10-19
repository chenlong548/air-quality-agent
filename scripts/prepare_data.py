# -*- coding: utf-8 -*-
"""
数据准备脚本：把 CN-AIR 原始「日宽表」转换为分析用「日长表」。

原始数据（每个自然日一个 CSV，宽表：date,hour,type,城市1..城市N）
    →  tidy 长表（一行 = 一个城市一天）：
    城市, 日期, AQI, 质量等级, PM2.5, PM10, SO2, NO2, CO, O3,
    PM2.5_24h, PM10_24h, SO2_24h, NO2_24h, CO_24h, O3_8h

日值口径（与《环境空气质量标准》GB 3095-2012 / HJ 633-2012 对齐）：
  - *_24h 系列：取当日最后一个非空小时值（滚动 24h 序列的末日发布值），
    全空时回退为对应小时浓度当日均值；
  - O3_8h：取当日各小时 8h 滑动平均的最大值，即「日最大 8 小时平均」；
  - AQI：取当日最后一个非空小时值（日发布 AQI）；
  - 质量等级：由日 AQI 按 HJ 633-2012 分级推导（优/良/轻度/中度/重度/严重污染）。

用法：
    python scripts/prepare_data.py --raw-dir <CN-AIR年份目录> --year 2024 \
        --out data/processed/air_quality_2024.csv
"""
import argparse
import os
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# tidy 表保留的指标列（浓度单位：μg/m³，CO 为 mg/m³，与 GB 3095-2012 一致）
METRICS = [
    "PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
    "PM2.5_24h", "PM10_24h", "SO2_24h", "NO2_24h", "CO_24h", "O3_8h",
]

# AQI -> 质量等级（HJ 633-2012）
AQI_LEVELS = [
    (0, 50, "优"),
    (51, 100, "良"),
    (101, 150, "轻度污染"),
    (151, 200, "中度污染"),
    (201, 300, "重度污染"),
    (301, 10**9, "严重污染"),
]


def aqi_to_level(aqi):
    if pd.isna(aqi):
        return None
    for lo, hi, name in AQI_LEVELS:
        if lo <= aqi <= hi:
            return name
    return None


def process_one_day(path, keep_cities=None):
    """把单个日宽表文件转成该日的 tidy 片段（DataFrame），全向量化。"""
    df = pd.read_csv(path)
    if "type" not in df.columns:
        return None
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
    df = df.dropna(subset=["hour"])
    city_cols = [c for c in df.columns if c not in ("date", "hour", "type")]
    if keep_cities is not None:
        city_cols = [c for c in city_cols if c in keep_cities]
    long = df.melt(id_vars=["hour", "type"], value_vars=city_cols,
                   var_name="城市", value_name="v")
    long["v"] = pd.to_numeric(long["v"], errors="coerce")
    long = long.dropna(subset=["v"])
    g = long.groupby(["城市", "type"], observed=True)
    last = g["v"].last()    # 每城每指标：按小时排序后的最后非空值
    mean = g["v"].mean()
    mx = g["v"].max()

    def pick(stat, metric):
        try:
            return stat.xs(metric, level="type")
        except KeyError:
            return pd.Series(dtype=float)

    out = pd.DataFrame(index=pd.Index(city_cols, name="城市"))
    aqi = pick(last, "AQI")
    out["AQI"] = aqi.reindex(out.index)
    for m in METRICS:
        if m == "O3_8h":
            out[m] = pick(mx, m).reindex(out.index)
        elif m.endswith("_24h"):
            s = pick(last, m).reindex(out.index)
            # _24h 全空回退：小时浓度当日均值
            fallback = pick(mean, m[:-4]).reindex(out.index)
            out[m] = s.fillna(fallback)
        else:
            out[m] = pick(mean, m).reindex(out.index)
    out = out.reset_index()
    out.insert(1, "日期", pd.to_datetime(
        os.path.basename(path).replace("china_cities_", "").replace(".csv", ""),
        format="%Y%m%d", errors="coerce"))
    out["质量等级"] = out["AQI"].map(aqi_to_level)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, help="CN-AIR 年份目录（含 china_cities_YYYYMMDD.csv）")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    files = sorted(
        os.path.join(args.raw_dir, f)
        for f in os.listdir(args.raw_dir)
        if f.startswith("china_cities_") and f.endswith(".csv") and f[13:17] == str(args.year)
    )
    if not files:
        print(f"[ERROR] 在 {args.raw_dir} 下没有找到 {args.year} 年的日文件")
        sys.exit(1)

    out_path = args.out or os.path.join("data", "processed", f"air_quality_{args.year}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"[prepare] 共 {len(files)} 个日文件，开始分片转换 ...")
    parts = []
    for i, fp in enumerate(files, 1):
        part = process_one_day(fp)
        if part is not None and len(part):
            parts.append(part)
        if i % 60 == 0 or i == len(files):
            print(f"[prepare] 进度 {i}/{len(files)}")

    full = pd.concat(parts, ignore_index=True)
    cols = ["城市", "日期", "AQI", "质量等级"] + METRICS
    full = full[cols].sort_values(["日期", "城市"]).reset_index(drop=True)
    full.to_csv(out_path, index=False, encoding="utf-8-sig")

    # 数据概况报告
    print(f"[prepare] 输出: {out_path}")
    print(f"[prepare] 行数: {len(full)}  城市数: {full['城市'].nunique()}  "
          f"日期范围: {full['日期'].min().date()} ~ {full['日期'].max().date()}")
    miss = full[["AQI"] + METRICS].isna().mean().mul(100).round(1)
    print("[prepare] 各字段缺失率(%):", miss.to_dict())


if __name__ == "__main__":
    main()
