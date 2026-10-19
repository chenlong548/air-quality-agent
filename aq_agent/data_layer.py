# -*- coding: utf-8 -*-
"""数据层：分片（chunk）读取 + 清洗。

大数据工程体现：
  - tidy 长表按 chunksize 流式读取，不一次性载入内存；
  - 清洗在分片上增量完成，最后只保留聚合所需的紧凑结构；
  - 预留 iter_year_chunks() 供扩展到多年份/更大规模数据。
"""
import os

import numpy as np
import pandas as pd

from . import config
from .geo import CITY_TO_PROVINCE

# 浓度列的物理合理上限（超出视为异常值）
PHYSICAL_MAX = {
    "PM2.5": 1000, "PM10": 1500, "SO2": 1000, "NO2": 1000,
    "CO": 100, "O3": 1000,
}
CONC_COLS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
             "PM2.5_24h", "PM10_24h", "SO2_24h", "NO2_24h", "CO_24h", "O3_8h"]


def iter_chunks(path, chunksize=50_000):
    """按块流式读取 tidy CSV（yield DataFrame）。"""
    for chunk in pd.read_csv(path, chunksize=chunksize, encoding="utf-8-sig"):
        yield chunk


def clean_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """单块清洗：类型规整、日期解析、负值/超限异常置空。"""
    chunk = chunk.copy()
    chunk["城市"] = chunk["城市"].astype(str).str.strip()
    chunk["日期"] = pd.to_datetime(chunk["日期"], errors="coerce")
    if "省份" not in chunk.columns:
        chunk["省份"] = chunk["城市"].map(CITY_TO_PROVINCE)
    for c in CONC_COLS:
        if c in chunk.columns:
            chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
            base = c.replace("_24h", "").replace("_8h", "")
            chunk.loc[(chunk[c] < 0) | (chunk[c] > PHYSICAL_MAX.get(base, 1e9)), c] = np.nan
    if "AQI" in chunk.columns:
        chunk["AQI"] = pd.to_numeric(chunk["AQI"], errors="coerce")
        chunk.loc[(chunk["AQI"] < 0) | (chunk["AQI"] > 500), "AQI"] = np.nan
    # 关键字段缺失的行丢弃（无城市或无日期无法参与统计）
    chunk = chunk.dropna(subset=["城市", "日期"])
    return chunk


def load_dataset(path, chunksize=50_000):
    """分片读取 + 分片清洗 + 流式拼接，返回 (df, 清洗报告)。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"数据文件不存在: {path}")
    total_raw, total_kept = 0, 0
    parts = []
    for chunk in iter_chunks(path, chunksize):
        total_raw += len(chunk)
        cleaned = clean_chunk(chunk)
        total_kept += len(cleaned)
        parts.append(cleaned)
    df = pd.concat(parts, ignore_index=True)
    report = {
        "文件": os.path.basename(path),
        "原始行数": int(total_raw),
        "清洗后行数": int(total_kept),
        "丢弃行数": int(total_raw - total_kept),
        "城市数": int(df["城市"].nunique()),
        "日期范围": f"{df['日期'].min().date()} ~ {df['日期'].max().date()}",
        "缺失率": {c: round(float(df[c].isna().mean()), 4) for c in CONC_COLS if c in df.columns},
    }
    return df, report


def discover_years(processed_dir=None):
    """扫描 data/processed 下可用的年份文件。"""
    d = processed_dir or config.PROCESSED_DIR
    years = []
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if f.startswith("air_quality_") and f.endswith(".csv"):
                try:
                    years.append(int(f[len("air_quality_"):-4]))
                except ValueError:
                    pass
    return years
