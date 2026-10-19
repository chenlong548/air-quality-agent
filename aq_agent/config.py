# -*- coding: utf-8 -*-
"""全局配置：路径、GB 3095-2012 限值、季节定义等常量。"""
import os

# ---------------- 路径 ----------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
CHART_DIR = os.path.join(OUTPUT_DIR, "charts")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

for _d in (CHART_DIR, REPORT_DIR, LOG_DIR):
    os.makedirs(_d, exist_ok=True)

# 默认加载的数据文件（年份 -> 文件名）
def data_file(year: int) -> str:
    return os.path.join(PROCESSED_DIR, f"air_quality_{year}.csv")


# ---------------- GB 3095-2012《环境空气质量标准》二级限值 ----------------
# 浓度单位：μg/m³（CO 为 mg/m³）。annual=年平均，24h=24小时平均，
# 8h=日最大8小时平均，1h=1小时平均。
GB3095_GRADE2 = {
    "PM2.5": {"annual": 35, "24h": 75, "unit": "μg/m³"},
    "PM10": {"annual": 70, "24h": 150, "unit": "μg/m³"},
    "SO2": {"annual": 60, "24h": 150, "unit": "μg/m³"},
    "NO2": {"annual": 40, "24h": 80, "unit": "μg/m³"},
    "O3": {"8h": 160, "1h": 200, "unit": "μg/m³"},   # 无年均值，按日最大8h平均判定
    "CO": {"24h": 4, "1h": 10, "unit": "mg/m³"},     # 无年均值
}

# 超标判定默认口径（污染物 -> 使用哪个限值档）
DEFAULT_LIMIT_KIND = {
    "PM2.5": "24h", "PM10": "24h", "SO2": "24h", "NO2": "24h", "CO": "24h", "O3": "8h",
}

# tidy 数据集中各污染物对应的「日判定列」
DAILY_JUDGE_COL = {
    "PM2.5": "PM2.5_24h", "PM10": "PM10_24h", "SO2": "SO2_24h",
    "NO2": "NO2_24h", "CO": "CO_24h", "O3": "O3_8h",
}

POLLUTANTS = list(GB3095_GRADE2.keys())

# ---------------- 季节定义 ----------------
SEASONS = {
    "春季": (3, 5), "夏季": (6, 8), "秋季": (9, 11), "冬季": (12, 2),
}

# ---------------- 南北方分界（用于多维度对比，粗粒度） ----------------
NORTH_PROVINCES = {
    "北京", "天津", "河北", "山西", "内蒙古", "辽宁", "吉林", "黑龙江",
    "山东", "河南", "陕西", "甘肃", "青海", "宁夏", "新疆",
}
