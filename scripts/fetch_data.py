# -*- coding: utf-8 -*-
"""下载 CN-AIR 原始数据（小时级日宽表）。

数据源：GewisLab/CN-AIR（中国环境监测总站公开数据整理，MIT 许可）
      https://gitcode.com/GewisLab/CN-AIR

使用 git 稀疏检出 + Git LFS，只下载指定年份，避免拉取 2.2GB 全库。
用法：
    python scripts/fetch_data.py --year 2024
    python scripts/fetch_data.py --year 2024 --out data/raw
"""
import argparse
import os
import subprocess
import sys

REPO = "https://gitcode.com/GewisLab/CN-AIR.git"


def run(cmd, cwd=None):
    print("[fetch]", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--out", default="data/raw")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    folder = f"城市_{args.year}0101-{args.year}1231"
    target = os.path.join(out_dir, folder)

    if os.path.isdir(os.path.join(target, folder)) and \
            any(f.endswith(".csv") for f in os.listdir(os.path.join(target, folder))):
        print(f"[fetch] {target} 已存在数据，跳过下载")
        return

    run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
         REPO, target])
    run(["git", "sparse-checkout", "set", folder], cwd=target)
    run(["git", "lfs", "pull"], cwd=target)
    n = len([f for f in os.listdir(os.path.join(target, folder)) if f.endswith(".csv")])
    print(f"[fetch] 完成：{target}/{folder} 共 {n} 个日文件")
    print(f"[fetch] 下一步：python scripts/prepare_data.py --raw-dir "
          f"{os.path.join(target, folder)} --year {args.year}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
