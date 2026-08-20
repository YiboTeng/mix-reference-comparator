#!/usr/bin/env python3
"""把完整混音分析依赖安装到用户指定的任务目录。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, type=Path)
    args = parser.parse_args()
    # 使用显式目录隔离依赖，避免污染系统 Python 或插件源码目录。
    target = args.target.resolve()
    target.mkdir(parents=True, exist_ok=True)
    requirements = Path(__file__).with_name("requirements.txt")
    command = [
        sys.executable, "-m", "pip", "install", "--target", str(target),
        "-r", str(requirements),
    ]
    print("正在执行：", subprocess.list2cmdline(command))
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
