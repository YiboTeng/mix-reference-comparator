#!/usr/bin/env python3
"""把分析器依赖安装到用户明确指定的当前任务目录。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    """读取目标目录，并通过 pip 安装 requirements.txt 中声明的分析依赖。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, type=Path)
    args = parser.parse_args()
    # 强制使用显式目标目录，避免把依赖写进系统 Python 或污染其他项目环境。
    target = args.target.resolve()
    target.mkdir(parents=True, exist_ok=True)
    requirements = Path(__file__).with_name("requirements.txt")
    # 使用当前正在运行的 Python 调用 pip，保证解释器与依赖版本对应。
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--target",
        str(target),
        "-r",
        str(requirements),
    ]
    print("正在执行：", subprocess.list2cmdline(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
