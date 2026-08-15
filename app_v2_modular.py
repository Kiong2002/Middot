"""
智能中间点推荐系统 v2 — 模块化版本
====================================
这是 app_v2.py 的模块化版本，使用 middot 包

入口：python app_v2_modular.py
"""

import os
import sys

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(__file__))

from middot import create_app, print_startup_info
from middot.config import DEEPSEEK_API_KEY, AMAP_KEY, PORT


def main():
    print_startup_info()

    app = create_app()
    app.run(debug=False, host="0.0.0.0", port=PORT, threaded=True)


if __name__ == "__main__":
    main()
