"""UniCardio Rectified Flow 项目根模块。"""
import logging
import time

logging.Formatter.converter = lambda *args: time.gmtime(time.time() + 8 * 3600)

__all__: list[str] = []
