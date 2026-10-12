#!/usr/bin/env python3
"""学屿客户端入口：python main.py"""
import sys
import traceback


def _crash_log():
    """冻结环境下 windowed 模式没有 stderr，把致命错误落到临时文件便于诊断。"""
    if getattr(sys, "frozen", False):
        import faulthandler

        log = open("/tmp/xueyu-crash.log", "a", encoding="utf-8")
        faulthandler.enable(file=log)
        sys.excepthook = lambda *exc: log.write("".join(traceback.format_exception(*exc)) + "\n" + "-" * 60 + "\n")


_crash_log()

if __name__ == "__main__":
    from desktop.shell import main

    main()
