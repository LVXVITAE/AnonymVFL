#!/usr/bin/env python3
# 测试subprocess是否能正确捕获输出

import subprocess
import threading
import time
from pathlib import Path

# 获取项目根目录
project_root = Path(__file__).parent.parent


def test_start_ray():
    print("测试启动Ray脚本的输出捕获...")

    process = subprocess.Popen(
        ['bash', 'start_ray_only.sh'],
        cwd=str(project_root / 'company'),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )

    print(f"进程已启动，PID: {process.pid}")

    def monitor():
        count = 0
        while process.poll() is None and count < 50:
            line = process.stdout.readline()
            if line:
                print(f"[输出] {line.strip()}")
                count += 1
            time.sleep(0.1)
        print(f"监控结束，共读取{count}行")

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()

    time.sleep(10)
    print("10秒测试完成")

    if process.poll() is None:
        print("进程仍在运行，发送终止信号")
        process.terminate()
        time.sleep(2)
        if process.poll() is None:
            process.kill()


if __name__ == "__main__":
    test_start_ray()
