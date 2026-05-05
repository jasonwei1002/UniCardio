"""极简 torch.compile 可用性自检：构造一个玩具 MLP，包 torch.compile，
跑一次 forward + backward + optimizer step，打印环境信息与耗时。

服务器上：
    python script/test_compile.py
本地（macOS / CPU）也能跑，只是 Inductor 走 CPU 后端编译会慢一点。
"""

from __future__ import annotations

import platform
import time

import torch
from torch import nn


def main() -> None:
    print(f"python      : {platform.python_version()}")
    print(f"torch       : {torch.__version__}")
    print(f"cuda avail  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda runtime: {torch.version.cuda}")
        print(f"gpu         : {torch.cuda.get_device_name(0)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"device/dtype: {device} / {dtype}")

    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(512, 1024), nn.GELU(),
        nn.Linear(1024, 1024), nn.GELU(),
        nn.Linear(1024, 1),
    ).to(device=device, dtype=dtype)

    compiled = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.randn(64, 512, device=device, dtype=dtype)
    y = torch.randn(64, 1, device=device, dtype=dtype)

    # 第一次 forward 触发编译；之后稳定步耗时。
    t0 = time.perf_counter()
    out = compiled(x)
    loss = (out - y).pow(2).mean()
    loss.backward()
    optim.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_compile = time.perf_counter() - t0
    print(f"first step  : {t_compile*1000:.1f} ms (含编译)")

    optim.zero_grad(set_to_none=True)
    t0 = time.perf_counter()
    for _ in range(10):
        out = compiled(x)
        loss = (out - y).pow(2).mean()
        loss.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_steady = (time.perf_counter() - t0) / 10
    print(f"steady step : {t_steady*1000:.2f} ms")
    print("torch.compile OK")


if __name__ == "__main__":
    main()
