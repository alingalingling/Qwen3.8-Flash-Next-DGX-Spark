#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Q4_K_XL 并行分块下载器(代理 7895,断点续传,代理长连接中断自动重连)
Qwen3.8-Flash-Next-UD-Q4_K_XL: 4 分片,共 111.3 GB
分片 1(元数据,10.9MB)直连下载;分片 2/3/4 并行分块下载。
"""
import os, sys, time, threading, urllib.request, urllib.error

PROXY = "http://127.0.0.1:7895"
DIR = os.path.expanduser("~/models/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL")
LOG = os.path.expanduser("~/models/Qwen3.8-Flash-Next-GGUF/download-q4.log")
RESOLVE_BASE = "https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main/UD-Q4_K_XL/"
FILES = {
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf": 49859583136,
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf": 49376141504,
    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf": 12087983520,
}
SMALL = {"Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf": 10946624}
NCHUNK = 6   # 3 文件 × 6 块 = 18 连接
THREADS = 18

def log(msg):
    with open(LOG, "a") as f:
        f.write(time.strftime("%F %T ") + msg + "\n")

def opener():
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": PROXY, "https": PROXY}))

def resolve_cdn(fname):
    url = RESOLVE_BASE + fname
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "curl/8"})
    try:
        with opener().open(req, timeout=30) as r:
            return r.geturl()
    except Exception as e:
        log(f"resolve失败 {fname}: {e}")
        return None

def dl_small(fname, size):
    """小分片(元数据)直接下载"""
    out = os.path.join(DIR, fname)
    if os.path.exists(out) and os.path.getsize(out) == size:
        log(f"小分片已存在 {fname}")
        return True
    url = resolve_cdn(fname)
    if not url:
        log(f"FATAL: 无法解析 {fname}")
        return False
    for attempt in range(30):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
            with opener().open(req, timeout=60) as r, open(out, "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
            got = os.path.getsize(out)
            log(f"小分片完成 {fname}: {got}/{size}" + (" ✅" if got == size else " ❌"))
            return got == size
        except Exception as e:
            log(f"小分片 {fname} 中断({attempt}): {type(e).__name__}")
            time.sleep(2)
    return False

CDN = {}
for fname in FILES:
    u = resolve_cdn(fname)
    if u:
        CDN[fname] = u
        log(f"CDN: {fname} -> {u[:90]}")
    else:
        log(f"FATAL: 无法解析 {fname}")
        sys.exit(1)

def dl_chunk(fname, ci):
    import random
    size = FILES[fname]
    n = NCHUNK
    cs, ce = size * ci // n, size * (ci + 1) // n - 1
    part = os.path.join(DIR, fname + f".part{ci}")
    url = CDN[fname]
    fails = 0
    while True:
        cur = os.path.getsize(part) if os.path.exists(part) else 0
        if cur >= ce - cs + 1:
            log(f"完成 {fname} 块{ci}")
            return
        req = urllib.request.Request(url, headers={
            "Range": f"bytes={cs + cur}-{ce}",
            "User-Agent": "curl/8"})
        try:
            with opener().open(req, timeout=45) as r, open(part, "ab") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
            fails = 0
        except Exception as e:
            fails += 1
            log(f"块{ci} {fname} 中断({fails}): {type(e).__name__} 已下 {(cs+cur)//1024//1024}/{ce//1024//1024}MB")
            if fails % 8 == 0:  # 怀疑 URL 过期,重新解析
                u = resolve_cdn(fname)
                if u:
                    CDN[fname] = u
                    url = u
            time.sleep(random.uniform(0.5, 4))  # 抖动重连,避免同步

def concat(fname):
    size = FILES[fname]
    out = os.path.join(DIR, fname)
    with open(out, "wb") as w:
        for ci in range(NCHUNK):
            with open(os.path.join(DIR, fname + f".part{ci}"), "rb") as p:
                while True:
                    b = p.read(8 << 20)
                    if not b:
                        break
                    w.write(b)
            os.remove(os.path.join(DIR, fname + f".part{ci}"))
    got = os.path.getsize(out)
    log(f"拼接完成 {fname}: {got}/{size}" + (" ✅" if got == size else " ❌ 大小不符!"))
    return got == size

os.makedirs(DIR, exist_ok=True)
log("=== Q4_K_XL 下载开始 ===")
dl_small(*list(SMALL.items())[0])

tasks = [(f, ci) for f in FILES for ci in range(NCHUNK)]
results = {}
def worker(t):
    f, ci = t
    dl_chunk(f, ci)
    results[t] = True

ths = [threading.Thread(target=worker, args=(t,)) for t in tasks]
for t in ths: t.start()
for t in ths: t.join()

ok = True
for f in FILES:
    ok &= concat(f)
log("全部完成 ✅" if ok else "存在大小不符,请检查")
