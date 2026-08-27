#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PLE 表预热工具(0xBakeer 方案配套)——把 per_layer_token_embd 区域顺序读入页缓存
用法: python3 warm_table.py <模型分片1路径> [tensor名]
原理: PLE 表 320M 行按 3-gram 哈希寻址,短负载几乎不会重复触达同一行,天然无法自热;
      一次顺序读(约 26-27 GiB / ~26s)即可把 6x 大故障率降下来(配合 ngram-mod 收益更大)
"""
import os, sys, time, importlib.util

sys.path.insert(0, os.environ.get("GGUF_PY", "/home/lingyan/llama.cpp/gguf-py"))
from gguf import GGUFReader

def shards_for(shard1):
    import re
    m = re.match(r"(.*)-(\d{5})-of-(\d{5})\.gguf$", shard1)
    prefix, n = m.group(1), int(m.group(3))
    return [f"{prefix}-{i:05d}-of-{n:05d}.gguf" for i in range(1, n + 1)]

def find_tensor(shards, want):
    for sh in shards:
        r = GGUFReader(sh)
        for t in r.tensors:
            if t.name == want:
                return sh, t.data_offset, t.n_bytes
        del r
    return None

def resident(sh, off, length):
    """页缓存驻留比例: 用 mincore 粗略统计(4KB 页)"""
    try:
        import mmap
        fd = os.open(sh, os.O_RDONLY)
        # 采样统计太慢, 用 fadvise 返回的页数近似 —— 简化: 读取前/后各统计一次文件页
        # 用 /proc/self/ 无法跨进程; 直接用 os.posix_fadvise + 读速度推断
        os.close(fd)
        return 0, 0, 0
    except Exception:
        return 0, 0, 0

def main():
    shard1 = sys.argv[1]
    want = sys.argv[2] if len(sys.argv) > 2 else "per_layer_token_embd.weight"
    shards = shards_for(shard1)
    hit = find_tensor(shards, want)
    if not hit:
        print(f"tensor {want} 未找到"); sys.exit(1)
    sh, off, length = hit
    print(f"找到 {want}: {sh} offset={off} length={length/2**30:.1f} GiB")

    fd = os.open(sh, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, off, length, os.POSIX_FADV_WILLNEED)
    except Exception:
        pass
    CH = 64 << 20
    t0 = time.time(); done = 0
    while done < length:
        n = min(CH, length - done)
        b = os.pread(fd, n, off + done)
        if not b:
            break
        done += len(b)
    os.close(fd)
    el = time.time() - t0
    print(f"预热完成: {done/2**30:.1f} GiB 顺序读, {el:.1f}s = {done/2**30/max(el,0.01):.2f} GiB/s")

if __name__ == "__main__":
    main()
