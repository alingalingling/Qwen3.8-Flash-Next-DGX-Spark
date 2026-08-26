#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probe_mtp.py — 检查 GGUF 模型是否携带 MTP(Multi-Token Prediction)头

投机解码(draft-mtp)需要 GGUF 内含 MTP 头张量。unsloth 的 Qwen3.8-Flash-Next
量化版默认移除 MTP 头(180B → 176.94B 参数),而 Qwen3.8-27B 的 GGUF 自带。

用法:
    python3 probe_mtp.py model.gguf
    python3 probe_mtp.py --dir /path/to/shards/   # 自动找 00001-of-XXXX 分片

输出: MTP 张量清单 + 结论(可投机 / 不可投机 / 需注入)
"""
import os, struct, sys

MAGIC = b"GGUF"

def read_string(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", "replace")

def skip_value(f, vtype):
    """跳过 GGUF 元数据中的一个值;返回 None"""
    if vtype == 0: f.read(1)            # uint8
    elif vtype == 1: f.read(1)          # int8
    elif vtype == 2: f.read(2)          # uint16
    elif vtype == 3: f.read(2)          # int16
    elif vtype == 4: f.read(4)          # uint32
    elif vtype == 5: f.read(4)          # int32
    elif vtype == 6: f.read(4)          # float32
    elif vtype == 7: f.read(1)          # bool
    elif vtype == 8: read_string(f)     # string
    elif vtype == 9:                    # array
        atype = struct.unpack("<I", f.read(4))[0]
        count = struct.unpack("<Q", f.read(8))[0]
        for _ in range(count):
            skip_value(f, atype)
    elif vtype == 10: f.read(8)         # uint64
    elif vtype == 11: f.read(8)         # int64
    elif vtype == 12: f.read(8)         # float64
    else:
        raise ValueError(f"未知 GGUF 值类型 {vtype}")

def scan_tensor_names(path):
    """解析 GGUF 头部,返回 (tensor_names, params_hint)"""
    with open(path, "rb") as f:
        if f.read(4) != MAGIC:
            raise ValueError("不是有效的 GGUF 文件")
        version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n_kv):
            read_string(f)                     # key
            vtype = struct.unpack("<I", f.read(4))[0]
            skip_value(f, vtype)
        names = []
        for _ in range(n_tensors):
            name = read_string(f)
            n_dims = struct.unpack("<I", f.read(4))[0]
            f.read(8 * n_dims)                 # dims (GGUF v3 = uint64)
            f.read(4)                          # type
            f.read(8)                          # offset
            names.append(name)
        return names, n_tensors

def find_first_shard(directory):
    for fn in sorted(os.listdir(directory)):
        if fn.endswith(".gguf") and "-00001-of-" in fn:
            return os.path.join(directory, fn)
    ggs = [fn for fn in sorted(os.listdir(directory)) if fn.endswith(".gguf")]
    return os.path.join(directory, ggs[0]) if ggs else None

def find_shards(path):
    """给定一个分片,返回同目录全部 GGUF 分片(按序)"""
    d = os.path.dirname(path) or "."
    return [os.path.join(d, fn) for fn in sorted(os.listdir(d))
            if fn.endswith(".gguf")]

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    target = sys.argv[1]
    if os.path.isdir(target):
        path = find_first_shard(target)
        if not path:
            print(f"目录 {target} 中未找到 GGUF 分片"); sys.exit(1)
    else:
        path = target

    # 多分片模型:元数据头(00001)通常无张量,需扫描全部兄弟分片
    shards = find_shards(path) if os.path.isfile(path) else [path]
    print(f"扫描 {len(shards)} 个分片: {path}")

    total_tensors, mtp = 0, []
    for sp in shards:
        try:
            names, n = scan_tensor_names(sp)
        except Exception as e:
            print(f"  跳过 {os.path.basename(sp)}: {e}")
            continue
        total_tensors += n
        mtp += [name for name in names if "mtp" in name.lower() or "multi" in name.lower()]

    print(f"张量总数: {total_tensors}")
    if mtp:
        print(f"\n✅ 发现 {len(mtp)} 个 MTP 相关张量 → 可尝试 --spec-type draft-mtp:")
        for n in mtp[:10]:
            print(f"   {n}")
    else:
        print("\n❌ 未发现 MTP 张量 → draft-mtp 不可用")
        print("   推测: 该 GGUF 的 MTP 头已被移除(如 unsloth Flash-Next 量化版)")
        print("   出路: ① 等带 MTP 的量化版发布 ② 用 inject_mtp.py 从官方 BF16 注入(见 docs/mtp-tracker.md)")
    return 0 if mtp else 3

if __name__ == "__main__":
    sys.exit(main())
