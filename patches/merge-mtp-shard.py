#!/usr/bin/env python3
"""Attach an MTP draft head to a split qwen4exp GGUF as an extra shard.

  merge-mtp-shard.py <target shard 1> <head gguf> <out dir>

Writes shard 1 anew (block_count, nextn_predict_layers, compress_ratios, split.count,
split.tensors.count) and a last shard holding the head's blk.N.* tensors, raw bytes
copied; the target's other shards are hard-linked under the new -of-0000M names.
"""
import os, re, sys
sys.path.insert(0, os.environ.get("GGUF_PY", "/work/fn-tree/gguf-py"))
import gguf
from gguf import GGUFReader, GGUFWriter, GGUFValueType

src1, head_path, out_dir = sys.argv[1:4]
m = re.match(r"(.*)-(\d{5})-of-(\d{5})\.gguf$", os.path.basename(src1))
prefix, n_old = m.group(1), int(m.group(3))
n_new = n_old + 1
src_dir = os.path.dirname(src1)
os.makedirs(out_dir, exist_ok=True)

r1 = GGUFReader(src1)
head = GGUFReader(head_path)
arch = bytes(r1.fields["general.architecture"].parts[-1]).decode()
head_tensors = [t for t in head.tensors if t.name.startswith("blk.")]
n_layer = int(r1.fields[f"{arch}.block_count"].parts[-1][0])
n_nextn = len({t.name.split(".")[1] for t in head_tensors})
assert n_nextn == 1, n_nextn
assert all(t.name.startswith(f"blk.{n_layer}.") for t in head_tensors), "head layer index must equal block_count"
old_total = int(r1.fields["split.tensors.count"].parts[-1][0])

def copy_fields(reader, writer, override, skip=()):
    for f in reader.fields.values():
        if f.name.startswith("GGUF.") or f.name in skip:
            continue
        if f.name in override:
            val, vt, st = override[f.name]
        else:
            vt = f.types[0]; st = f.types[-1] if vt == GGUFValueType.ARRAY else None; val = f.contents()
        try:
            writer.add_key_value(f.name, val, vt, sub_type=st)
        except ValueError as e:  # e.g. the writer already set general.architecture
            if "duplicat" not in str(e).lower(): raise

# shard 1: metadata only
w1 = GGUFWriter(os.path.join(out_dir, f"{prefix}-00001-of-{n_new:05d}.gguf"), arch=arch, endianess=r1.endianess)
ratios = [int(x) for x in r1.fields[f"{arch}.attention.compress_ratios"].contents()]
ov = {
    f"{arch}.block_count": (n_layer + n_nextn, GGUFValueType.UINT32, None),
    f"{arch}.attention.compress_ratios": (ratios + [0] * n_nextn, GGUFValueType.ARRAY, r1.fields[f"{arch}.attention.compress_ratios"].types[-1]),
    "split.count": (n_new, GGUFValueType.UINT16, None),
    "split.tensors.count": (old_total + len(head_tensors), GGUFValueType.INT32, None),
}
copy_fields(r1, w1, ov)
w1.add_key_value(f"{arch}.nextn_predict_layers", n_nextn, GGUFValueType.UINT32)
assert len(r1.tensors) == 0, "shard 1 is expected to carry metadata only"
w1.write_header_to_file(); w1.write_kv_data_to_file(); w1.write_ti_data_to_file(); w1.close()
print(f"shard 1: block_count {n_layer}->{n_layer+n_nextn}, nextn_predict_layers {n_nextn}, "
      f"compress_ratios {len(ratios)}->{len(ratios)+n_nextn}, split.count {n_old}->{n_new}, tensors {old_total}->{old_total+len(head_tensors)}")

# middle shards: hard links under the new names
for i in range(2, n_old + 1):
    s = os.path.join(src_dir, f"{prefix}-{i:05d}-of-{n_old:05d}.gguf"); d = os.path.join(out_dir, f"{prefix}-{i:05d}-of-{n_new:05d}.gguf")
    if os.path.exists(d): os.unlink(d)
    os.link(s, d)

# last shard: the head's block tensors, raw
wN = GGUFWriter(os.path.join(out_dir, f"{prefix}-{n_new:05d}-of-{n_new:05d}.gguf"), arch=arch, endianess=head.endianess)
wN.add_key_value("split.no", n_new - 1, GGUFValueType.UINT16)
wN.add_key_value("split.count", n_new, GGUFValueType.UINT16)
wN.add_key_value("split.tensors.count", old_total + len(head_tensors), GGUFValueType.INT32)
for t in head_tensors:
    wN.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
wN.write_header_to_file(); wN.write_kv_data_to_file(); wN.write_ti_data_to_file()
for t in head_tensors:
    wN.write_tensor_data(t.data, tensor_endianess=head.endianess)
wN.close()
print(f"shard {n_new}: {len(head_tensors)} tensors, {sum(t.data.nbytes for t in head_tensors)/2**30:.2f} GiB")
