#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3.8-Flash-Next 量化版监控器
目标:在 HuggingFace 上发现"可以在 DGX Spark(128 GB 统一内存)部署"的版本:
  - 权重总大小 <= 95 GB(留出 KV cache + 系统开销)
  - NVIDIA 可加载格式(GGUF / NVFP4 / FP8 safetensors,排除 MLX=Apple 专属)
  - 质量 >= 3-bit(排除 IQ1/UD-IQ1/oQ1/Q1 等 1-bit 垃圾质量)
每次运行输出当前状态 + 与 baseline.json 的差异;新发现可部署版本时单独高亮。
"""
import json, os, re, sys, time, urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
BASELINE = os.path.join(BASE, "baseline.json")
TAG_URL = "https://huggingface.co/api/models?filter=base_model:quantized:Qwen/Qwen3.8-Flash-Next&limit=200"
SEARCH_URL = "https://huggingface.co/api/models?search=Qwen3.8-Flash-Next&limit=200&full=true"
WEIGHT_EXT = (".safetensors", ".gguf")
SIZE_BUDGET_GB = 95.0     # 权重预算(GB)
TIGHT_GB = 105.0          # 临界线:超过则运行时基本无 KV 空间
BAD_Q1 = re.compile(r"(iq1|ud[-_]?iq1|oq1|q1_|_i1_|ternary|1\.58)", re.I)
CAUTION_Q2 = re.compile(r"(oq2|q2_|iq2|2\.5bpw)", re.I)
MLX_MARK = re.compile(r"mlx|apple", re.I)
SHARD = re.compile(r"-(\d{4,5})-of-(\d{4,5})", re.I)

def get_json(url, tries=2):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "monitor/1.0"})
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.load(r)
        except Exception as e:
            if i == tries - 1:
                return None
            time.sleep(3)
    return None

def classify(repo_id, files):
    """files: [(name, size_bytes)] -> dict"""
    rec = {"id": repo_id, "total_gb": 0.0, "n": len(files),
           "gguf": [], "mlx": False, "nvfp4": False, "fp8": False,
           "bf16_gb": 0.0, "q1": False, "q2": False, "empty": len(files) == 0,
           "projected_gb": 0.0, "uploading": False, "shard_total": 0}
    low = repo_id.lower()
    if MLX_MARK.search(low):
        rec["mlx"] = True
    shard_totals = []  # (seen_ratio, projected_ratio_total)
    for name, size in files:
        n = name.lower()
        if n.endswith(".gguf"):
            rec["gguf"].append(name.split("/")[-1])
        if n.endswith(".safetensors"):
            if "nvfp4" in n or "fp4" in n:
                rec["nvfp4"] = True
            if "fp8" in n:
                rec["fp8"] = True
            if "bf16" in n:
                rec["bf16_gb"] += size / 1e9
        rec["total_gb"] += size / 1e9
        m = SHARD.search(name)
        if m:
            idx, total = int(m.group(1)), int(m.group(2))
            shard_totals.append((idx, total, size / 1e9))
    # 分片上传中检测:文件名含 -0000X-of-0000Y,且未见全部片
    if shard_totals:
        seen = {s[0] for s in shard_totals}
        tot_n = max(s[1] for s in shard_totals)
        rec["shard_total"] = tot_n
        if tot_n > len(seen):
            rec["uploading"] = True
            avg_shard = sum(s[2] for s in shard_totals) / len(shard_totals)
            rec["projected_gb"] = round(avg_shard * tot_n, 1)
        else:
            rec["projected_gb"] = rec["total_gb"]
    blob = " ".join(rec["gguf"]) + " " + repo_id.lower()
    if BAD_Q1.search(blob):
        rec["q1"] = True
    if CAUTION_Q2.search(blob):
        rec["q2"] = True
    return rec

def verdict(rec):
    """返回 (状态, 理由)"""
    if rec["empty"]:
        return "PLACEHOLDER", "仓库已建,文件未上传(可能正在传)"
    if rec["uploading"]:
        return "UPLOADING", "上传中:已见 %.1f GB,预计总量 %.1f GB(共 %d 分片)" % (
            rec["total_gb"], rec["projected_gb"], rec["shard_total"])
    if rec["total_gb"] > TIGHT_GB:
        return "TOO_BIG", "%.0f GB 超出 128 GB 预算" % rec["total_gb"]
    if rec["mlx"]:
        return "MLX_ONLY", "Apple Silicon 专属格式,GPU 跑不了"
    if rec["q1"]:
        return "Q1_BAD", "1-bit 量化,质量很差(用户明确排除)"
    if rec["total_gb"] > SIZE_BUDGET_GB:
        return "TOO_TIGHT", "%.0f GB > 95 GB 预算,无 KV 空间" % rec["total_gb"]
    if rec["bf16_gb"] > 25:
        return "INFLATE", "含 %.0f GB BF16 大件,加载可能膨胀" % rec["bf16_gb"]
    if rec["gguf"] or rec["nvfp4"] or rec["fp8"]:
        return "DEPLOYABLE", "✓ 可以部署!"
    return "UNKNOWN", "格式未识别"

def check_llamacpp_pr():
    """检查 llama.cpp 对 qwen4_exp 的支持 PR 状态"""
    pr = get_json("https://api.github.com/repos/ggml-org/llama.cpp/pulls/27742")
    if pr is None:
        return "llama.cpp PR 状态获取失败"
    state = pr.get("state")
    merged = pr.get("merged")
    if merged:
        return "llama.cpp 已合并 qwen4_exp 支持 PR(#27742)! GGUF 立即可跑"
    return "llama.cpp qwen4_exp PR#27742: state=%s, mergeable=%s, updated=%s" % (
        state, pr.get("mergeable"), (pr.get("updated_at") or "")[:16])

def main():
    # 双通道:tag 过滤(用户网页视图)+ 名称搜索超集,取并集
    seen, data = {}, []
    for url in (TAG_URL, SEARCH_URL):
        d = get_json(url)
        if d:
            for m in d:
                rid = m.get("id", "")
                if rid and rid not in seen:
                    seen[rid] = m
                    data.append(m)
    if data is None:
        print("FETCH_FAILED")
        sys.exit(2)
    repos = []
    for m in data:
        rid = m.get("id", "")
        if "Qwen3.8-Flash-Next" not in rid:
            continue
        b = get_json("https://huggingface.co/api/models/%s?blobs=true" % rid)
        if b is None:
            repos.append({"id": rid, "total_gb": -1, "n": 0, "verdict": "FETCH_FAIL",
                          "reason": "blobs 接口失败"})
            continue
        files = [(f["rfilename"], f.get("size", 0))
                 for f in b.get("siblings", [])
                 if f["rfilename"].endswith(WEIGHT_EXT)]
        rec = classify(rid, files)
        st, why = verdict(rec)
        repos.append({"id": rid, "total_gb": round(rec["total_gb"], 1),
                      "n": rec["n"], "q1": rec["q1"], "mlx": rec["mlx"],
                      "verdict": st, "reason": why,
                      "gguf": rec["gguf"], "created": (m.get("createdAt") or "")[:10]})
    repos.sort(key=lambda r: r["total_gb"])

    # 与基线对比
    baseline = {}
    if os.path.exists(BASELINE):
        baseline = {r["id"]: r for r in json.load(open(BASELINE))}
    new_ids = [r["id"] for r in repos if r["id"] not in baseline]
    changed = [r for r in repos if r["id"] in baseline and
               (baseline[r["id"]].get("total_gb", -1) != r["total_gb"])]

    print("== 当前 %d 个相关仓库 ==" % len(repos))
    for r in repos:
        flag = ""
        if r["verdict"] == "DEPLOYABLE":
            flag = "  <<<<<<<< DEPLOYABLE"
        print("%-52s %8.1f GB  %-11s %s%s" % (r["id"], r["total_gb"], r["verdict"], r["reason"], flag))
    if new_ids:
        print("\n[NEW] 新出现仓库:", ", ".join(new_ids))
    if changed:
        for r in changed:
            print("[CHANGED] %s: %.1f GB -> %.1f GB" % (r["id"], baseline[r["id"]].get("total_gb", -1), r["total_gb"]))

    deployable = [r for r in repos if r["verdict"] == "DEPLOYABLE"]
    if deployable:
        print("\n!!! 发现可部署版本 !!!")
        for r in deployable:
            print("   ", r["id"], r["total_gb"], "GB", r["reason"])
    print("\n== llama.cpp 支持状态 ==")
    print("  " + check_llamacpp_pr())
    json.dump(repos, open(BASELINE, "w"), ensure_ascii=False, indent=1)
    return 0 if not deployable else 3

if __name__ == "__main__":
    sys.exit(main())
