# MTP 头追踪与注入方案(Qwen3.8-Flash-Next)

**[中文](mtp-tracker.zh.md) | [English](mtp-tracker.md)**

> 背景:投机解码(draft-mtp)可大幅提升解码速度。
> 前提:GGUF 必须携带 MTP 头张量。unsloth 的 Flash-Next 量化版**默认移除** MTP 头(180B → 176.94B 参数,probe_mtp.py 已验证)。

## 当前状态(2026-08-27 夜更新)

| 项 | 状态 |
|---|---|
| llama.cpp qwen4exp PR(#27742)| open、可合并;**42 commits,仍无 MTP 提交**;作者 WIP |
| 前身 PR #27739 | closed,但**已含 MTP + PLE 卸载实现**,社区已提醒作者参考 |
| **dzannotti MTP 头**([仓库](https://huggingface.co/dzannotti/Qwen3.8-Flash-Next-MTP-GGUF))| **重大突破**:独立 `MTP-Q4_K_M.gguf`(2.44GB,标准量化,任意后端)+ `qwen4exp-mtp-draft-head.patch`(459 行,PR #27739 MTP 图调和版 + converter `--mtp` 导出) |
| **本机构建** | 补丁在本机 qwen4exp 分支(基线 bea3b12d 与补丁目标树一致)**干净应用,重编译成功**——llama-server 现支持 `--spec-type draft-mtp`(2026-08-27 19:16 构建) |
| unsloth GGUF | 全部 7 档无 MTP 头;dzannotti 独立头可作为 `-md` 草稿 |
| 独立草稿模型 | 不存在训练过的 qwen4exp 小模型;**MTP 草稿头(4B,官方 BF16 导出)即正确草稿** |

## 作者实测数据(AMD Strix Halo 128GB 统一内存——与 DGX Spark 同级别)

| 后端,目标模型 | 裸跑 | + MTP | 接受率(code / prose) |
|---|---:|---:|---|
| ROCm, UD-Q4_K_XL | 20.3 t/s | **35.8** code / 22.6 prose | 0.90 / 0.74 |
| ROCm, UD-IQ4_XS | 18.0 / 18.6 | 32.8 / 22.1 | 0.84 / 0.68 |
| Vulkan (RADV, Laurent fork), UD-IQ4_XS | 24.2 / 24.3 | **37.2** code / **30.3** prose | 0.88 / 0.82 |

推荐参数:`--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.75`
(p-min 0.75 比加深深度更重要:草稿头自带 MoE,每个草稿 token 都是真实前向)。
**必须设置 `LLAMA_ATTN_ROT_DISABLE=1`**(上游 quantized-KV 旋转 #21038 不被 qwen4exp attention 路径支持,不设则加载即 abort)。

## 运行方式(独立草稿头)

```bash
LLAMA_ATTN_ROT_DISABLE=1 llama-server \
  -m Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf \
  -md Qwen3.8-Flash-Next-MTP-Q4_K_M.gguf -ngld 999 \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.75 \
  -ngl 999 -fa on -ctk q8_0 -ctv q8_0 -c 262144
```

## 注入方案(免重下 82/90GB 模型——merge-mtp-shard.py)

`patches/merge-mtp-shard.py`(来自 dzannotti 仓库,已归档到本项目)可把 MTP 头作为
**任意现有分片 qwen4exp GGUF** 的附加分片:

1. 保留目标模型分片 2..N,重命名为 `-0000N-of-0000(M+1).gguf`
2. 放入重写后的分片 1(元数据:`block_count 49`、`nextn_predict_layers 1`、`compress_ratios` 扩展、`split.count M+1`)和新末分片(29 个头张量,与 Q4_K_M 头同字节)
3. 用 `--spec-type draft-mtp` 且**不带 `-md`** 运行;草稿上下文基于目标模型创建并共享内存(比独立头省 ~0.5GB)

实测:与 `-md` 形式仅差几个百分点(34.5 vs 36.7 t/s,code 场景)——便利性是重点,非速度。

## 本机行动计划

1. IQ3_XXS(82GB)下载中(~25MB/s)→ 完成后测速链跑 4 组:
   IQ3 基线 / IQ3 + cafe fork + quimmedes 草稿 / Q3_K_XL 基线 / **IQ3 + MTP 官方头**(新增组)
2. MTP-Q4_K_M 头(2.44GB)下载中
3. 验证通过后,可选:用 merge-mtp-shard.py 给 IQ3_XXS(甚至 Q3_K_XL)内嵌 MTP 头

## 监控信号(仓库内 monitor.py 已跟踪)

- llama.cpp PR #27742 出现 MTP 相关 commit / 合并
- unsloth 仓库出现带 MTP 的文件名(如 *-MTP-* 或 mtp.gguf)
- Baekpica 混合量化版通过验证门(其 MQ 版本含 MTP 专家 Q8_0)
- dzannotti 仓库更新(头重量化、更佳补丁)

## 参考

- [llama.cpp PR #27742](https://github.com/ggml-org/llama.cpp/pull/27742)
- [llama.cpp PR #27739](https://github.com/ggml-org/llama.cpp/pull/27739)(含 MTP 实现)
- [dzannotti/Qwen3.8-Flash-Next-MTP-GGUF](https://huggingface.co/dzannotti/Qwen3.8-Flash-Next-MTP-GGUF)(独立 MTP 头 + 补丁 + 合并脚本)
