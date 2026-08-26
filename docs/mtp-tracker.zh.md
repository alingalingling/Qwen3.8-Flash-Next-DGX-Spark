# MTP 头追踪与注入方案(Qwen3.8-Flash-Next)

**[中文](mtp-tracker.zh.md) | [English](mtp-tracker.md)**


> 背景:投机解码(draft-mtp)可大幅提升解码速度 。
> 前提:GGUF 必须携带 MTP 头张量。unsloth 的 Flash-Next 量化版**默认移除** MTP 头(180B → 176.94B 参数,probe_mtp.py 已验证)。

## 当前状态(2026-08-27)

| 项 | 状态 |
|---|---|
| llama.cpp qwen4exp PR(#27742)| open、可合并;**20 commits,无 MTP 提交**;作者 WIP |
| 前身 PR #27739 | closed,但**已含 MTP + PLE 卸载实现**,社区已提醒作者参考 |
| unsloth GGUF | 全部 7 档无 MTP 头;仓库无独立 MTP 模块 |
| 独立草稿模型 | 不存在训练过的 qwen4exp 小模型(0.2B 测试件未训练) |

## 解锁条件(满足其一即可开启投机)

1. **llama.cpp PR 完成 MTP 支持** + unsloth 发布带 MTP 的量化版 → 直接 `--spec-type draft-mtp --spec-draft-n-max 2`
2. **自行注入 MTP 头**(llama.cpp 支持后):见下方方案

## 注入方案(等 llama.cpp MTP 支持落地后执行)

```bash
# 1) 下载官方 BF16(360 GB,磁盘需充足)
hf download Qwen/Qwen3.8-Flash-Next --local-dir ~/models/Qwen3.8-Flash-Next-BF16

# 2) 用 qwen4exp 转换器转成 GGUF(需 PR 分支的 convert_hf_to_gguf.py)
python3 convert_hf_to_gguf.py ~/models/Qwen3.8-Flash-Next-BF16 \
  --outfile ~/models/flashnext-bf16.gguf --outtype q8_0   # 180GB 中间件

# 3) 从中间件抽取 31 个 MTP 张量(*mtp*),Q4_0 量化(约 2GB)

# 4) 用 gguf-py 库把 MTP 张量注入 unsloth 的 UD-Q3_K_XL GGUF
#    (更新 metadata 张量列表 + 追加数据 + 重写分片)

# 5) 验证 + 启动
python3 scripts/probe_mtp.py ~/models/Qwen3.8-Flash-Next-GGUF/UD-Q3_K_XL/
llama-server -m model.gguf --spec-type draft-mtp --spec-draft-n-max 2 ...
```

> 注:完整注入脚本(inject_mtp.py)将在 llama.cpp MTP 支持合入后提供;
> 届时先检查 unsloth 是否已发布带 MTP 版本(优先用官方,免注入)。

## 监控信号(仓库内 monitor.py 已跟踪)

- llama.cpp PR #27742 出现 MTP 相关 commit / 合并
- unsloth 仓库出现带 MTP 的文件名(如 *-MTP-* 或 mtp.gguf)
- Baekpica 混合量化版通过验证门(其 MQ 版本含 MTP 专家 Q8_0)

## 参考

- [llama.cpp PR #27742](https://github.com/ggml-org/llama.cpp/pull/27742)
- [llama.cpp PR #27739](https://github.com/ggml-org/llama.cpp/pull/27739)(含 MTP 实现)
