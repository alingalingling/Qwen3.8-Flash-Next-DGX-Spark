# Qwen3.8-Flash-Next on DGX Spark — 单机部署与调优

**[中文](README.zh.md) | [English](README.md)**


> **单台 NVIDIA DGX Spark(GB10,128 GB 统一内存,无多卡、无集群),180B 混合架构 MoE 模型的量化版完整部署并稳定服务**
> 包含:llama.cpp qwen4exp 分支一键部署、GGUF MTP 头探测器、
> HF 仓库监控器、全套实测基准与 MTP 注入方案。

![status](https://img.shields.io/badge/status-部署完成-green) ![license](https://img.shields.io/badge/license-Apache--2.0-blue) ![model](https://img.shields.io/badge/model-Qwen3.8--Flash--Next-orange) ![single](https://img.shields.io/badge/单机-单台DGX_Spark-blue)

> **项目时效性说明**
>
> 本项目创建于 **Qwen3.8-Flash-Next 开源发布当天(2026-08-26)**,所有部署、实测与结论均基于**当时社区生态的最新状态**:
> llama.cpp qwen4exp 支持(PR #27742 分支)、unsloth Dynamic 3.0 全部量化档位、当时已出现的全部推理引擎与量化仓库。
>
> 模型发布后的生态仍在快速繁荣:llama.cpp 官方支持即将合并、MTP 投机解码即将落地、
> Baekpica 混合量化(含 SSD-PLE 方案)与全量 NVFP4 正在验证与发布中。
> **因此本项目将持续更新**——随生态演进,README 结论、docs 实测数据、工具脚本与路线图会同步迭代,
> 建议关注本仓库获取 Flash-Next 单机部署的最新方案。

## 这份仓库有什么

一句话:**把"官方要双卡 GB300 才能跑的 180B 模型"在单台 DGX Spark 上部署了起来,
并把整个过程(怎么装、多快、卡在哪、怎么提速)开源。**

- **部署** ： [docs/deploy_playbook.zh.md](docs/deploy_playbook.zh.md):部署手册。从下载、编译到启动的完整命令,版本能否装入的数据信息。
- **真实数据**：  [docs/benchmarks.zh.md](docs/benchmarks.zh.md):我们实测的全部对比数字——7 个量化档位怎么选、GPU 比 CPU 快多少倍、32K/128K/262K 上下文各是什么速度、内存各占多少,连每次测试的原始耗时记录都在里面。
- **让它更快?** ： [docs/speculative-analysis.zh.md](docs/speculative-analysis.zh.md):投机解码(让生成变快的技术)的五条路——ngram / DFlash / DSpark / MTP / 小草稿——我们**全部实测了一遍**,告诉你为什么现在最快只有 24 t/s、瓶颈在哪、未来哪个方案有希望到 40-60 t/s、什么时候能解锁。
- **想跟进加速进展?** → 看 [docs/mtp-tracker.zh.md](docs/mtp-tracker.zh.md):MTP 头(投机加速的关键部件)为什么被量化版删掉了、怎么把它装回去、llama.cpp 的支持进度,以及三个"可以动手了"的信号。
- **想直接开跑?** → 用 `scripts/` 下的工具:`run_qwen38_q3.sh` 一键启停服务,`probe_mtp.py` 3 秒检查你的 GGUF 能不能投机,`monitor.py` 帮你盯着 HF 上有没有新版本。

## 背景

Qwen3.8-Flash-Next 是 Qwen4 架构的预览版(180B 参数混合 MoE,512 专家 top-10 路由,
GDN 线性注意力 + QSA 稀疏注意力 + 51B PLE n-gram 查表,262K 原生上下文)。
官方验证环境为双卡 GB300/B300,本仓库证明:**单台 DGX Spark 即可部署其量化版**。

本仓库记录并开源了在 **单台 NVIDIA DGX Spark(GB10,128 GB 统一内存)** 上部署它的
全部工具与实测数据——包括目前问题所在(MTP 缺失)和解决方案。

## 部署结果(实测)

| 项 | 值 |
|---|---|
| 模型 | unsloth UD-Q3_K_XL(90 GB,质量保持 90.4%) |
| 引擎 | llama.cpp qwen4exp 分支(PR #27742) |
| 上下文 | **262K(原生最大)** |
| 解码速度 | **22-24 t/s**(GPU 卸载,纯 CPU 仅 1.9 t/s) |
| 内存 | ~102 / 128 GB |

## 量化档位

| 档位                    | 大小           | 与 BF16 一致率 | 内存需求      | 128 GB 判定 |
| --------------------- | ------------ | ---------- | --------- | --------- |
| UD-Q4_K_XL(4-bit)     | 111.3 GB     | 93.5%      | 112 GB    | 超预算       |
| UD-IQ4_XS(4-bit)      | 93.7 GB      | 91.1%      | 112 GB    | 临界        |
| **UD-Q3_K_XL(3-bit)** | **90.0 GB**  | **90.4%**  | **90 GB** | 选用        |
| UD-IQ3_XXS(3-bit)     | 82.0 GB      | 87.6%      | 90 GB     | 备选        |
| UD-Q2_K_XL(2-bit)     | 78.9 GB      | 85.2%      | 79 GB     | 有取舍       |
| UD-IQ1 系(1-bit)       | 72.5-74.5 GB | 80-82%     | 75 GB     | 质量崩坏      |

## 文档导航(建议阅读顺序)

| 文档                                                           | 内容                                    | 适合谁       |
| ------------------------------------------------------------ | ------------------------------------- | --------- |
| [docs/deploy_playbook.zh.md](docs/deploy_playbook.zh.md)           | 部署手册:验收公式、三条运行路径、质量红线                 | 想部署的人     |
| [docs/benchmarks.zh.md](docs/benchmarks.zh.md)                     | 全套实测对比:量化档位/引擎/上下文/投机/原始 timings      | 想验证/对比的人  |
| [docs/speculative-analysis.zh.md](docs/speculative-analysis.zh.md) | ⭐ 投机解码全景实测分析:五条路径逐一验证、接受率数学、因果链、解锁路线图 | 想提速/研究的人  |
| [docs/mtp-tracker.zh.md](docs/mtp-tracker.zh.md)                   | MTP 头追踪:llama.cpp PR 状态、注入方案、监控信号     | 想跟进加速进展的人 |

## 快速开始

```bash
# 1. 下载模型(90GB)
hf download unsloth/Qwen3.8-Flash-Next-GGUF \
  --include "UD-Q3_K_XL/*" --local-dir ~/models/Qwen3.8-Flash-Next-GGUF/UD-Q3_K_XL

# 2. 构建 llama.cpp(qwen4exp 分支,PR #27742)
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp && cd ~/llama.cpp
git fetch origin pull/27742/head:qwen4exp && git checkout qwen4exp
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j $(nproc)

# 3. 启动服务(OpenAI 兼容,端口 8889,262K 上下文)
bash scripts/run_qwen38_q3.sh start

# 4. 验证
curl http://127.0.0.1:8889/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8","messages":[{"role":"user","content":"你好"}],"max_tokens":100,"chat_template_kwargs":{"enable_thinking":false}}'
```

## 目录结构

```
├── scripts/
│   ├── run_qwen38_q3.sh      # 服务启停脚本(start/stop/status/logs)
│   ├── qwen38-q3.service     # systemd 开机自启模板
│   ├── probe_mtp.py          # ⭐ GGUF MTP 头探测器(纯 Python,零依赖)
│   └── monitor.py            # HF 量化仓库监控器(含 llama.cpp PR 状态)
├── docs/
│   ├── deploy_playbook.md        # 部署手册(验收公式/三条运行路径/质量红线)
│   ├── benchmarks.md             # 全套实测基准(速度/上下文/内存/投机)
│   ├── speculative-analysis.md   # ⭐ 投机解码全景实测分析(五条路径逐一验证+因果链+路线图)
│   └── mtp-tracker.md            # MTP 头追踪 + 注入方案
└── LICENSE                   # Apache-2.0
```

## 核心脚本说明

### probe_mtp.py — 判断你的 GGUF 能不能投机
投机解码(draft-mtp)需要 GGUF 内含 MTP 头。unsloth 的 Flash-Next 量化版移除了它
(180B → 176.94B 参数),而 27B 的 GGUF 自带。用本工具 3 秒判断:
```bash
python3 scripts/probe_mtp.py /path/to/model.gguf   # 或分片目录
```

### monitor.py — 盯住新版本
双通道扫描 HF(官方 quantized tag + 名称搜索),自动分类:
`DEPLOYABLE / UPLOADING / TOO_BIG / MLX_ONLY / Q1_BAD`,含 llama.cpp PR 状态。

## 已知结论

1. **投机解码当前不可用**:五条路径全部实测验证(GGUF 无 MTP 头 / 无 DFlash 结构 / 无训练过的同架构小草稿 / llama.cpp MTP 支持 WIP)——24 t/s 是当前天花板。**完整实测分析见 docs/speculative-analysis.zh.md**
2. **后台服务用 setsid 启动**:防止终端超时把服务一起杀掉
3. **上下文大胆开**:QSA 稀疏 KV 让 262K 的内存增量只有几 GB;但**大窗口会拖慢短请求的 prompt 处理**(262K 下 36.5 vs 128K 下 80.8 tok/s),短对话场景建议 128K

## 路线图

- [ ] llama.cpp PR #27742 合并(上游)
- [ ] MTP 支持落地 → 提供 inject_mtp.py(从官方 BF16 注入 MTP 头)
- [ ] Baekpica 混合量化版验证门通过后的适配评估(ds4 运行时)
- [ ] 全量 NVFP4 出现后的 SGLang/DFlash2 对比

## 致谢

- [unsloth](https://huggingface.co/unsloth) — Dynamic 3.0 GGUF
- [llama.cpp](https://github.com/ggml-org/llama.cpp) PR #27742(daniehanchen)

## 许可证

Apache-2.0。模型权重属 Qwen Community License,请从原始仓库获取,本仓库不打包模型文件。
