---
title: "vLLM 与 CUDA 模型部署：从显存到性能排障"
date: "2026-09-07"
description: "以 Qwen3-4B-Instruct-2507 和 24GiB 级 NVIDIA 显卡为教学样例，沿餐饮 RAG 请求链解释 CUDA、模型文件、vLLM 调度、显存预算、压测与生产排障。"
tags: ["模型部署", "vLLM", "CUDA", "LLM推理", "性能工程"]
---

很多“模型已经部署”的项目，上线后才暴露问题：驱动可见但服务加载失败；单请求很快，并发就排队；权重只有几 GiB，长上下文却 OOM；方案里的 `Triton` 也常分不清是语言还是服务器。

把问题放回一条具体请求：员工询问三号店牛肉面是否缺货、有什么替代菜，系统检索库存和菜品文档后交给 Qwen3-4B-Instruct-2507。沿着这条链看，显卡资源、版本锁定和延迟指标就不再是孤立的部署参数。

文中的命令是面向 Linux NVIDIA 主机的可复制模板。示例硬件是 24GiB 级显卡；下文显存数字为理论预算，启动参数是待在目标环境验证的示例，不代表实测性能。部署时应把已审核的版本写入变量，先在目标平台验收，再推广到镜像或集群。

## 1. 一次推理请求怎样走到 GPU

### 1.1 一次请求经过什么

“推理”是用已经训练好的参数，根据输入 token 逐步计算下一个 token 的概率分布并采样或选择输出。在线服务则还要负责排队、批处理、取消、流式传输、限流和指标。

餐饮 RAG 的链路可以写成：

```text
员工问题
  -> API 鉴权、租户和门店过滤
  -> 查询改写/向量与关键词检索
  -> 拼接证据、系统提示词、chat template
  -> tokenizer 把文本变成 input_ids
  -> vLLM scheduler 排队并组成动态 batch
  -> GPU prefill 计算提示词，decode 逐 token 生成
  -> detokenizer 恢复文本
  -> SSE 将增量答案发回前端
```

这里有三个容易混淆的边界。检索结果是否正确是 RAG 召回和权限问题；模型是否依据证据作答是生成和提示词问题；GPU 利用率、KV cache 和队列则是服务问题。GPU 变快不会修复把一号店库存泄露给三号店的 ACL 缺陷。

权限应在检索前把租户、门店和角色条件传给检索层，或在检索过程中逐条过滤；证据拼装完成、送入模型前还要做一次权限复核。这样生成模型收到的上下文本身就是允许该用户看到的内容，而不是先把跨店结果合并后才尝试补救。

### 1.2 硬件知识会改变后端判断

在 CPU 服务中，线程池和内存较直观；LLM 请求还同时占用权重、激活、KV cache、workspace 和调度元数据，且资源随 token 动态增长。

因此“显卡还有 20% 空闲”不等于还能接 20% 请求：KV 碎片或空间不足仍会阻塞调度；计算利用率高也不代表体验好，排队可能已把 TTFT 推到秒级。

### 1.3 vLLM 在链路中的位置

vLLM 是模型推理和服务引擎，负责把 Hugging Face 模型加载到设备、调度请求、管理 KV cache，并提供 OpenAI 兼容接口。它不是模型本身，也不是检索数据库。你的 Java 或 Python 网关仍应负责租户、超时、重试、审计和业务错误。

PagedAttention 把逻辑 token 序列映射到不要求连续的物理 block，减少动态序列的碎片和复制；实际吞吐仍依赖模型、硬件、版本和工作负载。[PagedAttention 论文（vLLM）](https://arxiv.org/abs/2309.06180)

### 1.4 PyTorch 到底做什么

PyTorch 是张量计算、自动求导和模型组装框架。部署时它定义参数和中间张量，执行 `forward` 算子，把张量移到 CPU/CUDA，并调用 cuBLAS、cuDNN 或自定义 kernel。`config.json` 描述结构，PyTorch 模块和权重把结构变成计算。

训练时 autograd 保存反向所需结果并计算梯度；推理用 `torch.inference_mode()` 减少开销。`transformers` 组装 Qwen 的配置、权重和模型类，`generate()` 是通用生成循环。vLLM 重做调度、KV block、批处理和执行路径，同时使用 PyTorch 张量生态及融合 kernel。

下面的极简例子只演示“张量在 GPU 上做矩阵乘法”，不等于一个完整 LLM。完整模型还要经过多层 attention、MLP、位置编码、采样和 KV cache。

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"  # 选择设备
x = torch.randn(2, 4, device=device, dtype=torch.float32)  # 输入张量
w = torch.randn(4, 3, device=device, dtype=torch.float32)  # 权重张量
with torch.inference_mode():                              # 推理模式，不保存梯度
    y = x @ w                                             # 调度矩阵乘法 kernel
print(y.shape, y.device, y.dtype)
```

`device` 决定张量在主机还是显存，`dtype` 影响精度和 kernel，`@` 调度矩阵乘法。这里用 float32 以适配更多环境；真实 BF16 要确认 GPU、PyTorch 和 kernel 支持。不同设备的张量必须显式搬运。

`torch.inference_mode()` 不会自动调用 `model.eval()`。完整模型部署前仍应执行 `model.eval()`，让 dropout 等训练态行为关闭；vLLM 的模型 runner 会按自身路径处理这一点。看到 `torch.cuda.is_available()` 为真，只能证明基础 CUDA 路径可用，还要验证目标模型 forward、tokenizer、chat template 和 vLLM 服务。

PyTorch 的设备与张量语义可用官方文档核对；CUDA 编程模型对线程块和 warp 的说明也适合结合下一节阅读。[PyTorch CUDA 语义](https://pytorch.org/docs/stable/notes/cuda.html)；[CUDA Programming Guide：Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)

## 2. GPU 基础：SM、warp、Tensor Core 和数据搬运

### 2.1 GPU 不是一颗“大 CPU”

GPU 由许多 Streaming Multiprocessor（SM）组成，SM 执行线程块并管理寄存器/共享内存；kernel 被分派到 SM。线程以 warp 为执行组，NVIDIA 中通常一个 warp 有 32 个线程并执行同一指令。

warp 内分支会产生分歧，矩阵维度与 tile 不匹配也会浪费边界线程。因此 batch 变大未必有效：形状更饱满提高吞吐，却也推高排队和显存压力。

### 2.2 Tensor Core 与普通 CUDA Core

Transformer 中最重的操作通常是矩阵乘法，例如线性层的 `X @ W`。现代 NVIDIA GPU 的 Tensor Core 针对 FP16、BF16、部分 FP8/INT8 等低精度矩阵运算提供专用路径；CUDA Core 更像执行通用标量、地址和控制指令的通用单元。

Tensor Core 的理论 FLOPS 需满足 dtype、矩阵形状、kernel 和架构条件；端到端 tok/s 还受权重/KV 带宽、同步、采样和网络影响。

### 2.3 显存容量、带宽与 PCIe 是三件事

显存容量回答“能否放下”，显存带宽回答“读写多快”，PCIe 连接主机内存与显存。加载时权重通常经磁盘、主机内存和 PCIe 到显存；offload 也走这条路径。

decode 阶段每生成一个 token，都要访问多层权重和历史 KV。此时计算量相对小而内存访问占比高，显存带宽常比 Tensor Core 峰值更关键。PCIe 带宽远低于片上和显存路径，所以“把更多 KV 放到 CPU”一般是容量换延迟，适合保证可用性，不应默认当成性能优化。

### 2.4 用一个简单模型判断瓶颈

可以按 TTFT/TPOT、等待数、KV 使用率和启动显存判断瓶颈；不要从单个 `nvidia-smi` 时间点下结论，应保存请求时间线、vLLM 指标、GPU 采样和应用队列。

## 3. CUDA 版本地图：Driver、Runtime、Toolkit 各管什么

### 3.1 四个“CUDA 版本”分别是什么

NVIDIA Driver 是主机内核和用户态驱动，负责与 GPU 交互；CUDA Runtime 是应用运行时依赖，通常由 PyTorch 或容器带入；CUDA Toolkit 是开发工具和头文件、库的集合；`nvcc` 是 Toolkit 中的 CUDA 编译器驱动。它们不是同一个版本号。

`nvidia-smi` 顶部的 “CUDA Version” 通常表示该驱动声明支持的最高 CUDA 兼容级别，不等于当前 Python 环境实际链接的 runtime。`torch.version.cuda` 通常反映 PyTorch 构建时对应的 CUDA 版本，也不等于驱动版本。`nvcc --version` 反映 PATH 中 Toolkit 的编译器版本。

最低驱动要按目标 CUDA 和发行版核对，不要把 `nvidia-smi` 一行文字当完整矩阵。[NVIDIA CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/latest/)

### 3.2 先采集事实，再决定安装方式

```bash
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
which nvcc && nvcc --version || true
python - <<'PY'
import torch
print("torch", torch.__version__)
print("torch.version.cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
PY
```

`nvidia-smi` 的字段含义和查询格式以 NVIDIA 的工具文档为准；不同驱动版本可见字段会有差异。[nvidia-smi 官方参考](https://docs.nvidia.com/deploy/nvidia-smi/index.html)

没有 `nvcc` 不自动说明 PyTorch 不能运行，预编译 wheel 往往已带用户态库；有 `nvcc` 也不说明自定义 kernel 一定能编译，还要核对 ABI、编译器和 compute capability。

### 3.3 compute capability 为什么是部署约束

compute capability 是 GPU 架构暴露给 CUDA 编译和运行时的能力编号，影响支持的数据类型、Tensor Core 指令和可用 kernel。某个 wheel 可能能导入，但运行到特定 attention 或 FP8 kernel 才因架构不支持而失败。

安装前固定三类版本变量并审核：

```bash
export PYTHON_VERSION="<已审核版本>"
export TORCH_VERSION="<已审核版本>"
export VLLM_VERSION="<已审核版本>"
export CUDA_WHEEL_INDEX="<与目标平台匹配的官方索引>"
```

这里故意不填“最新版”。把占位符替换为团队锁定组合后，再写入 requirements、容器 digest 和发布记录。[vLLM GPU 安装文档](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)

### 3.4 cuBLAS、cuDNN、NCCL 的分工

cuBLAS 提供 GPU 线性代数和矩阵乘法；cuDNN 面向深度学习常用算子；NCCL 面向多 GPU 集合通信，如 all-reduce、all-gather。vLLM、PyTorch 或其依赖可能调用这些库，排查时要看实际动态库和日志，而不是只看包名。

Triton 有两个概念：Triton language 是写 GPU kernel 的语言/编译器；NVIDIA Triton Inference Server 是提供模型仓库和协议的服务端。vLLM 使用 Triton kernel 不等于部署 Triton Server。

## 4. 模型文件与精度：加载成功还不够

### 4.1 一个模型目录各自负责什么

权重文件保存参数，`safetensors` 是强调安全加载和张量分片的权重格式；tokenizer 文件定义文本与 token id 的映射；`config.json` 描述层数、隐藏维度、注意力头、rope 和上下文等结构；generation 配置影响默认采样；chat template 把多轮消息渲染成模型训练时见过的控制 token 序列。

base 模型是预训练续写能力，instruct 模型经过指令对齐，通常应使用适配消息格式。LoRA 是挂在 base 上的低秩增量，不是完整模型；加载时要确认 base、target modules、dtype、量化方式与运行时支持一致。

### 4.2 Qwen3-4B-Instruct-2507 的示例边界

本文采用 Hugging Face 官方仓库的 Qwen3-4B-Instruct-2507 作为教学模型。部署前应固定 revision 或 commit，下载后校验文件清单和哈希，避免同名目录被不同版本覆盖。

官方 `config.json` 中可以看到 `hidden_size=2560`、`num_hidden_layers=36`、`num_attention_heads=32`、`num_key_value_heads=8`、`head_dim=128` 和 `max_position_embeddings=262144`。`head_dim=128` 是 config 明确给出的模型结构值；`hidden_size / num_attention_heads = 80` 只是另一个除法结果，不能拿它替代 `head_dim`。当前配置的 `sliding_window=null`、`use_sliding_window=false` 表示本文按常规全注意力示例推导。MLA、滑动窗口或混合注意力模型的 KV 组织不同，不能直接套这条公式。[Qwen3-4B-Instruct-2507 官方 config](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/main/config.json)

### 4.3 FP16、BF16、FP8、INT4 的取舍

FP16 和 BF16 都是 16-bit 浮点。FP16 的指数范围较小，BF16 保留更大的指数范围，训练和推理中通常更不容易溢出；最终选择仍取决于硬件、kernel 和模型验证。FP8 是 8-bit 浮点，依赖量化校准、缩放因子和硬件/后端支持；“显存减半”不自动等于质量和速度都提升。

INT4 常见于权重 4-bit、激活保持 16-bit 的 W4A16。它主要减少权重占用，但解量化、group scale 和 kernel 可能改变吞吐。量化还可能应用到 KV cache，边界和权重不同。生产选择应同时看离线任务集质量、长上下文表现、首 token 和 TPOT，而不是只看模型目录后缀。

一个粗略的权重下界是：参数总量 × 每参数字节数，再加 scale、量化元数据、临时空间和显存碎片。参数总量已经包含 embedding，不能再次把 embedding 加一遍。4B 参数 BF16 约为 8 GB 十进制（约 7.45 GiB）级别；这只是下界，不是启动后必需显存。24GiB 卡可能适合教学部署，但上下文和并发预算仍必须收紧。

### 4.4 加载前的可重复检查

```bash
MODEL_DIR="/models/Qwen3-4B-Instruct-2507"
test -f "$MODEL_DIR/config.json"
test -f "$MODEL_DIR/tokenizer_config.json"
find "$MODEL_DIR" -maxdepth 1 -name '*.safetensors' -print
python - <<'PY'
from transformers import AutoTokenizer, AutoConfig
p = "/models/Qwen3-4B-Instruct-2507"
c = AutoConfig.from_pretrained(p, trust_remote_code=False)
t = AutoTokenizer.from_pretrained(p, trust_remote_code=False)
print(c.num_hidden_layers, c.num_attention_heads,
      c.num_key_value_heads, getattr(c, "head_dim", None))
print("chat_template", bool(getattr(t, "chat_template", None)))
PY
```

如果模型要求自定义代码，必须审核 `trust_remote_code`，不要为绕过错误盲目打开。chat template 不匹配常表现为回答风格异常或工具调用格式错。

## 5. Prefill、Decode 和 KV Cache：显存预算的核心

### 5.1 两阶段不是两个服务

Prefill 把整段输入 prompt 一次性送入 Transformer，建立每层的 key/value；它的工作量随输入 token 增长，矩阵计算更饱满。Decode 每次生成一个或少量 token，并读取此前 token 的 KV；它反复进行，通常更受内存带宽和调度影响。

首 token 延迟 TTFT 包含排队、prefill、采样和首个网络事件；每输出 token 的 TPOT 或 ITL 主要描述 decode 间隔。把两者混成一个平均 latency，会掩盖“输入很长首 token 慢”和“输出很慢”这两种完全不同的故障。

### 5.2 GQA 下的 KV 公式

对每个 token、每层，KV cache 的字节数近似为：

```text
bytes_per_token = 2(K 和 V) × num_hidden_layers × num_key_value_heads × head_dim × bytes_per_element
```

对官方 config 的 Qwen3-4B-Instruct-2507，BF16 每元素 2 bytes：

```text
2 × 36 × 8 × 128 × 2 = 147,456 bytes ≈ 144 KiB/token
8192 tokens × 147,456 bytes ≈ 1.125 GiB
```

这是 GQA 的关键价值：query 有 32 个头，但 KV 只有 8 个头；如果错误地用 32 计算，会把 KV 预算放大四倍。8 条完整 8K 序列的 KV 约 9GiB 级别，尚未计入权重、激活、workspace 和运行时保留。

若真把上下文推到 262144 token，单序列 KV 就可能达到约 36GiB 级别，24GiB 卡不能仅靠“调高 max model len”解决。这个估算针对本文的常规全注意力 GQA 示例；遇到 MLA、滑窗或混合层，先读 config 和实现再建模。

### 5.3 PagedAttention、连续批处理和 chunked prefill

PagedAttention 把每条序列的逻辑 KV 映射到固定大小 block。请求结束时 block 可回收，新请求不需要等待一整块连续大内存。实现层面还要处理 block table、引用计数和跨请求前缀共享，不能把论文中的“近零浪费”理解成零开销。

Continuous batching 让 scheduler 在每个迭代步把刚进入的 prefill 请求和已有 decode 请求组合起来，完成的序列随时退出。它比“等一整个静态 batch”更适合长度不齐的在线流量，但调度策略会影响 TTFT 与 TPOT 的公平性。

Chunked prefill 把长 prompt 切块，避免占住 GPU；代价是长请求可能变慢，要测试目标版本的调度预算。

### 5.4 FlashAttention、CUDA Graph、Prefix Cache 与投机解码

FlashAttention 通过 tile、片上内存和融合计算减少 attention 的中间矩阵读写，节省 IO 与显存；它不会改变 KV 的信息量，也不是“上下文无限”。原论文重点是 IO-aware 的 attention 计算，实际可用 kernel 仍取决于后端与硬件。[FlashAttention 论文](https://arxiv.org/abs/2205.14135)

CUDA Graph 把形状和执行序列固定后捕获 kernel 调用，减少 CPU launch 开销。动态 batch、不同序列长度或不支持的算子可能回退到 eager；看到回退日志要结合真实形状判断收益。[vLLM CUDA Graphs 设计说明](https://docs.vllm.ai/en/stable/design/cuda_graphs/)

Prefix cache 将精确相同 token 前缀（例如系统提示词和固定门店规则）的 KV block 复用到后续请求；它不是把最终 RAG 答案做结果缓存。命中还要求模型 revision、adapter/LoRA 配置、tokenizer/template 等影响计算的条件一致。vLLM 的 Automatic Prefix Caching 不会自动理解你的 tenant ACL，应用应利用目标版本支持的隔离配置或 cache salt，把租户、门店、权限版本纳入隔离边界，绝不能把一家门店私有证据的前缀跨租户复用。[vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/)

投机解码用较小 draft model 一次提出多个 token，再由 target model 并行验证。接受率高且 draft 成本低时，decode 循环可以减少；若问题分布、模型语言或采样设置导致接受率低，额外验证反而增加延迟。vLLM 的支持项和参数随版本变化，应以对应版本文档为准。[vLLM 优化与调优](https://docs.vllm.ai/en/stable/configuration/optimization/)

## 6. 单卡 vLLM：从自检到 SSE 验证

### 6.1 安装与目录原则

建议使用干净的 Linux NVIDIA 主机或固定 digest 的 CUDA 基础镜像。模型目录用只读挂载，日志和临时目录分开，服务账号不拥有宿主机无关路径的写权限。下面的变量是审核入口：

```bash
export IMAGE="vllm/vllm-openai:<已审核tag>@sha256:<已审核digest>"
export MODEL="/models/Qwen3-4B-Instruct-2507"
export MODEL_REVISION="<已审核的模型 revision>"
export PORT="8000"
export GPU_MEMORY_UTILIZATION="0.82"
export MAX_MODEL_LEN="8192"
export MAX_NUM_SEQS="8"
```

`GPU_MEMORY_UTILIZATION` 和并发不是越大越好。先为权重、CUDA workspace、通信和碎片保留余量，再用压测增加；`MAX_MODEL_LEN` 是保护上限，不是承诺每个请求都能低延迟完成。

若采用 Python venv，先选择团队支持的 Python 版本，再让 vLLM 安装路径解析配套 PyTorch，不要独立随意安装一个 torch：

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
export VLLM_VERSION="<已审核版本>"
python -m pip install "vllm==${VLLM_VERSION:?请先设置经过审核版本}"
python -m pip check
python -m pip freeze > requirements.lock.txt
```

这是与容器二选一的教学路径，命令没有在本文写作环境实测；venv 路径要保存 Python、pip、wheel 和 freeze 证据。

### 6.2 启动模板

```bash
docker run --rm --name qwen3-vllm \
  --gpus 'device=0' \
  --shm-size=8g \
  -p "127.0.0.1:${PORT}:8000" \
  -v "${MODEL}:/models/model:ro" \
  --entrypoint vllm \
  "${IMAGE}" serve /models/model \
    --served-model-name qwen3-4b-instruct-2507 \
    --dtype bfloat16 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --host 0.0.0.0 --port 8000
```

这里显式使用 `--entrypoint vllm`，避免镜像已有 entrypoint 时重复追加。`-p 127.0.0.1:${PORT}:8000` 只绑定本机，外部流量经网关。对于本地只读目录，`--revision` 不会重新验证已下载文件；应在下载阶段固定 revision/commit 并校验哈希。实际 CLI 以审核版本 `vllm serve --help` 和官方文档为准，网关负责鉴权、限流和超时。[vLLM serve CLI](https://docs.vllm.ai/en/stable/cli/serve/)

`--gpus` 依赖 NVIDIA Container Toolkit 将宿主机的 GPU 设备和驱动接口透传给容器。宿主机不要求安装与容器相同版本的 CUDA Toolkit；需要核对的是驱动、容器用户态 CUDA 和 vLLM/PyTorch 的兼容关系。只在已经配置并审核过 NVIDIA Container Toolkit 的 Linux NVIDIA 主机执行以下预检：

```bash
nvidia-smi
docker info
export CUDA_TEST_IMAGE="nvidia/cuda:<已审核tag>"
docker run --rm --gpus 'device=0' "$CUDA_TEST_IMAGE" nvidia-smi
```

最后一条只是验证容器能看到 GPU，不代表模型服务已通过。Toolkit 的安装方式与发行版相关，应遵循 NVIDIA 官方指南并纳入主机基线。[NVIDIA Container Toolkit 安装指南](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### 6.3 先自检，再用 curl 验证 SSE

先验证 GPU 可见性，再验证模型结构和服务启动日志，最后做协议和业务链路。健康检查应包含模型名、版本、最大上下文和实际可用任务；HTTP 200 只能说明路由响应，不能替代一次真实生成。

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"qwen3-4b-instruct-2507",
    "messages":[
      {"role":"system","content":"只依据给定门店证据回答，缺证据就说明未知。"},
      {"role":"user","content":"三号店今天午市牛肉面库存为0，有哪些同价替代菜？"}
    ],
    "temperature":0.2,
    "max_tokens":128,
    "stream":true
  }'
```

SSE 客户端应处理 `data:` 增量、`[DONE]`、断连和超时。网关重试只适用于可判断安全的读请求；生成请求断连后是否重试要考虑重复计费、用户体验和下游副作用。把 request id、model revision、输入/输出 token、排队时间和错误码写入 trace，才有可能把“前端转圈”定位到具体阶段。

### 6.4 参数的第一版预算

| 参数 | 语义 | 教学阶段关注点 |
| --- | --- | --- |
| `max-model-len` | 单请求输入加输出的 token 上限，输入包含 chat template | 先设 4096 或 8192，再按真实长度扩展 |
| `max-num-seqs` | 每个调度轮次可运行的 sequence 数，不是用户数，也不是稳定并发承诺 | 控制动态 batch 和 KV 竞争 |
| `max-num-batched-tokens` | 每个调度轮次新处理的 token 预算，包含 prefill 分块，不是所有序列的完整历史 | 平衡长 prompt 与 decode |
| `gpu-memory-utilization` | vLLM 为该实例规划的显存比例，不是 GPU 计算利用率百分比 | 留出 workspace、碎片和系统余量 |
| `kv-cache-dtype` | KV cache 的存储 dtype，可与权重 dtype 不同但需版本/硬件支持 | 先用默认或 BF16 验证质量 |
| `tensor-parallel-size` | 将模型层内张量切到多少张 GPU | 必须结合拓扑和 NCCL 通信测量 |
| `enforce-eager` | 强制 eager 执行，跳过 CUDA Graph 等捕获路径 | 先排障或比较回退，再评估性能 |

这些参数的准确默认值随 vLLM 版本变化，表格解释语义，不承诺默认配置。命令行和对应版本文档是最终依据。

并发预算可用一个保守模型表达：

```text
可用 KV GiB = 显存总 GiB × 利用率 - 权重 - workspace - 运行时余量
最大活跃 token ≈ 可用 KV GiB / KV GiB per token
```

这只是上界。实际还受每个请求的 prompt、生成上限、block 粒度和调度公平性影响。先用 8K max length、较小 `max_num_seqs` 跑正确性，再逐步扫描长度和并发。把 `max_tokens` 限在业务需要的范围，餐品推荐通常不需要让模型自由生成几千 token。

可以用一轮调度做直观预算：假设 `max-num-batched-tokens=4096`，本轮有 8 条正在 decode 的序列，每条只新增约 1 token，那么剩余约 4088 个 token 才能给新请求的 prefill 块；这不是“8 条请求各自有 4096”，也不是完整历史 KV 被重新计算。若长 RAG prompt 需要 6000 token，它会被拆成后续轮次或等待，具体行为以版本 scheduler 为准。压测时应同时记录每轮新 token、活跃 sequence 和等待数。

## 7. 从单卡走向生产：并行、容器、Kubernetes 和压测

### 7.1 TP、PP、DP、EP 的通信代价

Tensor Parallel（TP）把同一层的张量切到多 GPU，层内需要 all-reduce/all-gather；卡间互联和拓扑直接影响收益。Pipeline Parallel（PP）按层切分，微批次在 stage 之间传递，存在 pipeline bubble，更适合模型放不下但能接受更高调度复杂度的场景。

Data Parallel（DP）是多个完整副本接不同请求，扩容简单但每卡都放完整权重；Expert Parallel（EP）分布 MoE 专家，routing 会引入 all-to-all。先按容量、流量、互联和容错选择，不要把多卡当单一优化。

CPU offload 可以把部分权重或 KV 放到主机内存，缓解容量不足；PCIe 往返会增加延迟，且主机内存带宽和 NUMA 布局也会成为约束。它更像可用性兜底。量化、上下文上限和请求排队通常应先于 offload 进入方案评审。

### 7.2 Docker 和 Kubernetes 的边界

容器要固定镜像 digest、模型 revision 和 Python 依赖，显式声明 GPU、共享内存、优雅终止和日志。Kubernetes 至少定义 startup、readiness、liveness 三类探针：

```text
startup：模型是否完成加载和 warmup，失败时延长启动窗口
readiness：是否接收新流量，滚动发布期间先摘除旧实例
liveness：进程是否卡死，避免把“正在排队”误杀
```

readiness 不应只探测 TCP 端口，应检查已加载目标 revision，并可执行极小的受控生成。探针请求要有独立 token 上限和频率。

灰度按 revision、镜像 digest 和硬件池分流，先用少量租户和固定问题集比较错误率、TTFT、P95、显存、质量和成本，再扩大；回滚应恢复可复现的镜像、配置和模型三元组。

### 7.3 把 TTFT、TPOT、队列和吞吐分开

建议把一次请求拆成（输出 token 数为 N，且 N>1）：

```text
TTFT = 请求发送到首个输出 token 事件的时间
TPOT = (最后一个 token 时间 - 首个 token 时间) / (N - 1)
E2E  = 最后一个 token 时间 - 请求发送时间
       （若协议有完成标记，可另计完成标记后的尾部时间）
```

`N=1` 时 TPOT 不适用。TTFT（Time To First Token）衡量用户多久看到第一段文字；TPOT（Time Per Output Token）是平均输出 token 间隔；ITL（Inter-Token Latency）应由相邻 token 的时间戳计算。SSE event 不等于一个 token，一个 event 可能包含多个 token 或只有 role/finish 字段，不能按事件数伪造 ITL。队列时间仍应拆出网关 queue 和 vLLM waiting，便于定位容量边界。vLLM 指标名和暴露方式以目标版本文档为准。[vLLM Metrics](https://docs.vllm.ai/en/stable/usage/metrics/)

SLI 是实际测量的指标，如成功请求率、TTFT P95、TPOT P95、有效 output tok/s；SLO 是团队承诺的目标窗口，例如“正常长度请求 TTFT P95 小于某阈值”；SLA 是对外合同和违约责任。稳定并发表示在固定输入/输出分布下，队列、显存和错误率经过一段观察窗口仍不发散；它不同于瞬时 `max-num-seqs`。有效吞吐应排除错误和取消请求，并注明是 output tok/s、request/s 还是总 tok/s。

压测不能只发同一条短 prompt。短问答、长 RAG、不同输出上限、突发并发和稳定到达率都要覆盖，同时冻结 revision、采样参数、输入集和 warmup。结果记录 P50/P95/P99，并拆开排队与生成阶段，保留并发、输入/输出 token、TTFT、TPOT、错误率、峰值显存和有效吞吐。

并发增加而吞吐上升、P95 急剧上升，说明批处理收益已被排队吞噬；GPU 利用率低而队列高，查 CPU tokenization、网络、锁和 scheduler；GPU 满且 TPOT 恶化，降并发或输出上限并查带宽/KV。不要用一次最佳 tok/s 替代容量曲线。

## 8. 用证据排障，再组织面试表达

### 8.1 按症状建立证据链

| 症状 | 先收集 | 常见处理方向 |
| --- | --- | --- |
| `CUDA driver version is insufficient` | driver、torch CUDA、镜像标签、实际库 | 选择兼容组合或升级经审核的驱动；不要只改 `CUDA_VISIBLE_DEVICES` |
| `no kernel image` / 架构不支持 | compute capability、wheel 编译目标、kernel 日志 | 换匹配 wheel/镜像，确认硬件支持；必要时降级功能 |
| 启动加载 OOM | 权重 dtype、分片、临时峰值、workspace | 降精度/量化、减少加载并行、留余量；核对权重是否重复加载 |
| 运行一段时间 OOM | 活跃 token、KV 使用率、长度上限、block | 降低 max length/并发，限制输出，启用合适 cache 策略 |
| TTFT P95 飙升 | 网关队列、vLLM waiting、输入长度、chunked prefill | 限制长请求、分离队列、调整调度预算并重新压测 |
| 输出乱码或工具调用失败 | tokenizer、chat template、model revision、请求 JSON | 使用官方模板，固定 revision，做端到端协议测试 |
| 多卡吞吐下降 | NCCL 日志、拓扑、TP/PP 配置、跨 NUMA/节点路径 | 检查互联与通信比例，比较单卡副本和 TP |

排障按时间戳对齐网关 trace、vLLM metrics、CUDA/NCCL 日志和 GPU 采样，先证明请求在哪层等待再改参数。GPU 利用率与慢请求同时出现只是相关，数据库慢也不等于 GPU 是根因。

### 8.2 按证据推进学习

学习顺序可以沿着请求链推进：先熟悉模型目录和一次 SSE，能解释 tokenizer、config、chat template 与权重 dtype；再补硬件和版本，读懂 `nvidia-smi`、`torch.version.cuda`、`nvcc` 与 compute capability 的差异。接着手算 GQA KV，改变长度和并发，观察 TTFT、TPOT、waiting 及 KV 使用率；再读 PagedAttention 和相关优化文档，理解 batching、prefix cache、FlashAttention、CUDA Graph 的收益与回退，最后进入 TP/PP/DP/EP、灰度和容量模型。每一步都保存命令、版本、revision、配置、输入集、指标和结论，面试时才分得清实际实验与文档峰值。

### 8.3 30 秒面试表达

“我把 LLM 部署拆成兼容性、模型文件、显存、调度和治理五层。以 Qwen3-4B 为例，先锁定 driver、PyTorch/vLLM、镜像和模型 revision，按权重、GQA KV、workspace 与余量预算显存。在线请求分 prefill/decode，用 vLLM 做 paged KV 和 continuous batching；压测看 TTFT、TPOT、队列、P95 与有效吞吐，故障用 trace、vLLM 指标和 GPU/CUDA 日志对齐定位。”

### 8.4 两分钟面试表达

“餐饮 RAG 先做租户/门店过滤，tokenizer 生成输入，vLLM 组成动态批次。Prefill 决定 TTFT，decode 逐 token 读权重和 KV，决定 TPOT。Qwen3-4B 是 36 层、8 个 KV 头、128 head_dim，BF16 KV 按 2×36×8×128×2 字节/token 预算，所以 context 和并发必须一起看。

“兼容性上区分 driver、runtime、Toolkit/nvcc、PyTorch CUDA 和 `nvidia-smi`，再核对 compute capability；cuBLAS、cuDNN、NCCL 和 Triton language/server 各有边界。部署锁定 digest/revision，先自检再压测；P95 变差看 queue、KV、带宽和 NCCL，多卡按容量与拓扑选 TP/PP/DP/EP，配合探针灰度。”

部署工作最后要落到可复现的证据：版本、硬件、数据路径、动态 token 资源和 SLA 都能对上。先填好版本变量，跑通单卡自检，再建立 TTFT、TPOT 和 P95 的容量曲线。
