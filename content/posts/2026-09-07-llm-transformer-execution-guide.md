---
title: "一条餐饮请求怎样穿过 Transformer"
date: 2026-09-07
description: "沿着餐饮助手的一次请求示例，拆开 tokenizer、注意力、RoPE、KV Cache、训练和 Agent 边界，配合维度公式与手算例理解大模型如何生成下一个 token。"
tags: [AI, Transformer, 大模型, 深度学习, 推理]
draft: false
---

大模型回答一句话，表面上是“输入文字，输出文字”，内部却是一条有严格数据依赖的计算链：消息先套入聊天模板，再被 tokenizer 切成 token id；id 查到 embedding 后，经过多层 Transformer decoder；最后一层 hidden state 经 lm head 变成整个词表的 logits，采样器选出下一个 token，循环往复直到停止。

用一个餐饮助手来跟踪这条链：顾客说“今晚两位，想吃清淡的川菜，预算每人 80 元”，系统还知道顾客不吃花生，只能推荐当前门店菜单中的菜。这样既能看清模型怎样处理上下文，也能看清参数、检索、库存工具和业务校验各自负责哪一段。

## 请求先变成 token 和向量

### 聊天模板先确定消息边界

服务端收到请求后，先验证用户、门店和会话权限，再把系统约束、历史对话和新消息组织成模型认识的格式。概念上的模板可能是：

```text
<|system|>
你是门店餐饮助手，只推荐当前菜单中的菜……<|end|>
<|user|>
我不吃花生。<|end|>
<|assistant|>
好的，我会避开含花生的菜。<|end|>
<|user|>
今晚两位，想吃清淡的川菜，预算每人80元。<|end|>
<|assistant|>
```

特殊 token 的名称、顺序和是否需要结尾标记由具体模型的 tokenizer 配置决定。一个模型的模板不能直接套给另一个模型。模板的作用是标出“谁说了什么”和“回答从哪里开始”，它不是给模型增加事实。

### tokenizer、embedding 和 hidden state

tokenizer 按词、子词、字符、字节或混合规则切分文本。为便于演示，假定词表中有 `<|user|>=11`、`<|assistant|>=12`、`<|end|>=13`，并把“今晚”“两位”“想吃”“清淡”“川菜”“预算”“每人”“80”“元”分别暂记为 201 到 209。局部输入可以写成：

`[11, 201, 202, 203, 204, 205, 206, 207, 208, 209, 13, 12]`

这只是数值示意，真实中文可能按字符、子词或字节拆开，“川菜”不一定是一个 token，“80”也可能分成多个片段。token 数会影响上下文长度、计算量和计费。训练和推理必须使用匹配的词表、合并规则和特殊 token 配置；一旦 tokenizer 不匹配，后续 embedding 查表就已经错位。

设 embedding 矩阵 `E` 的形状为 `[Vocab, D]`。输入 id 序列选择 E 中对应的行，得到 `X`，形状是 `[T,D]`，加上 batch 后为 `[B,T,D]`。例如教学用的某一行可以写成 `[0.30,-0.10,0.70,0.20]`。

这个向量不是人工写好的“清淡属性表”，而是训练得到的初始表示，之后会被每层网络按上下文持续变换。每层的 hidden state 是上下文相关的连续表示；logit 则是词表中每个候选 token 的未归一化分数；softmax 后才是概率。

### 不要混淆消息、表示与输出分数

消息是带角色的业务输入，token 是 tokenizer 产出的片段，token id 是词表索引，embedding 是查表得到的初始向量，hidden state 是层间传递的上下文表示，logit 是候选 token 的分数，概率是归一化后的分布。

模型不是执行 `menu[清淡][川菜]` 的菜单查表器：attention 会在连续表示中动态聚合上下文，MLP 会做非线性组合；当前库存、价格和过敏原仍应由检索或工具提供并由业务层校验。

## 先认清模型家族和张量维度

### 三种模型家族各自处理什么

Llama、GPT 一类模型通常是 decoder-only。输入和待生成内容在一条序列上，因果 mask 规定位置 `t` 只能读取位置 `s<=t`，训练目标是预测下一个 token，推理时从 assistant 起始位置逐个生成。

BERT 一类 encoder-only 模型对输入使用双向注意力，位置可以读取前后文，常见目标是掩码语言模型、分类、匹配和 embedding，通常不直接按自回归方式生成长回答。

原始 Transformer 则是 encoder-decoder：encoder 双向编码源序列，decoder 对目标序列做因果 self-attention，并通过 cross-attention 读取 encoder 输出。翻译是典型任务。原始论文的 post-norm、正弦位置编码与现代 Llama 类 pre-norm 实现并不相同，不能把所有 Transformer 都当成同一种架构。

### 一组贯穿全文的维度

为避免把 value 和 vocabulary 都写成 V，下面统一使用这组符号：

| 符号 | 含义 | 示例 |
| --- | --- | ---: |
| `B` | batch size | 2 |
| `T` | 当前序列长度 | 128 |
| `D` | hidden size | 4096 |
| `Hq` | query 头数 | 32 |
| `Hkv` | key/value 头数 | 8 |
| `dh` | 每头维度，通常 `D/Hq` | 128 |
| `F` | MLP 中间维度 | 14336 |
| `Vocab` | 词表大小 | 32000 |
| `K` | 历史缓存长度 | 512 |
| `L` | Transformer 层数 | 32 |

隐藏状态是 `[B,T,D]`。拆头后 Q 是 `[B,Hq,T,dh]`；GQA 中 K、V 是 `[B,Hkv,T,dh]`；多头合并后回到 `[B,T,D]`；lm head 输出 `[B,T,Vocab]`。若 `Hq=32,Hkv=8`，每个 KV 头服务四个 query 头。GQA 减少缓存的 K/V 存储和读取量，但不是把四个 query 头合成一个头。

### 位置如何进入计算

绝对位置编码可以直接加到输入表示。Llama 类实现常在 Q、K 上使用 RoPE：把每两个维度看作二维平面，按位置角度旋转。对位置 `p` 和维度对 `(2i,2i+1)`，角度可写作 `theta(p,i)=p*inv_freq(i)`，旋转公式为：

```text
q'2i   = q2i*cos(theta) - q2i+1*sin(theta)
q'2i+1 = q2i*sin(theta) + q2i+1*cos(theta)
```

K 使用其所在位置对应的旋转。点积因此携带相对位置信息。缓存 K 时，要保留已经按原位置旋转后的结果，或严格遵循实现约定恢复。位置 id 是时间位置，和 token id 是两套不同的编号。

## 一层 Llama 类 decoder 如何前向

### pre-norm、QKV 与残差

现代 Llama 类层可以抽象成：

```text
x = x + Attention(RMSNorm(x))
x = x + MLP(RMSNorm(x))
```

同一层中不同 token 的矩阵运算可以并行，相邻层必须等待上一层的 hidden state；自回归生成的下一个 token 还要等待当前 token 被选出。GPU 并行会提高吞吐，但不会消除这些数据依赖。

RMSNorm 对每个 token 的 D 个分量计算均方根：

`rms(x)=sqrt(mean(x_i^2)+eps)`，`RMSNorm(x)_i=weight_i*x_i/rms(x)`。

它通常没有 LayerNorm 的均值减法和 bias，作用是控制激活尺度、帮助深层计算稳定。归一化不会把向量变成某个业务标签。

归一化结果 `h` 经过可学习线性投影：`Q=hWq+bq`、`K=hWk+bk`、`V=hWv+bv`。不少 Llama 实现没有 bias。Q reshape 为 `[B,Hq,T,dh]`，K/V reshape 为 `[B,Hkv,T,dh]`。MHA 中两类头数相同，MQA 的 KV 头数为 1，GQA 则是多个 query 头共享较少的 KV 头。

### Q、K、V 的业务直觉

当前位置正在准备生成菜品建议，可以把 Q 理解为“我现在要读取什么”，K 理解为每个历史位置的“我适合被什么需求匹配”，V 则是匹配后真正要汇总的内容。它们来自同一个 hidden state 的三组不同投影，不是人工定义的问题、标签或菜单记录。同一个词在不同上下文、层和头中都可能有不同的 Q、K、V，也不能武断地说某个固定头永远负责预算或语法。

QK 相似度决定读取比例，乘 V 才得到读出的内容。多个头在不同投影子空间中并行读取，拼接后经 output projection 融合，功能来自联合训练。

### SwiGLU 与两次残差

attention 输出经过 output projection 回到 D 维，与输入逐元素相加。残差 `y=x+f(x)` 的反向导数包含恒等路径，有助于梯度穿过深层网络，但不保证梯度绝不消失或爆炸。

第二个分支先 RMSNorm，再做常见的 SwiGLU：

`MLP(h)=down_proj(SiLU(gate_proj(h))*up_proj(h))`。

`gate_proj`、`up_proj` 把 D 映射到 F，逐元素相乘仍是 `[B,T,F]`；`down_proj` 再映射回 D。`SiLU(x)=x*sigmoid(x)`。门控分支学习哪些中间特征通过。attention 混合不同 token 的信息，MLP 主要在每个 token 内做非线性变换；MLP 结果再与 attention 后状态做残差相加。

### final norm 到 logits

所有层完成后，hidden state 仍是 `[B,T,D]`。final RMSNorm 后，lm head 映射到 `[B,T,Vocab]`。自回归生成通常只取最后一个有效位置的 logits：第 j 项是下一个 token id j 的未归一化偏好。经过温度、top-k、top-p 和停止规则后选出一个 token，再把它追加进序列。

## Attention 公式：用一个小例子算一遍

### scaled dot-product 的形状

单个 query 头的 Q 为 `[Tq,dh]`，K 为 `[Tk,dh]`，V 为 `[Tk,dh]`。分数矩阵为：

`S=QK^T/sqrt(dh)`，形状 `[Tq,Tk]`。

除以 `sqrt(dh)` 是为了抑制维度增大导致的点积方差过大。随后加上 causal mask、padding mask 或其他结构 mask。允许位置使用原分数，被屏蔽位置概念上加负无穷。

### softmax 沿 key 轴

固定 query 位置 `t`，模型要在所有可见 key 位置之间形成分布，因此：

`A[t,s]=exp(S[t,s])/sum_j exp(S[t,j])`。

归一化沿 `Tk` 这一条 key 轴进行；对固定 `t`，有效位置的权重之和为 1。工程实现通常先减去每行最大值以避免指数溢出。然后执行：

`O=AV`。

形状是 `[Tq,Tk] @ [Tk,dh] = [Tq,dh]`。每个 query 得到所有可见 value 的加权向量。它是连续内容聚合，不是从 value 表中选一行。

### 两个 key 的数值演示

设 `Tq=1,Tk=2,dh=2`，当前 query 为 `q=[1,0]`，两个 key 为 `k1=[1,0]`、`k2=[0,1]`，value 为 `v1=[10,0]`、`v2=[0,20]`。

未缩放点积是 `[1,0]`，除以 `sqrt(2)` 后约为 `[0.707,0]`。softmax 约为 `[0.670,0.330]`，所以：

`O=0.670*[10,0]+0.330*[0,20]=[6.70,6.60]`。

如果第二个 key 被 padding mask 屏蔽，权重变为 `[1,0]`，输出就是 `[10,0]`。如果未来位置没有 causal mask，模型可能偷看到训练标签，训练目标就失去意义。这个小例子同时说明了动态内容寻址、key 轴归一化和 mask 的作用。

### causal mask 与 padding mask

causal mask 解决“不能看未来”，padding mask 解决“不能读无效填充”。两者经常叠加，但语义不同。

decode 时 query 长度和 key 长度往往不同：历史长度为 128、新 token 为 1 时，Q 可能是 `[B,Hq,1,dh]`，K 是 `[B,Hkv,129,dh]`，score 最后两维是 `[1,129]`，mask 不是 `[129,129]`。

左 padding、右 padding 和 packed sequence 会进一步改变布局，`attention_mask` 的 0/1 语义要以具体框架实现为准。

## Prefill、decode 和 KV Cache 的时序

### 第一枚输出 token 从哪里来

假设 prompt 有 P 个 token，回答依次为 `y1,y2,y3`。在无 padding、完整因果注意力下：

| 本轮送入 forward | 本轮结束每层缓存长度 | 使用哪个位置 logits | 随后选出 |
| --- | ---: | --- | --- |
| 完整 prompt | P | prompt 最后位置 | `y1` |
| `y1` | P+1 | `y1` 位置 | `y2` |
| `y2` | P+2 | `y2` 位置 | `y3` |

prefill 已经提供 `y1` 的分数，不需要先做一次空 decode。刚选出的 `y1` 还没有经过模型，所以缓存里没有 `y1` 的 K/V；下一轮送入 `y1` 时才计算并追加。若生成 `y3` 后立即遇到 EOS 或业务停止条件，`y3` 可以从未进入缓存。核心关系是“位置 t 的输出预测位置 t+1”：forward 负责 hidden state 和 logits，生成循环负责选择、追加和停止。

### prefill 和 decode 的工作差异

prefill 一次处理系统消息、历史、新问题和检索上下文，能够并行计算整段 prompt。每层都产生 prompt 各位置的 K/V 并写入 cache，最后位置 logits 决定第一个回答 token。长菜单、长历史和长检索上下文会增加首 token 延迟。

decode 通常每轮只输入一个新 token。每层为它计算一个 Q、K、V，把新 K/V 追加到 cache，用新 Q 与全部历史 K 做 attention，再聚合全部历史 V。旧 token 的 Q 不需要重新计算，因为未来位置不会用历史 Q 去构造新的 score。

### 历史 K/V 为什么可以复用

在确定性的因果前向中，未来 token 不会改变过去位置的隐藏状态。第一层的历史输入不变，历史 K/V 就不变；后一层的历史状态也只依赖其自身及更早位置，这个结论因此逐层成立。新位置只需要自己的 Q，却要读取历史 K/V，所以缓存 K/V 有用，历史 Q 通常无需保留。

改动历史 token、权重、位置规则或 mask 会使相关缓存失效。双向 attention 的过去表示会受新增后文影响，不能直接套用这个结论。批量处理多个新增 token 时，第 i 个新 query 只能看到已有历史和截至 i 的新增 key，不能因已有 cache 就省掉这部分因果约束。

### cache 保存什么、不能解决什么

KV cache 通常按层保存历史 K、V，形状可抽象为 `[B,Hkv,K,dh]`。它不是所有中间激活、完整 score 矩阵或 logits 的永久保存，也不保存 Q。cache 只适用于同一个模型、token 前缀、模板和位置规则；共享 prefix cache 还必须满足租户、权限和内容边界，不能把一个用户的私密上下文作为另一个用户的前缀。

设 `B=1,L=32,Hkv=8,K=4096,dh=128`，半精度每元素约 2 bytes。每层 K、V 合计约：

`2*Hkv*K*dh*2 = 16,777,216 bytes ≈ 16 MiB`。

32 层约 512 MiB，还未计入元数据和 batch 增长。若改为同样头数为 32 的 MHA，cache 大约是 GQA 的四倍；MQA 可进一步减少存储，但可能牺牲部分表达能力。cache 让 attention 的历史部分避免重复投影，不能让新 token 不经计算就出现，也不能消除读取全部历史 K/V 的带宽成本。

## 生成、训练和推理成本

### 温度、top-k、top-p 与停止

attention 中 softmax 的候选是历史位置，用来决定读取谁；输出 softmax 的候选是词表 token，用来决定写出什么。贪心可直接对 logits 做 argmax，无需显式求概率。温度缩放要求温度为正；产品界面中的零温度通常由贪心或特殊逻辑处理，不能直接除以零。

logits 经温度缩放后再归一化：温度较低时分布更尖锐，较高时更平坦；它只改变选择分布，不能补充库存事实，也不能保证答案正确。top-k 保留分数最高的 k 个候选，top-p 按概率从高到低累积到阈值，保留动态数量的候选。repetition penalty 可降低重复，但也可能损伤必要的重复。

贪心解码每次选最高概率 token，结果稳定但可能陷入重复；随机采样按处理后的分布选择，能增加多样性。EOS、最大新 token 数、停止字符串和业务状态都可能终止生成。流式输出只是把已经产生的增量更早推送给客户端，并不代表请求已经完成。

### teacher forcing 与 next-token loss

训练样本是 token 序列 `x1,x2,...,xT`，目标是让位置 t 预测 `x(t+1)`。训练时把真实历史 token 直接喂回模型，这叫 teacher forcing；因果 mask 仍阻断未来。框架常接收等长的 `input_ids` 和 `labels`，在模型内部做 shift，数据侧再移一次会造成错位。

padding 或不参与监督的位置应使用 ignore index；对话 SFT 常只对 assistant 输出计算 loss。

单位置交叉熵是 `-log p(correct_token)`。若三个候选的预测概率是 `[0.7,0.2,0.1]`，第一个为正确标签，则 loss 为 `-ln(0.7)≈0.3567`；正确概率只有 0.1 时，loss 为 `-ln(0.1)≈2.3026`。

softmax 与交叉熵合并后，logits 梯度为 `p-one_hot(label)`，此例为 `[-0.3,0.2,0.1]`。梯度经过 lm head 向较早的层反向传播，优化器更新共享参数，并不是只修改这一条问答记录。

### 参数更新与对齐边界

前向保存反向所需的激活，反向按链式法则求出 embedding、投影矩阵、归一化权重和 MLP 参数的梯度，AdamW 根据一阶、二阶统计和权重衰减更新参数。混合精度、梯度累积和梯度裁剪用于显存与稳定性管理。线上推理一般关闭梯度并使用 eval 模式，生成本身不会更新参数。

SFT 用带目标答案的监督样本训练模型模仿答案；DPO 比较 chosen 与 rejected 回答，优化偏好相对概率；RLHF 还包括奖励模型和策略优化等环节。把 temperature 调低属于推理控制，不是对齐训练。

### 计算量要拆开看

令 `Dkv=Hkv*dh`，单层稠密计算量可粗略写成：

| 运算 | 长度 T 的 prefill | 历史 K 的单 token decode |
| --- | --- | --- |
| Q 与输出投影 | `O(T*D²)` | `O(D²)` |
| K、V 投影 | `O(T*D*Dkv)` | `O(D*Dkv)` |
| QK 与 AV | `O(T²*Hq*dh)` | `O(K*Hq*dh)` |
| SwiGLU | `O(T*D*F)` | `O(D*F)` |

全部层还要乘 L，batch 还要考虑 B；词表投影单位置约 `O(D*Vocab)`。从长度 P 的 prompt 生成 N 个 token，有 cache 时 attention 历史项随 `N*P+N²` 增长；每轮重算全序列则会重复累加平方项。所谓“cache 后单步是 O(K)”只描述 attention 随历史增长的部分，不代表整个模型单步只有这一项。

推理显存至少包括权重、KV cache、激活和工作区、框架开销。7B 参数模型用 BF16 存权重，理论载荷约 14×10⁹ bytes，约 13.0 GiB，不能当作完整推理显存；理想 4-bit 权重载荷约 3.26 GiB，还要加量化尺度等元数据。训练还需梯度、优化器状态和反向激活。

### 常见优化的边界

FlashAttention 通过分块和在线 softmax 减少中间矩阵的显存读写，数学目标仍是 attention，也不会自动把二次复杂度变成线性。量化可以降低权重、激活或 KV cache 的位宽，W4A16、W8A8 和 KV 量化是不同配置，误差会受敏感层、异常值和校准方法影响。

低 batch 的 decode 常受权重和 KV 读取带宽影响；prefill 更容易发挥矩阵计算吞吐。连续批处理可在 decode 阶段动态加入和移除请求，但服务端必须分别管理 cache、取消、停止状态、最大上下文和租户隔离。投机解码让小模型先提出候选，大模型批量验证，被接受的 token 可减少大模型逐 token 调用；它要求正确处理 cache 回滚和随机状态，收益取决于候选模型与目标模型的分布接近程度。

## RAG、Agent 和模型前向各自负责什么

### 餐饮助手的真实链路

模型前向只把 token 序列映射为 logits，它不会天然读取数据库，也不会自动知道实时库存和用户权限。一个可靠的餐饮请求通常经历：API 验证用户、门店和会话；应用判断是否需要查菜单、库存和过敏原；检索系统先按租户、文档 ACL 和版本过滤，再做关键词或向量检索；工具返回菜名、价格、过敏原和库存状态；应用把经过授权的证据和约束放入新的模型请求；模型生成推荐或工具参数；应用再次校验参数、幂等键和权限后，才执行下单或改桌等副作用。

这里的顺序很关键：权限和租户边界必须在检索前落实，避免先召回再把不应见到的内容交给模型。模型可以生成“推荐某道菜”，但是否存在于当前菜单、是否有库存、是否含花生，要由事实工具和业务层决定。

### Agent 循环与流式事件

Agent 通常是“理解任务—选择工具—执行工具—观察结果—继续推理或结束”的应用层循环。工具执行不是 Transformer 层中的 attention。模型输出的工具参数可能分多个流式片段到达，不能因为收到 HTTP 200 或半截 JSON 就无条件执行；应在结构化解析、权限校验和幂等控制后产生副作用。

SSE 或 WebSocket 可以把文本增量、工具调用增量、错误和完成事件推给客户端。客户端要区分这些事件，服务端也要在取消、重试和断线重连时维护请求状态。TTFT 可能包含排队、检索、网络和 prefill，不能把它简单等同于模型矩阵计算时间；后续 token 速度还会受到 KV cache、批调度和显存带宽影响。

## 把公式对回源码，再用演示排障

### 如何把公式映射到实现

阅读类 Llama 实现时，可以按职责寻找：模型类组织 embedding、decoder layers、final norm 和 cache；decoder layer 组织两次 pre-norm 与残差；attention 模块负责 qkv、RoPE、cache 更新、attention 和 output projection；MLP 负责 gate、up、down 投影与激活；CausalLM 外壳接 lm head。

概念性伪代码如下，不能直接当作具体框架 API：

```python
def causal_lm(input_ids, position_ids, cache=None):
    x = embed(input_ids)
    for layer in layers:
        h = layer.attn_norm(x)
        q, k, v = project_qkv(h)
        q, k = apply_rope(q, k, position_ids)
        k_all, v_all = append_to_layer_cache(layer, k, v, cache)
        a = softmax((q @ transpose(k_all)) / sqrt(dh) + make_mask()) @ v_all
        x = x + layer.o_proj(merge_heads(a))
        x = x + layer.mlp(layer.mlp_norm(x))
    return lm_head(final_rms_norm(x)), cache
```

对照真实代码时先确认张量布局，再确认 transpose、GQA 的 repeat/index 和 mask 广播，接着确认 cache 在哪一层更新，最后确认返回 logits 是全部位置还是由服务层只取最后位置。

- 可参考 Hugging Face 的 [`modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)、[Cache explanation](https://huggingface.co/docs/transformers/cache_explanation) 和 [KV cache 文档](https://huggingface.co/docs/transformers/kv_cache)。

### 一个可复现的微型演示

配套脚本使用固定随机权重的两层微型模型，演示 RMSNorm、QKV、RoPE、残差、SwiGLU 和 KV cache。它没有真实语言能力，作用是验证张量形状和增量计算时序。

脚本中还用 `Q=[1,0]`、`K=[[1,0],[0,1],[1,1]]`、`V=[[2,0],[0,2],[2,2]]` 演示 key 轴 softmax，得到约 `[0.401112,0.197776,0.401112]` 的权重和约 `[1.604448,1.197776]` 的加权输出。

下载：[Transformer 注意力与 KV 缓存演示脚本](/downloads/transformer-attention-kv-demo.py)

```bash
python3 transformer-attention-kv-demo.py
```

脚本已验证：完整序列与逐 token KV Cache 前向的 logits 最大误差为 0；多个新增 token 的非方形因果 mask 验证通过；改变未来 token 不影响更早位置的输出。这些结果验证微型模型的计算等价性，不代表真实模型的回答质量或性能。

### 从症状回到数据路径

排查首 token 变慢时，先看 prompt 长度、检索结果规模、排队和 prefill；后续 token 变慢，则转向 KV cache 显存、历史长度、批调度和带宽。重复输出要核对采样、重复惩罚、EOS 与上下文追加；看到未来内容要查 label shift、causal mask 和 packed sequence mask；菜单外菜品则回到库存工具、证据版本和最终业务校验。

如果出现跨租户回答，要同时核对会话上下文、RAG ACL、共享 prefix/cache key、模型与模板版本。乱码通常来自 tokenizer、chat template、特殊 token 或 decode 配置不匹配。一个接口返回 HTTP 200 只表示 HTTP 层报告成功，不能代替工具调用、事件流、权限和最终业务结果的验收。

### 延伸来源

- Transformer 的 scaled dot-product attention、多头注意力和 encoder-decoder 结构见 Vaswani 等人的 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)；RoPE 见 [RoFormer](https://arxiv.org/abs/2104.09864)，RMSNorm 见 [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)。

- GQA、SwiGLU、FlashAttention 和 DPO 分别可参照 [GQA](https://arxiv.org/abs/2305.13245)、[GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)、[FlashAttention](https://arxiv.org/abs/2205.14135) 与 [DPO](https://arxiv.org/abs/2305.18290)。

可以先运行脚本，再沿着一枚 token 核对每层的输入、QKV、mask 和缓存长度。真正掌握这条链的标志，是能说清当前位置读取了什么、logits 预测哪个位置，以及哪些状态可以复用。
