---
title: "大模型微调怎么选：SFT、LoRA、DPO 与 PPO"
date: "2026-09-07"
description: "用餐饮客服贯穿解释继续预训练、监督微调、参数高效微调、偏好优化和 RLHF，并把数据、指标、显存、部署与故障诊断连成一条可执行的训练链路。"
tags: [大模型, 微调, SFT, LoRA, DPO, PPO]
---

“微调”经常被当成一个按钮：准备几条问答，跑几个 epoch，loss 降了就上线。真实项目里，最容易出错的反而是边界：训练的是知识还是行为？更新的是全部权重还是一个 adapter？偏好数据中的 chosen 为什么胜过 rejected？线上使用的模板、量化和版本是否与训练一致？

假设餐饮 SaaS 有一个客服模型：用户问“今天晚市两人套餐能否预约，过敏原怎么查”，系统要遵守门店规则，遇到不确定的成分还要转人工。这个场景能把训练目标、参数范围和线上约束放到同一条链上。文中的数据、数字和显存估算都是教学假设，生产结论仍要回到自己的日志、评测集和硬件实测。

## 先把训练目标和参数范围分开

继续预训练（continued pretraining，CPT）仍然做语言建模：给定前文，预测下一个 token。它适合把模型暴露给门店手册、菜品说明、行业术语等领域文本，改变的是“知道什么、如何续写”的分布。它不天然教会模型遵循“先核验门店再回答”的对话协议，也不等于把文档变成可追溯知识库；经常变化的库存、价格和权限仍应由 RAG 或工具实时查询。

监督微调（SFT）给出输入和期望输出，训练模型模仿示范。客服样本可以是：用户询问过敏原，助手先调用菜品成分工具，再引用返回结果；工具失败时说明无法确认并转人工。SFT 的目标通常是回答 token 的交叉熵，数据把行为写成了可学习的规范。偏好对齐则进一步提供同一问题的 chosen 与 rejected，例如“引用已核验成分并说明门店差异”的回答优于“凭常识保证绝对安全”的回答。

这里要分开两个维度：CPT/SFT/DPO/PPO 说的是训练信号和目标，full fine-tuning、freeze、partial、PEFT、LoRA 说的是哪些参数可训练。目标与 trainable scope 正交。可以做全参 SFT，也可以做冻结底座的 LoRA-SFT；也可以用 LoRA 做 DPO。看到“LoRA 微调”时要追问：LoRA 是参数更新方式，SFT 才是监督目标。

参数范围从大到小可这样理解：全参 SFT 更新基座的全部可训练权重；部分参数 SFT 只解冻指定层、embedding、lm_head 或若干模块；冻结策略保持大部分权重不动。PEFT 是减少训练成本的一组方法，既包括增加 LoRA 这类参数，也包括只选择已有参数训练的 BitFit；LoRA 只是其中一种。

QLoRA 是“量化底座 + LoRA 训练”的组合，量化降低底座显存，不代表 adapter 也自动可以随意量化。

| 名称 | 解决的问题 | 数据或更新方式 |
| --- | --- | --- |
| SFT | 学会怎样完成任务 | 输入与示范输出，监督学习 |
| 全参 SFT | 用全部基座权重学习示范 | 更新全部基座参数 |
| 部分参数 SFT | 限制已有权重的更新范围 | 只解冻指定层或模块 |
| SFT + LoRA | 低成本学习示范 | 冻结基座，训练低秩增量 |
| QLoRA | 进一步降低基座存储显存 | 量化基座加 LoRA，可用于 SFT 等目标 |
| DPO | 从成对答案学习偏好 | prompt、chosen、rejected 与冻结参考策略 |
| PPO | 利用当前策略生成结果的奖励更新 | rollout、奖励、优势估计与裁剪更新 |

SFT、DPO、PPO 不是必须依次执行的三个等级。先用提示词和工具建立基线；示范行为不稳定时考虑 SFT，有可信偏好对且主要缺陷是回答选择时考虑 DPO。只有具备可用奖励、独立评测和在线采样训练预算时，才考虑 PPO 等强化学习方法。这里的“在线”指训练过程中重新采样，不等于直接在生产用户请求上试错。

## SFT 先把示范写成训练目标

对一个 token 序列 `x_1,...,x_T`，因果语言模型常见目标是

`L_CE = - sum_t m_t log p_theta(x_t | x_<t) / sum_t m_t`

其中 `m_t` 是 mask。客服对话通常只在 assistant 回复 token 上令 `m_t=1`，system、user 和工具返回正文通常令 `m_t=0`；assistant 的结束标记保留监督，这样模型学习“怎样回答”，而不是把用户问题当成答案复述。若把整段文本都算 loss，训练可能仍能下降，但会浪费预算，并可能强化不该生成的提示内容。

对于多轮示例，要明确每一轮哪些 token 是监督目标；工具调用格式是否监督，取决于线上是否由模型生成该调用。

聊天模板决定 role 标记、特殊 token 和轮次边界。应使用与底座相同 tokenizer 的 `apply_chat_template`，训练时通常 `add_generation_prompt=False`，并检查渲染后的 token，而非只看字符串。模板中多一个换行或错误的 special token 都可能造成分布偏移。

assistant 回复末尾的 turn-end、EOS 或 eot token 是模型需要学会输出的目标，不能机械地把所有分隔符 mask 为 0；应按底座模板和 collator 的定义决定哪些结束 token 属于 assistant 标签。模板和停止 token 要在训练、离线评测、服务端 generation 三处保持一致。

参考 [Transformers 聊天模板文档](https://huggingface.co/docs/transformers/chat_templating)。

长度处理有三个选择。truncation 把过长样本截断，必须决定从尾部还是头部截断，并避免把问题截掉只剩答案；padding 让 batch 对齐，通常配合 attention mask；packing 把多个短样本拼到一个上下文窗口，提高 token 利用率，应核实是否采用样本间 attention 隔离；仅插入 EOS 并不阻止后一个样本关注前一个样本。

因果 mask 防止看到未来 token，loss mask 只决定哪些位置参与损失，两者不能混用。用 `max_seq_length` 时，统计被截断样本比例；若餐厅规则常在文档末尾，盲目截断会造成系统性漏答。

训练命令之前先处理数据。按门店、时间、问题模板和用户会话去重，按会话或来源拆分 train/validation/test，避免同一问题的改写同时出现在训练和验证中。最新规则、人工升级案例和安全拒答应放进独立测试。隐私字段要脱敏，也要确认租户、员工、订单和客户数据的使用授权；数据被复制到日志、缓存或第三方平台，同样属于数据边界。

质量检查包括：答案是否真由工具结果支持、chosen/rejected 是否有明确差异、拒答是否合理、标签员一致性、语言和格式分布、异常长样本以及重复率。

超参没有通用固定值。需要记录 base model、tokenizer、模板版本、数据 revision、学习率、scheduler、warmup、epoch、weight decay、最大长度、精度、梯度累积与随机种子。

effective batch 要说明数据并行（DP）副本数与每卡 batch：`effective examples = micro_batch × grad_accum × DP`；对语言模型更可比的是有效 token 数，padding、packing 和不同长度会使“每 step”含义不同。梯度裁剪只改变异常大梯度的更新，不能替代找出数据、学习率或数值溢出的根因。

显存也不能只按“7B”回答。GB 是十进制单位，GiB 才是 `2^30` 字节。若把 7B 仅按每参数 2 bytes 的权重存储估算，是约 14 GB；全参 Adam 还需梯度、FP32 master weight 和一阶/二阶状态，激活、临时 buffer、通信和框架碎片另算。

一个常见教学账本是每参数约 2B weight + 2B grad + 4B master + 8B moments = 16B，7B 参数约 112 GB，即约 104 GiB，但这只是某种实现和精度组合，不是硬件保证。LoRA 省的是可训练参数的梯度与优化器状态，但推理底座仍要加载；QLoRA 用低比特底座进一步节省显存。

## 从样本到梯度：SFT 的一个训练 step

### 两种最小数据格式

一条客服 SFT 样本可以保留结构化消息，训练前再交给模板渲染：

```json
{"messages":[{"role":"system","content":"你是门店客服，不能猜测过敏原。"},{"role":"user","content":"两人套餐含花生吗？"},{"role":"assistant","content":"我先查询本店今日套餐成分；若结果不完整，我会转人工确认。"}]}
```

偏好样本必须保证 prompt 相同、两个回答可比：

```json
{"prompt":"两人套餐含花生吗？","chosen":"我会先核对本店成分表；当前结果不完整，不能保证无花生，建议联系门店并转人工。","rejected":"放心，套餐肯定不含花生，直接下单即可。"}
```

样本中可以额外保存 `tenant_id`、门店、工具返回快照、标注来源和 revision，但送入模型前要按权限清理。偏好标签不是事实真值：chosen 代表在给定标准下更好，仍需要独立事实测试。

### 前向、反向和更新

一个真实训练 step 不是“调用 trainer 就结束”。数据加载器取 batch，tokenizer 和 collator 生成 `input_ids`、attention mask 和 labels；模型前向得到 logits；交叉熵只聚合有效 label；反向传播得到可训练参数的梯度。

若用了梯度累积，要按有效 token 数及框架的归一化约定聚合多个 micro-batch；只有等权 batch 等条件下才可简单按累积步数平均。随后再在边界处裁剪梯度、执行 optimizer.step、scheduler.step、清空梯度和记录日志。DP 通信会在反向阶段同步梯度；混合精度还要经过 scaler 或 autocast。

```text
load -> template/tokenize -> forward -> masked loss
     -> backward -> accumulate -> clip -> optimizer.step
     -> scheduler.step -> zero_grad -> checkpoint/eval
```

理解这条链能解释常见故障：只更新 adapter 是因为其他参数 `requires_grad=false`；loss 统计正常但参数不变，可能是忘了 `step` 或梯度全被 mask；梯度累积的有效 batch 增大，却不一定降低单卡峰值激活。

### 超参数首先要能复现

| 参数 | 它控制什么 | 实验记录方式 |
| --- | --- | --- |
| learning rate | 每次参数更新尺度 | 记录 base、adapter、head 是否分组 |
| epochs/steps | 数据重复或更新次数 | 同时记录有效 token 数 |
| max length | 单样本窗口上限 | 报告截断比例和分桶长度 |
| r、alpha、dropout | LoRA 容量、缩放、正则 | 写清 target modules |
| warmup ratio/steps | 早期学习率爬升 | 与总 steps 一起记录 |
| precision | 权重、激活和计算 dtype | 说明 BF16/FP16、量化和 scaler |

这些字段没有适用于所有模型的神奇固定值。先小规模扫学习率、rank 或长度，再用验证集、独立任务和成本选点。调整 epoch 的同时改变数据重复次数，不能把结果简单归因于 LoRA 配置。

## 选择更新范围：全参、部分参数、LoRA 与 QLoRA

全参 SFT 的表达能力和可塑性最大，适合数据量、算力和模型版权都充足且确实需要改变底座能力的任务。代价是显存、训练时间、灾难性遗忘和版本管理成本高；一条不严谨的客服数据可能改坏通用推理和安全边界。部分参数 SFT 是工程折中，例如冻结 embedding 与底层，仅解冻高层 attention/MLP，或只训练 lm_head。

它能降低成本，但解冻范围是架构相关选择，应以独立任务评测验证，不能只凭“最后几层有效”的经验套用。

LoRA 把某个线性层更新写成低秩增量：

`W' = W + (alpha / r) B A`

底座 `W` 冻结，训练小矩阵 `A`、`B`；`r` 是 rank，决定低秩容量，`alpha` 是缩放系数，实际增量尺度与 `alpha/r` 有关，不能把 alpha 单独理解成学习率。`dropout` 只作用于 adapter 分支的输入或相关实现，用于正则化。

`target_modules` 指定在哪些模块插入 adapter，如 attention 的 `q_proj`、`v_proj`；不同架构命名不同，应打印模块名并确认可训练参数数量。

Hugging Face PEFT 的 [LoRA 配置文档](https://huggingface.co/docs/peft/package_reference/lora) 说明了 `r`、`target_modules`、`lora_alpha`、`lora_dropout` 等字段。

rank 太小可能表达不了复杂格式或领域风格，太大则增加参数和过拟合风险；target 过窄可能学不会需要改动的路径，过宽会增加成本。一个可解释的最小实验是固定数据和训练步数，只比较 `q/v` 与全部线性层、两个 rank 和是否 dropout，报告独立任务、通用能力、安全集和显存，而不是只比较训练 loss。

LoRA adapter 可独立保存，线上按租户或业务版本选择；也可把增量 merge 到底座得到一个权重文件。merge 后不能再把它当成同一个“裸 base + adapter”版本，回滚需保留原始底座和 adapter 元数据。

QLoRA 通常以 4-bit 存储冻结底座，计算时反量化至指定计算精度，梯度通过底座计算图传到可训练的 LoRA 参数。参数存储精度与计算精度要分别记录。量化方式、double quant、计算 dtype、paged optimizer 等都是实现细节，效果取决于库和硬件。

PEFT 文档给出了用 `target_modules="all-linear"` 覆盖线性层的 QLoRA 风格配置，见[量化指南](https://huggingface.co/docs/peft/developer_guides/quantization)。量化误差可能影响小数据任务；因此要比较同一 checkpoint 的 FP16 LoRA、QLoRA 和合并后量化推理，检查答案与延迟。

## 显存和并行：训练能跑到哪里

### LoRA 参数量的可核算例子

对一个输入维度 4096、输出维度 4096 的线性层，rank=8 的 LoRA 参数是 `A: 8×4096` 加 `B: 4096×8`，合计 `65,536` 个参数；原始权重是 `4096×4096=16,777,216` 个参数，adapter 约为原层的 0.39%。

若 q、k、v、o 四个矩阵都为上述尺寸，同时添加 adapter 则是 262,144 个参数；GQA 等架构的 k/v 尺寸可能不同，必须按实际 shape 计算。实际总数还取决于层数、MLP target 和是否训练 bias。这个算式能帮助解释“少量参数”，也提醒我们不能只看 rank 而忽略 target 数量。

### 分布式和激活节省

ZeRO-1 主要切分 optimizer state，ZeRO-2 再切分梯度，ZeRO-3 连参数也分片；它们减轻单卡状态压力，但增加通信和实现复杂度。FSDP 将参数、梯度和状态按 shard 管理并在需要时 all-gather，效果依赖 wrapping、通信和 checkpoint 策略。

gradient checkpointing 不保存所有中间激活，而是在反向时重算，通常用计算换显存。FlashAttention 是对 attention 计算和 IO 的优化 kernel，减少中间读写，并不改变训练目标，也不自动解决长序列的总计算量。

LoRA 场景常因 adapter 状态小而无需最复杂的并行，但底座仍可能很大。报告时应写卡数、DP/TP/FSDP 或 ZeRO 配置、序列长度、micro-batch、有效 token、峰值显存和计时区间。这样别人才能复现“能跑”的条件。

### QLoRA 的三个组件

NF4 是面向近似正态分布权重的 4-bit 表示，降低存储；double quant 对量化常数再次量化，继续减少量化元数据；paged optimizer 用分页方式管理优化器状态，降低显存峰值尖峰。4-bit 底座通常以低比特存储，计算时反量化到指定计算 dtype；LoRA 参数可以使用 FP32 或混合精度，不能概括为必然 FP16/BF16。

QLoRA 原论文见 [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)。

## DPO：直接利用偏好对

DPO 数据是一条 prompt、chosen、rejected。设训练策略为 `pi_theta`，冻结的 reference 为 `pi_ref`，对回答 token 的 log probability 求和得到 `log pi(y|x)`。常见 DPO 损失是

`L_DPO = - log sigmoid( beta * ( [log pi_theta(y_w|x)-log pi_ref(y_w|x)] - [log pi_theta(y_l|x)-log pi_ref(y_l|x)] ) )`

`y_w` 是 chosen，`y_l` 是 rejected。乘上 beta 后的差值可解释为隐式奖励差；reference 提供相对概率基准，来源于带 KL 正则的偏好优化推导。DPO 论文把 KL 约束的奖励最大化改写为偏好数据上的分类式目标，参见[原论文](https://arxiv.org/abs/2305.18290)。实践中常用 SFT checkpoint 作为 reference，而不是随意用另一个不匹配的模型。

括号内部是未乘 beta 的 log-ratio 差，若把它当作 implicit reward，通常定义为 `r_hat = beta * (log pi_theta(y|x)-log pi_ref(y|x))`；因此不能漏报 beta。`beta` 是温度/正则强度相关的系数，改变 logistic 目标对相对 log-ratio 的敏感度；不能简单说“beta 越大更新越大”。

它与数据噪声、学习率、序列长度和 reference 差异共同决定梯度，需用验证集和稳定性指标调节。长回答的 log probability 是 token 求和，天然有长度效应：回答更长可能绝对 logp 更低或差值更大。原始 DPO 使用回答 token 的 logp 求和，改成长度平均会改变优化目标，不能当作无影响的开关；不能把不同长度样本的 margin 直接横比。

DPO 不需要单独训练 reward model，也不需要 PPO 的在线 rollout，工程链路更短；但它仍依赖偏好标签质量，可能学会格式偏好、冗长和标注器偏差。记录 chosen/rejected 的长度、token 数、reference logp、policy logp、隐式 reward、margin 与 preference accuracy。

accuracy 高而独立任务变差，通常说明模型记住了偏好模式或过拟合；margin 极端增大也不是“对齐完成”的证明。DPO 可与 LoRA 组合，仍要先说明优化目标是 DPO、更新范围是 adapter。

## PPO 与 RLHF：在线反馈带来的复杂度

经典 RLHF 通常先用 SFT 模型收集人类偏好，训练 reward model（RM），再让当前 policy 生成 rollout，由 RM 打分，用 PPO 更新 policy；同时用 value model 估计未来回报，并以 reference policy 的 KL 惩罚限制漂移。

InstructGPT 论文描述了这种 SFT、偏好模型和 RLHF 的组合流程，见[论文](https://arxiv.org/abs/2203.02155)。PPO 的基础算法来源见[原论文](https://arxiv.org/abs/1707.06347)。

一次 rollout 可抽象为：prompt 进入 policy，模型生成回答；RM 输出奖励；每个 token 的 KL 惩罚约束 policy 相对 reference 的变化；value head 估计 `V(s_t)`；用回报与 value 的差得到 advantage `A_t`；最后以旧策略与新策略的概率比 `r_t(theta)=pi_theta(a_t|s_t)/pi_old(a_t|s_t)` 更新，并最大化以下 clipped surrogate（若以 loss 最小化实现，需要取负号）：

`L_clip = E[min(r_t A_t, clip(r_t,1-epsilon,1+epsilon) A_t)]`

这里的 `pi_old` 是采样这批 rollout 的旧 policy，`approxkl`/ratio 反映当前更新相对 old 的变化；它和 policy 相对 reference 的 KL 是两件事。前者用于 PPO 步长稳定性，后者是 RLHF 目标中的漂移惩罚；把二者混为一个指标会导致错误调参。

PPO 还要处理 value loss、entropy、GAE、padding 和终止 token，系统复杂度显著高于离线 DPO。

RM 本身通常先用 pairwise loss 学习：让 `r_phi(x,y_w)` 高于 `r_phi(x,y_l)`，例如 `-log sigmoid(r_w-r_l)`。GAE 则把多步 TD 误差按衰减系数累积成 advantage，在偏差和方差之间折中。

PPO 指标要成组看：RM reward 上升是否伴随人工偏好和独立任务上升；reference KL 是否失控；old/new `approxkl`、clipfrac 是否显示更新过大或几乎没有有效更新；entropy 是否迅速坍缩；value loss 和 explained variance 是否说明 value 学得住。

只看 RM reward 容易 reward hacking，例如客服模型学会重复“非常抱歉”、堆砌免责声明或钻评分器漏洞，却不再准确查库存。此时应冻结 checkpoint，抽样人工复核，增强对抗集和 RM 训练覆盖，并检查 KL 与长度奖励的设计。

GRPO 可先记作 PPO 家族的简化方向：一组同 prompt 的回答相互比较，用组内相对奖励构造优势，减少显式 value model 的依赖。它常用于可验证奖励的推理任务，但不是本教程的串行主线；餐饮客服这种工具调用、权限和多轮状态问题，不能因为名字新就自动替换 SFT、DPO 或完整 RLHF。

## 指标、实验和故障诊断要连起来

训练 loss 适合看优化是否在工作，不能直接等价于线上质量。只有当 loss 是自然对数、按有效预测 token 平均的负对数似然时，PPL 才是 `exp(loss)`；任意经过 label smoothing、加权、sequence average 或混合目标的 loss 都不能直接叫 PPL。

它受 tokenizer、mask、数据分布影响，跨 tokenizer 或不同有效 token 定义比较没有意义。`tokens/sec` 的分母是经过的时间，分子要说明是输入+标签 token 还是非 padding token，计时是否包含 packing、梯度累积、数据加载和通信；否则两次吞吐数字不可比。

| 指标 | 看什么 | 常见误读 |
| --- | --- | --- |
| train/validation loss、PPL | 优化与泛化趋势 | 越低不代表事实、格式和安全都更好 |
| tokens/sec、step time | 训练吞吐 | token 计数、单卡/全局范围和计时口径不同则不可比 |
| peak memory、GPU 利用率 | 资源上限与瓶颈 | 7B 参数大小不是完整训练显存 |
| grad norm、LR、NaN/Inf | 数值稳定与更新尺度 | NaN 不是靠继续跑就能恢复 |
| DPO implicit reward、margin、accuracy | chosen 与 rejected 的相对学习 | 高 margin 可能是长度效应或过拟合 |
| PPO RM reward、reference KL、approxkl、clipfrac | 奖励、漂移和裁剪 | approxkl 不是 reference KL；RM reward 会被 hacking |
| PPO entropy、value loss | 探索和价值估计 | 单独看 value loss 不能证明 policy 更好 |
| 独立任务、安全回归、成本 | 真正上线约束 | 只看训练集 loss 会漏掉回归和账单 |

最小实验应先固定一个小而干净的数据集、一个 base checkpoint、tokenizer/template 和评测脚本。建立未训练基线；再跑短步数 full/LoRA 或两个 LoRA 配置，保留同一验证集。餐饮客服评测至少分：准确引用工具结果、过敏原安全拒答、门店权限隔离、格式与 EOS、长上下文、未知问题转人工、通用能力回归。

记录 checkpoint、config、数据 hash、运行日志和成本；每次只改一个主要变量。

故障诊断按证据链走。loss 不降先检查 labels 是否全是 `-100`、shift 是否错位、tokenizer 与模板是否匹配、学习率和 dtype；loss 变 NaN 检查输入 NaN、溢出、异常长样本、梯度 norm、混合精度 scaler 和 optimizer state。

训练正常但模型复读 user，检查 assistant mask；回答不停止，检查 EOS、generation stop 和模板；离线好线上差，逐字节比较线上 prompt、system 消息、tokenizer、adapter 加载、量化和 generation 参数。

DPO accuracy 很高但人工偏好下降，检查长度归一化、重复问题泄漏、chosen/rejected 标签反转和 reference 是否正确；margin 异常大时按 token 长度分桶。PPO reward 上升而回答变差，检查 RM 输入模板、长度奖励、reference KL、entropy、clipfrac 和人工抽样，切勿只继续增加训练步数。

显存 OOM 则分别缩短序列、减小 micro-batch、启用 gradient checkpointing、减少可训练状态或采用量化；每个措施都要重新测峰值和质量。

部署时把 base model、adapter、merge 状态、量化方式、tokenizer、chat template、generation defaults 和训练数据版本作为同一个可回滚制品记录。adapter 模式适合多个业务版本和快速回滚；merge 适合固定服务，但会失去灵活切换。量化后重新测工具调用、中文、EOS 和安全集。

服务端若换了模板或把 system 消息拼接两次，权重没变也会像“微调失效”。微调改变模型行为，不能替代服务端租户授权、工具权限、事实校验和审计。

### 指标的分子与分母

当每个样本只应调用一次工具时，端到端工具调用正确率可定义为“选择正确工具且参数完全正确的样本数 / 应调用工具的测试样本数”。另报条件参数正确率：“参数正确的调用数 / 实际被评测的调用数”，同时报告漏调用和误调用；任务成功率是“完成业务验收条件的会话数 / 进入该任务测试的会话数”。幻觉率可以定义为“包含无法由给定证据支持的事实断言的回答数 / 需要事实判断的回答数”，必须在标注规范中说明什么算断言。

拒答 precision 是拒答中确实应该拒答的比例，recall 是应该拒答的样本中实际拒答的比例；二者不能只报一个。对二分类指标给出 bootstrap 或二项区间，注明样本数、抽样单位和随机种子。成对盲评要随机化回答顺序、隐藏模型名，并记录评审员、题目难度和一致性；同一租户泄漏、评审员知道实验条件、只挑成功案例都会造成偏差。

### 训练指标与业务指标并列

SFT validation loss 只能回答“在这个 token 分布上是否拟合”，不能证明工具调用参数正确。DPO accuracy 只能回答偏好对方向是否学到，不能证明事实准确。PPO 的 RM reward、KL、approxkl、clipfrac、entropy、value loss 要与人工盲评和任务成功率一起看。

成本应包含训练 GPU 时间、评测调用、adapter 存储、merge/量化和线上延迟，不能只报一次训练账单。

### 四步最小闭环

实验可以分成四段推进：先冻结可追溯的 base、tokenizer、template 和数据 revision，建立包含安全、工具、长上下文与通用能力的未训练基线；再用几十到几百条高质量样本短跑，确认 label mask、EOS/eot、有效 loss token 和 adapter 更新都正常。随后每轮只改一个主要变量，比较 full/partial/LoRA 或不同 rank，记录有效 token、峰值显存和有明确计时范围的 tokens/sec。最后用独立时间切分、未见门店、对抗安全集和人工盲评检查，给出置信区间与失败样例；如果训练和验证都变好而独立任务退化，就不应继续加大训练。

### 发布前的制品检查

发布制品应绑定 base revision、adapter revision、merge 状态、量化方案、tokenizer、chat template、停止 token、generation 参数和评测结果。先以 adapter 模式灰度，确认租户选择、权限、回滚和并发加载，再决定是否 merge。对 merge 后权重重新运行相同评测，不能假设数学上的 `W+BA` 就代表服务端行为完全一致。

最后用真实客户端请求验收：比较线上最终 prompt 和训练模板，确认工具 schema、权限校验、超时、重试和审计；抽查过敏原、库存、退款和未知问题转人工。任何“HTTP 成功但答案错误”的情况都要回到请求、模型、工具和证据链定位。微调只负责可学习的行为分布，动态事实和授权仍由系统控制。

### 面试复述与继续学习

面试可以这样复述：先区分目标和参数范围，CPT 补领域分布，SFT 学示范行为，DPO 从 chosen/rejected 离线学习相对偏好，PPO 通过 RM、value、advantage 和受约束的在线 rollout 更新 policy。

工程上先治理会话级数据拆分、权限和模板，再选择 full、partial 或 LoRA/QLoRA；用 mask、EOS、packing 保证训练目标正确，用 loss 之外的独立任务、安全、显存、吞吐和成本验收。出现线上回归时，沿 tokenizer/template、adapter/merge、量化、generation 和工具授权逐层对齐证据，而不是把“loss 降了”当成上线理由。

进一步阅读：[PEFT 参数高效微调概览](https://huggingface.co/docs/transformers/peft)、[LoRA 原论文](https://arxiv.org/abs/2106.09685)、[Transformers 聊天模板](https://huggingface.co/docs/transformers/chat_templating)、[TRL SFTTrainer 文档](https://huggingface.co/docs/trl/main/sft_trainer)、[TRL DPOTrainer 文档](https://huggingface.co/docs/trl/main/dpo_trainer)、[TRL PPO 指标说明](https://huggingface.co/docs/trl/ppo_trainer)、[DPO 原论文](https://arxiv.org/abs/2305.18290)、[InstructGPT](https://arxiv.org/abs/2203.02155)、[PPO 原论文](https://arxiv.org/abs/1707.06347)。
