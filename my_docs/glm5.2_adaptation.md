# GLM-5.2 DSpark 昇腾适配实践 —— 对我方工作的可用性评估

> 分析对象：内部稼先文章《GLM-5.2（753B MoE）投机解码：昇腾双机上的 DSpark 草稿模型全链路训练实践》（2026-07-13）
> 核对基准：speculators main @ 6030a44（2026-07-10）、vLLM main @ 487dfb3（2026-07-13）
> 结论：**高度可用。该工作与我方目标路线（speculators + DSpark + 昇腾）高度重合，且已解决我方风险清单上的三项关键前置问题。建议立即建立对接。**

---

## 1. 总体判断

这篇文章是目前已知**唯一**在昇腾上把 `speculators + DSpark` 全链路（数据 → 在线特征 → 训练 → 评测）跑通到生产规模的实践。它不是一篇纯经验分享，而是包含了可直接复用的代码级修复。

对我方的价值可分为三层：

| 层级 | 内容 | 对我方的可用性 |
|---|---|---|
| **L1 必须拿到** | FSDP2 混合精度修复、NPU 注意力后端适配、`--from-pretrained` 修复 | 直接影响训练正确性，不拿会重复踩坑并损失精度 |
| **L2 值得参考** | 双机在线特征架构、吞吐优化、checkpoint 元数据一致性、八项杂项坑 | 视我方是否走 online 模式而定 |
| **L3 仅作对标** | GLM-5.2 的训练配方与接受长度数字 | 目标模型不同，仅作收敛基线参照 |

---

## 2. L1：必须拿到的三项（直接影响正确性）

### 2.1 FSDP2 混合精度策略静默失效 —— 最高价值单项

**问题**：训练配置了 `MixedPrecisionPolicy(param_dtype=bfloat16)`，设计语义是主权重保持 fp32、仅计算时转 bf16。但实际不存在 fp32 主副本，优化器步进的张量是 bf16 的。后果：

- bf16 在 1.0 附近的最小可表示间隔为 2⁻⁹（约 2e-3），而归一化层权重的单步更新量级仅 1e-7 ~ 2e-5 → **每步更新在写回时被舍入吞没，权重被"冻结"在初始值 1.0**；
- **AdamW 的解耦权重衰减对全部参数、每一步被静默丢弃**（该角度上游此前无人提及）。

**代码核实（已验证，截至今日仍在 speculators 主线）**：

`src/speculators/train/distributed.py`：

```python
mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
)

for layer in model.layers:
    fully_shard(layer, mp_policy=mp_policy)   # decoder 层带精度策略

fully_shard(model)                            # ← 根节点漏传 mp_policy
```

**根节点持有的参数恰恰是受害最深的一批**：`self.norm`、`hidden_norm`、`verifier_norm`、`fc` 投影层，以及 **DSpark 新增的 markov_head 与 confidence_head**。

> 注：文章还提到「模型在 fully_shard 前已被整体转为 bf16」。该行为在当前 main 中未找到显式的全模型 `.to(bfloat16)` 调用，可能与版本有关，需按我方实际使用版本复核。但**根节点漏传 mp_policy 这一条已确认存在于主线**。

**修复（两处）**：
1. 参数以 fp32 进入 `fully_shard`（分片存储 fp32，精度策略按设计将聚合参数转 bf16 计算）；
2. 根节点 `fully_shard(model)` 补传同一份 `mp_policy`。

**代价**：每卡常驻显存 +约 1.7 GB，步速不变，checkpoint 体积与格式不变。

**收益（受控消融，同数据/同超参/同种子）**：

| 数据规模 | 修复前 val accept_len | 修复后 | 增幅 |
|---|---|---|---|
| 10k × 3 ep | 1.225 | 1.578 | **+28.9%** |
| 50k × 3 ep | 2.9596 | 3.3583 | **+13.5%** |

**⚠️ 对我方的直接行动项**：

> **立即对我方已有的全部 DFlash / DSpark checkpoint 跑一遍 canary 检查**：扫描所有含 `norm` 的张量，计算 `max |w − 1|`。病态路径下该值**恰好为 0**。
>
> 这是一个零成本判据。虽然我方主要在 SpecForge 上训练（分片实现不同，未必命中同一 FSDP2 路径），但该缺陷的**症状特征**（深位接受率偏低且不随训练改善、val 指标正常但实际效果差）与我方此前观察到的若干现象吻合，值得优先排除。
>
> 特别是 Magistral-Small-2509 从头自训"效果不佳"这一结论——主因大概率仍是数据量不足，但在排除该缺陷之前，不宜将其作为定论写入报告。

**上游状态**：社区 #668 独立发现在先（因缺真实训练证据被搁置关闭）；该团队的机理链、权重衰减取证、两规模消融与 canary 脚本已提交至维护者重构 PR **#711（open，进行中）**。→ **我方不能等上游合入，需自行 patch。**

---

### 2.2 NPU 注意力后端适配

**问题**：昇腾上没有 FlexAttention，而 DFlash / DSpark 的块状稀疏注意力依赖它。这是训练核心路径能否跑起来的先决条件。

**他们的解法**：稀疏 `BlockMask` 物化为**稠密布尔掩码** + 路由至 **NPU 融合注意力算子**，并与上游转换器做了**逐位一致性校验**。

**与我方已有工作的关系**：

| | 我方（SpecForge） | 他们（speculators） |
|---|---|---|
| 实现 | `Qwen3DFlashNPUFlashAttention`，用 `npu_fusion_attention` 替换 `flex_attention` | 稠密掩码物化 + NPU 融合算子路由 |
| 框架 | SpecForge | speculators |
| 校验 | —— | 与上游逐位一致 |

**思路同构，但落在不同框架里。** 我方迁移到 speculators 时，**不需要从零重做**，但需要确认两点：

1. 上游 PR #589（已合入）提供的 `--draft-attn-impl {sdpa, eager}` 是否已足够——若足够，则他们的融合算子路由属于**性能优化**而非**功能必需**；
2. 他们的稠密掩码物化在长序列下的显存开销（掩码是 O(L²) 的），以及与 `--max-anchors` 的交互。

**建议**：优先用上游的 `--draft-attn-impl sdpa` 跑通功能，再评估是否引入他们的融合算子路径提速。

---

### 2.3 `--from-pretrained` 丢弃注意力后端选择（#754 / #755）

**问题**：HF config 体系不序列化 `_attn_implementation` 运行时字段，而 speculators 的 `--from-pretrained` 加载路径不重新应用命令行的注意力后端选择 → **加载的模型静默回落到默认实现**。

- 在 CUDA 上：因 FlexAttention 可用而**不可见**；
- 在 NPU 上：**直接崩溃**。

**⚠️ 这条精准打在我方最核心的工作流上。** 我方在 Qwen3-32B 上的最佳实践是「基于开源 draft model 续训 5 epoch」，走的正是 `--from-pretrained` 这条路。

**上游状态**：#754（issue）+ #755（PR），由该团队首报首修，已完成两轮评审（LGTM），**待维护者终审合入**。

**行动项**：直接取 #755 的 patch，不要等合入。

---

## 3. L2：值得参考的架构与工程经验

### 3.1 双机分离架构（online 特征通路）

```
[推理机 · 昇腾 NPU]                    [训练机 · 昇腾 NPU]
 vLLM 承载目标模型 (w8a8)                speculators 训练循环 (FSDP2 全分片)
 hidden-states connector                 按需拉取特征文件，用后即删
 抽取指定 5 层位隐状态 → 落本机高速盘  ←── HTTP 直连拉取
```

**关键演进**：初版走共享 NFS，受制于 NFSv3 单连接吞吐上限成为步速瓶颈；改为训练侧 **HTTP 直连推理机拉取**后：

- 单流：约 30 → **85 MB/s**
- 多流聚合：约 74 → **294 MB/s**

配合「临时文件写入 + 原子改名 + 就绪判定」握手协议，杜绝读到半成品特征。

**对我方的适用性判断**：

| 场景 | 是否适用 |
|---|---|
| 他们：753B MoE（w8a8 后仍 ~750GB），**训练侧根本装不下 verifier**，只能跨机在线抽取 | 必须 |
| 我方：Qwen3-32B / Magistral-Small-2509，量级小两个数量级 | **不必须**。我方可用 offline 或混合模式（`--on-missing generate --on-generate cache`），首轮生成并落盘、后续 epoch 直接读盘 |

→ **该架构对我方是"备选方案"而非"必选方案"。** 但如果我方后续要训 Qwen3-235B 一类的 MoE 目标模型，这套拓扑是现成的。

### 3.2 训练吞吐优化：48.5 → 8.2 秒/步（5.9×）

| 优化项 | 内容 | 上游状态 |
|---|---|---|
| 特征保存同步阻塞推理引擎 | 改为异步线程 + 临时文件原子改名 | 上游现行版本已有等效修复 |
| 特征写盘在 TP 各 rank 重复执行 | 加 rank-0 门控 | 上游现行版本已有等效修复 |
| 训练侧特征读取走共享 FS | 改 HTTP 直拉 | 他们自研 |

**参考价值**：8.2 秒/步（5 层 draft、block_size 8、seq 4096、753B 目标模型在环）——这个数字**回答了我方此前"speculators 在 NPU 上训练速度是否够快"的疑问：够快**。我方目标模型小得多，步速应显著优于此。

### 3.3 其余可直接抄的坑（八项）

| 问题 | 根因 | 处置 |
|---|---|---|
| checkpoint 元数据 dtype 不一致 | 张量按惯例转 bf16 落盘，config 却按内存中的 fp32 写出 → vLLM 加载时超额分配显存或校验报错 | 保存后将 config 声明规范化为磁盘实际 dtype（注意不同 transformers 版本键名为 `dtype` 或 `torch_dtype`） |
| 深层特征捕获哨兵不触发 | 末层捕获点位于哨兵判断之后（在特定模型家族上永不触发） | 上游现行版本已含等效修复 |
| 长序列训练数据加载失败 | 打包长度与序列上限**差一** | 预处理按 4095 切齐 |
| `ASCEND_RT_VISIBLE_DEVICES` 置空引发错误绑卡 | **置空字符串与未设置语义不同** | 启动脚本一律显式赋值，纳入发射前检查 |
| 分布式训练偶发通信超时 | 评测负载与训练争抢推理机并发 | 评测让路制度 + 带护栏重启 |
| 异步落盘引入读写竞态 | 训练侧可能读到写入中的特征文件 | 临时文件 + 原子改名 + 就绪判定握手 |
| 数据预处理不保证样本顺序 | prepare 阶段并行分片打乱原始顺序 | 评测线按域切片 + 按样本 id 对账 |
| 同类数值缺陷在 NPU 上显形更早 | 部分 CUDA 算子对 bf16 原地累加内部采用 fp32 中间精度，**NPU 对应算子不做** | 无需单独修复；fp32 主权重方案从根源消除 |

> **最后一条值得单独划出来**：它解释了为什么一类数值缺陷能在 CUDA 生态长期静默、却在 NPU 上立刻显形。这是一条**通用的 NPU 移植经验**——凡是 GPU 上"能跑且看起来正常"的训练代码，在 NPU 上都要额外做数值健全性检查，不能默认精度行为一致。

---

## 4. L3：训练配方与结果（对标基线）

### 4.1 配方（与 Red Hat 公开配方对齐）

| 项 | 取值 |
|---|---|
| 草稿层数 | 5 |
| `block_size` | 8（7 draft + 1 bonus） |
| 目标层位 | `[8, 23, 39, 55, 70]` |
| 词表 | 全词表 |
| lr / scheduler | 6e-4 / cosine |
| 损失 | `{ce: 0.1, tv: 0.9}` + 置信度头 |
| seq_len | 4096 |
| `max_anchors` | **512**（Red Hat 用 1024） |
| 目标模型量化 | w8a8（Red Hat 用 FP8） |

### 4.2 六基准离线接受长度（50k × 3 epoch，k=7 含 bonus，每基准 N=32）

| 基准 | accept_len | accept_rate | pos1 | pos4 | pos7 |
|---|---|---|---|---|---|
| MATH-500 | 4.729 | 0.705 | 0.912 | 0.727 | 0.569 |
| GSM8K | 4.534 | 0.693 | 0.906 | 0.699 | 0.568 |
| HumanEval | 3.828 | 0.582 | 0.837 | 0.598 | 0.431 |
| MBPP | 3.369 | 0.531 | 0.817 | 0.525 | 0.379 |
| MT-Bench | 3.200 | 0.500 | 0.765 | 0.505 | 0.380 |
| Alpaca | 3.130 | 0.497 | 0.754 | 0.499 | 0.400 |
| **六基准均值** | **3.798** | **0.585** | | | |

**形状**：数学 > 代码 > 开放对话，极差约 1.5×；逐位接受率单调递减。

**与我方 DFlash 结果的形状高度一致**（我方 Qwen3-32B：gsm8k τ=2.546 > humaneval 2.206 > 通用/中文对话 1.5~1.7）。这佐证了 DFlash 家族"高接受率场景优势明显、低接受率场景收益有限"的普遍规律。

**规模轴**：50k×3ep → 3.3583（训练轴 held-out）；508k epoch-1 → **3.937**（+17.2%，仅用三分之一训练量）。

### 4.3 ⚠️ 读这些数字时的重要口径提醒

1. **这是离线接受长度（teacher forcing + 前缀匹配），不是端到端加速比。** 文章明确声明「端到端加速比与 TTFT 不在本文范围」。**不能与我方表格里的 τ / TPOT 直接比较。**
2. 参考应答生成用 2048 token 上限，各基准截断率 3.1%~34.4%（合计 15.6%）。
3. 与 Red Hat 的对比中，双方验证划分不同（对方 ep1/ep2 是训练集代理指标），文章自己也声明「不构成严格同条件对比」。

---

## 5. 需要向他们确认的问题（按优先级）

### P0 —— 阻塞性

1. **vLLM 侧的 GLM aux hidden states 抽取补丁在哪？**
   全文未提及。但经核实，**vLLM main 中没有任何 `glm*.py` 实现 `SupportsEagle3` 接口**（`Glm4MoeForCausalLM` 的继承链为 `nn.Module, SupportsPP, SupportsLoRA, Glm4MixtureOfExperts`）。而 `method="extract_hidden_states"` 会强制 `use_aux_hidden_state_outputs=True`，模型不实现该接口时 `gpu_model_runner.py` 会直接抛：
   ```
   RuntimeError: Model does not support EAGLE3 interface but aux_hidden_state_outputs was requested
   ```
   → **他们必然给 vLLM/vllm-ascend 的 GLM 模型打了补丁。这个补丁的形态，决定了他们整套在线特征通路的可复现性。**
   （对我方而言 Qwen3 原生支持该接口，不需要此补丁；但需确认他们用的是**社区 vllm-ascend** 还是**内部分支**——这决定我方能否直接复用其环境。）

2. **FSDP2 混合精度修复的完整 patch**（fp32 主权重进分片 + 根节点补精度策略）与 **canary 检查脚本**。

3. **NPU 注意力后端适配的代码**：稠密掩码物化 + 融合算子路由，及其逐位一致性校验用例。

### P1 —— 影响选型

4. **训练/推理各用了多少卡？** 文章只说「训练：A2 节点；推理：A3 节点」，未给卡数。这直接决定我方在现有卡约束下能否复现，以及 online 模式对我方是否可行。

5. **A5（Ascend 950DT）上的管线打通经验**。文章提到最初在 A5 上完成迁移打通与收敛验证，后因资源调配转到 A3+A2。若我方后续能拿到 950 系列，这条经验是现成的。

6. **speculators 的版本/commit**。DSpark 于 2026-06-29 才合入 main（PR #677），晚于最新发布版 v0.6.0（06-16），因此他们必然是**源码装 main 或 cherry-pick**。需要确切 commit 以复现。

### P2 —— 增量参考

7. `max_anchors` 取 512 而非 1024 的原因（显存约束？还是消融结果？）——这个参数是 DFlash/DSpark 的核心显存旋钮。
8. `--dflash-decay-gamma` 在 DSpark（有 Markov 头之后）取值多少。上游默认 4.0，但 Markov 头补齐块内依赖后，原衰减系数可能过强。

---

## 6. 我方行动建议

| 优先级 | 行动 | 说明 |
|---|---|---|
| **P0** | 对已有全部 DFlash/DSpark checkpoint 跑 norm canary 检查 | 零成本。`max\|w−1\| == 0` 即命中缺陷 |
| **P0** | 联系该团队，取 §5 的 P0 三项 | 直接决定我方 speculators 迁移的起跑线 |
| **P1** | speculators 环境搭建：源码装 main + 打 #755 patch + 打 FSDP2 patch | 三个 patch 都不能等上游 |
| **P1** | 用 `--draft-attn-impl sdpa` 先跑通 Qwen3-32B 的 DSpark 功能验证 | 先功能后性能 |
| **P1** | 复核 Magistral 从头自训"效果不佳"的结论 | 在排除 FSDP2 缺陷之前，该结论不宜作为定论 |
| **P2** | 评估 offline / 混合模式相对其 online 双机架构的优势 | 我方目标模型小得多，无需在线通路 |
| **P2** | 确认 Qwen3.6 PartialRoPE（speculators PR #568，**未合入**）状态 | 我方有 Qwen3.6-27B 线；issue #613 已实锤「Qwen3.6 + DFlash → 验证指标正常但接受率极低」 |

---

## 7. 一句话结论

> **这份工作把我方 speculators 迁移路线上的三块最硬的骨头（NPU 注意力、hidden states 抽取通路、`--from-pretrained` 缺陷）都啃掉了，还额外送了一个价值 13%~29% 接受长度的数值正确性修复。不去对接是纯粹的浪费。**
>
> 但要清醒地看到：**他们的 753B 双机在线架构对我方是过度设计**。我方目标模型量级小两个数量级，走 offline / 混合模式即可，不必照搬其在线特征通路。要拿的是**修复与适配**，不是**架构**。

---

## 附录：相关链接

- speculators #668（混合精度案 · 社区先行修复，已关闭）：https://github.com/vllm-project/speculators/pull/668
- speculators #711（AMP 重构 · 证据与必修项所在，open）：https://github.com/vllm-project/speculators/pull/711
- speculators #754（`--from-pretrained` 缺陷报告，open）：https://github.com/vllm-project/speculators/issues/754
- speculators #755（对应修复 PR，LGTM 待合入）：https://github.com/vllm-project/speculators/pull/755
- speculators #677（DSpark 训练支持，已合入 2026-06-29）：https://github.com/vllm-project/speculators/pull/677
- speculators #568（Qwen3.6 PartialRoPE，**未合入**）：https://github.com/vllm-project/speculators/pull/568
- Red Hat GLM-5.2 DSpark 模型卡：https://huggingface.co/RedHatAI/GLM-5.2-speculator.dspark
