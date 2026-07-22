# DeepSeek-V4-Flash(含量化版)draft 训练适配书

> 更新:2026-07-13。前置:先按《npu适配计划书》跑通 Qwen3-8B/32B。

## 1. 现状:DSv4 支持不在 main,在分支上

核心分支 **`origin/rtuli/dflash-dsv4-training-v2`**(5 commits,~1300 行,只覆盖 dflash):

| 内容 | 说明 |
|---|---|
| `hc_mult` | DSv4 每层输出多通道 hidden states,draft 的 fc 输入变宽(`len(target_layer_ids) * hc_mult * hidden_size`) |
| `hc_head_project` | vLLM `_hc_head_fused_reference` 的纯 PyTorch 移植(fp32 matmul + sigmoid 加权求和),**无 CUDA 专属算子,NPU 友好** |
| 非标准权重名加载 | DSv4 checkpoint 用 `embed.weight`/`head.weight`/`norm.weight`/`hc_head_fn|base|scale`,分支重写了 `load_verifier_weights` 按精确 key 从 safetensors 读 |
| vendored DSv4 chat template | 561 行,预处理管线用 |
| nested `rope_parameters` 处理 | draft config 构建时展开 |
| chunked prefill 可开关 | DSv4 长 prompt 需要开 chunked prefill(旧抽取路径要求关) |
| online/offline 示例脚本 | 含运维参数:`REQUEST_TIMEOUT=1800`、`SPECULATORS_DIST_TIMEOUT_SEC=3600`(DSv4 抽取慢,防集合通信超时) |

合入注意:分支只在 dflash 验证;**dspark 继承 dflash 理论上自动获得 hc_mult,但需补测**(dspark 的 Markov/confidence head 与 hc_mult 正交,预期无冲突)。

## 2. 量化版的关键认知

**量化只发生在 vLLM 推理端。** hidden states 是 bf16 激活值,训练端 draft 始终 bf16。所以"量化版训练"≈"量化版 serve + 原有训练流程",且这是**优点**:draft 学到的是部署时量化 verifier 的真实分布(on-policy 一致性)。

## 3. 工作清单

### A. 本仓库 — 合入
1. `rtuli/dflash-dsv4-training-v2`(必须),rebase 到 main 后补 dspark 路径测试
2. `sp_ulysses`(可选,>8k 序列)

### B. 本仓库 — 新开发(小)
1. **量化 checkpoint 权重加载**:分支版 `load_verifier_weights` 按精确 key 读 safetensors。Ascend 量化(modelslim W8A8/W4A8)导出若把 `head.weight` 等改名/量化/加 scale 伴生张量,则加载失败。两个方案:
   - 简单(推荐先做):加 `--verifier-weights-path`,让 embed/head/norm/hc_head 这几个权重从原始 bf16 checkpoint 读(safetensors 支持按 key 拉取,不必下全模型);
   - 完整:识别量化格式并反量化这几个 key。
2. NPU 通用性复查:hc_head_project 已是纯 torch;确认新增路径无 torch.compile/flex 依赖(分支在 GPU 上开发)。

### C. vllm-ascend 侧(仓库外,最大不确定性,建议最先摸底)
1. 支持 DeepSeek-V4-Flash 架构(MLA、hc 头)的推理;
2. 支持 Ascend 量化格式加载(**RedHat 的 FP8 checkpoint 在 910 系不可用**,需 modelslim W8A8/W4A8 权重);
3. `speculative_config {"method": "extract_hidden_states", eagle_aux_hidden_state_layer_ids}` + `ExampleHiddenStatesConnector` 在**量化算子路径**下仍能吐出指定中间层激活——量化 kernel 融合后抽取点可能不存在,这是最可能缺的一环。

### D. 验证
1. bf16 DSv4 小规模训练做基线(如果放得下),再换量化 verifier 对比 EAL/per-position acceptance,确认量化 hidden states 没有导致明显退化;
2. 长序列(8k)+ chunked prefill 开启时抽取正确性(token_ids 校验已在 dataloader 里);
3. 吞吐:hc_mult>1 时每 token 传输的 hidden states 更大,确认共享盘/网络带宽不是瓶颈。

## 4. 资源布局参考

- 分支 GPU 示例:单机 4 卡 vLLM(DP4 + EP)+ 4 卡训练,seq 8192,5 epochs,muon(MUON_LR=0.02),NUM_LAYERS=5,TARGET_LAYER_IDS="3 13 23 32 42",全层 SWA(window=2048)
- 910C 集群:W8A8 权重减半后按 vllm-ascend 实测确定 TP/EP 布局;训练 draft 只需少量卡,大部分卡给 serve
- trainer 单 endpoint 限制同样适用:多 vLLM 实例 → 负载均衡

## 5. 风险排序

1. 🔴 vllm-ascend 量化 DSv4 + hidden states 抽取(C.3)——决定整个项目可行性,先摸底
2. 🟡 量化 checkpoint 权重 key 兼容(B.1)——确定量化导出格式后一天内可解
3. 🟡 dspark + hc_mult 组合未验证(A.1)
4. 🟢 训练侧 NPU 兼容——Qwen3 试验会先扫清(见 npu 计划书)
