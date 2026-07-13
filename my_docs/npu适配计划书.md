# speculators NPU(Ascend 910B/910C)适配计划书

> 更新:2026-07-13,基于 main(`6030a44`)调研,工作分支 `npu-support`

## 1. 背景与目标

- 硬件:910B 单机试验 → 8 台 × 8 卡 910C(64GB)集群
- 路线:Qwen3-8B 跑通(910B)→ Qwen3-32B(64 卡)→ 量化 DeepSeek-V4-Flash(见另一份文档)
- 训练形态:在线训练。vLLM(vllm-ascend)serve verifier 并抽取 hidden states,训练进程只训小的 draft 模型(dflash/dspark),两边分卡。

## 2. 已具备的 NPU 支持(main 已合入)

| PR | 内容 |
|---|---|
| #232 | Ascend 平台支持:`torch.accelerator` 抽象(trainer/checkpointer/data gen/eagle3),分布式后端自动选 hccl |
| #600 | `conditional_torch_compile`:NPU 无 inductor 时跳过 torch.compile |
| #589 | `--draft-attn-impl sdpa/eager` 稠密 mask 后端(flex attention 依赖 inductor,NPU 不可用) |

结论:eagle3/dflash/dspark 的训练路径在仓库侧基本打通;MTP 例外(强制 flex attention,NPU 暂不可用)。

## 3. npu-support 分支已做的适配(2026-07-13)

1. **`src/speculators/train/trainer.py` `_StepTimer`**:#722(晚于 #232)引入的逐步性能计时器直接调 `torch.cuda.synchronize()`,而 `log_freq` 默认为 1 → NPU 上**第一步训练即崩**。已改为设备无关的 `synchronize()`。这是 910B 上唯一的硬阻塞点。
2. **`src/speculators/utils/util.py`**:新增 `synchronize()`、`manual_seed_all()` 通用助手(与已有 `empty_cache()` 同风格,基于 `torch.accelerator` + `torch.get_device_module`)。
3. **`scripts/train.py`**:
   - NPU 上 `--draft-attn-impl` 默认值 `simple_flex_attention` 自动降级为 `sdpa` 并告警(MTP 除外);
   - `set_seed` 的 `torch.cuda.manual_seed_all` → 通用 `manual_seed_all`(NPU 上也能正确播种);
   - 收尾的 `torch.cuda.empty_cache()` → 通用 `empty_cache()`。
4. **`examples/train/dflash_qwen3_8b_sharegpt_online_npu.sh`**:910B 版示例(`ASCEND_RT_VISIBLE_DEVICES`、显式 `--draft-attn-impl sdpa`、adamw 回退开关)。

## 4. 910B Qwen3-8B 试验步骤

1. 环境:torch + 配套 torch_npu、CANN;serve 侧 vllm + vllm-ascend(可以与训练分开 venv,`launch_vllm.py` 会 re-exec 当前解释器,同 venv 时注意)。
2. **先做 vllm-ascend 能力探测**(整条链路的最大外部依赖):
   ```bash
   python scripts/launch_vllm.py Qwen/Qwen3-8B --target-layer-ids 2 18 33 -- --port 8000
   ```
   确认 vllm-ascend 接受 `speculative_config {"method": "extract_hidden_states"}` 和 `ExampleHiddenStatesConnector` KV connector。**如果这一步不行,训练侧改多少都没用**,需要去 vllm-ascend 提需求/找等价实现。
3. 跑 `examples/train/dflash_qwen3_8b_sharegpt_online_npu.sh`(先 500 样本 1 epoch 冒烟,再 5k)。
4. 观察项:
   - sdpa 后端若在特定 CANN 版本上报算子错 → 换 `--draft-attn-impl eager`(纯 eager + 稠密 float mask,最保险);
   - muon 优化器(默认,纯 matmul Newton-Schulz)若异常 → `--optimizer adamw`;
   - 训练指标:loss 下降 + EAL/per-position acceptance 与 GPU 参考值同量级。

## 5. 扩展到 Qwen3-32B(64 卡 910C)

- draft 模型约 2~3B,FSDP 无压力;大头在 serve 侧:Qwen3-32B bf16 权重 ~65GB,64GB 卡需 TP2~TP4。
- 推荐布局:每节点 4 卡 vllm-ascend(TP4)+ 4 卡训练;先单节点跑通再横向扩。
- 已知限制:trainer 只接受单个 `--vllm-endpoint`(`train/data.py` ArrowDataset)。多节点方案:多 vLLM 实例前挂负载均衡,或小改脚本按 rank 分配 endpoint。
- 长序列(>8k)可考虑合入 `origin/sp_ulysses`(Ulysses 序列并行,DFlash/P-Eagle)。

## 6. 遗留问题清单

| 项 | 状态 | 说明 |
|---|---|---|
| vllm-ascend extract_hidden_states | ❓ 待验证 | 最大外部依赖,第 4.2 步先探测 |
| sdpa 稠密 mask 在 CANN 上的算子覆盖 | ❓ 待验证 | 失败则用 eager |
| muon on NPU | ❓ 待验证 | 失败回退 adamw |
| MTP 训练 | ❌ 不支持 | 强制 flex attention,需另行适配 |
| hidden states FP8 落盘(`feat/fp8-connector` 分支) | ❌ 不适用 | 910 系无原生 float8;有磁盘压力可做 int8 变体 |
| pin_memory/DataLoader | ✅ 应无碍 | torch 2.5+ 按当前 accelerator 处理 |
| 分布式 HCCL | ✅ 已支持 | `get_default_backend_for_device` 自动选 |
