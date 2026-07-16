# speculators 多机分片能力改造评估(ZeRO-2 / HSDP / TP)

> 更新:2026-07-16。目标:补齐 speculators 多机训练的分片策略,对齐 SpecForge。
> **本文是评估与改造方案,不含已实施代码**。落地前需确认。
> 关联:[[speculators-npu-dsv4-status]]、`glm5.2_adaptation.md`。

## 1. 问题

speculators 现在多机训练只有两个极端:

- **DDP**(默认,不加 `--fsdp-shard`):参数/梯度/优化器状态每卡各一份 → 32B draft 装不下;
- **ZeRO-3 全分片**(`--fsdp-shard`):全部按全局 rank 切 → 跨节点每层 all-gather,节点越多越慢(DeepSpec 实测 full_shard 2 节点 29s/it → 4 节点 180s/it)。

中间没有任何档位。SpecForge 多机快,正是因为它**默认不做全分片**,并提供了完整菜单。

## 2. 现状核对(代码事实)

- `src/speculators/train/distributed.py::apply_fully_sharded(model, param_dtype)`:对每个 decoder layer 和 root 各调一次 `fully_shard(..., mp_policy=mp_policy)`,**没有传 `reshard_after_forward`** → FSDP2 默认 `True` = ZeRO-3。
- 调用链:`scripts/train.py`(`--fsdp-shard` → `TrainerConfig.fsdp_shard`)→ `trainer.py::_setup_model_fsdp` → `apply_fully_sharded(self.model, param_dtype=hidden_states_dtype)`。
- 数据采样器 `dataloader.py`:`num_replicas=get_dp_size()`、`rank=get_dp_rank()`;`sp_size==1` 时 `dp_size==world_size`,即每个 rank 是独立数据并行 worker(自己一份 batch)。
- 运行环境:torch **2.12.1**,`fully_shard` 签名含 `mesh`、`reshard_after_forward`、`dp_mesh_dims`;`init_device_mesh` 可用。**FSDP2 的 ZeRO-2 与 HSDP 都能开,只是没暴露。**

## 3. SpecForge 对照(为什么它多机快)

`specforge/training/backend.py::ParallelConfig`:

- 默认 `sharding_strategy = "SHARD_GRAD_OP"`(**ZeRO-2**,不是 full_shard);
- 环境变量 `FSDP_SHARDING` 可切 `NO_SHARD`/`SHARD_GRAD_OP`/`FULL_SHARD`/`HYBRID_SHARD`;
- 还带 `tp_size`(张量并行)、`sp_ulysses_size` + `sp_ring_size`(序列并行)、`tp_device_mesh`。

即 DP × TP × SP × 分片档 四维可组合。快的核心:**默认 ZeRO-2 只切优化器状态+梯度(内存大头),参数不在反向重新 all-gather**,跨节点参数通信量比 ZeRO-3 少约 1/3;规模再大靠节点内 TP / HSDP,而不是把 ZeRO-3 铺到全局。

## 4. 三个改造选项

### 选项 A — 加 ZeRO-2 档(`reshard_after_forward=False`)

**改动量:S(约 10 行,3 文件 + 1 CLI 开关)。**

FSDP2 里 ZeRO-2 就是 `reshard_after_forward=False`:前向 all-gather 后**保留**聚合参数、反向不再重新 all-gather。语义与 SpecForge 默认档一致。

改造点(示意,未实施):

```python
# distributed.py
def apply_fully_sharded(model, param_dtype=torch.bfloat16,
                        reshard_after_forward: bool = True):
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype,
                                     reduce_dtype=torch.float32)
    for layer in model.layers:
        fully_shard(layer, mp_policy=mp_policy,
                    reshard_after_forward=reshard_after_forward)
    fully_shard(model, mp_policy=mp_policy,
                reshard_after_forward=reshard_after_forward)
    return model
```

`TrainerConfig` 加 `fsdp_reshard_after_forward: bool = True`;`_setup_model_fsdp` 透传;`train.py` 加 `--fsdp-zero2`(置 `False`)。

- **收益**:参数不重切,反向省一次全局 all-gather;优化器状态仍分片,32B draft 装得下。draft 只有 5 层,ZeRO-2 保留全部层聚合参数的显存代价很小(浅模型天然适配)。
- **不能解决什么**:前向 all-gather 仍走全局(跨节点),所以只是把 ZeRO-3 的跨机通信砍掉约 1/3,**不是跨节点问题的根治**。
- **风险**:低。纯参数透传,不动数据/checkpoint 路径。

### 选项 B — 加 HSDP(2D device mesh)★ 多机的真正解

**改动量:S–M。核心是构造 2D mesh 并透传,数据采样器无需改。**

HSDP = 节点内分片 + 节点间复制。FSDP2 通过给 `fully_shard(mesh=...)` 传一个 `(replicate, shard)` 的 2D mesh 实现:参数只在**节点内** all-gather(走 HCCS/HCCL 高速),节点间只做一次梯度 all-reduce。这正是 DeepSpec 用来避免 29→180s/it 退化的做法。

改造点(示意,未实施):

```python
# distributed.py —— 分布式初始化时建 2D mesh
from torch.distributed.device_mesh import init_device_mesh
# 设备类型按 accelerator 取(npu / cuda);shard 维 = 每节点卡数
mesh_2d = init_device_mesh(
    device_type, (num_nodes, gpus_per_node),
    mesh_dim_names=("replicate", "shard"),
)

# apply_fully_sharded 增加 mesh 形参,传给每个 fully_shard(mesh=mesh_2d, ...)
```

**关键:数据采样器不用改。** HSDP 下每个 rank 仍处理不同 micro-batch,全局数据并行宽度 = world_size,`get_dp_size()==world_size` 依旧正确(FSDP2 自动在 shard 维 reduce-scatter、在 replicate 维 all-reduce)。

- **收益**:参数 all-gather 只在节点内 → 跨节点吞吐随节点数近似线性,不再退化。这是 32B/更大 draft 多机训练的正解。
- **风险**:中。需验证:①`num_nodes`/`gpus_per_node` 从现有 `RANK/WORLD_SIZE/LOCAL_RANK` 正确推导;②DCP checkpoint 保存/加载在 2D mesh 下正确(FSDP2 通常原生支持,需实测);③与 #711 的 root `mp_policy`、DDP 分支不冲突。
- **可与 A 叠加**:`reshard_after_forward=False` + 2D mesh = HSDP-ZeRO2(节点内 ZeRO-2)。

### 选项 C — draft 张量并行(TP)

**改动量:L。暂不建议,除非 draft 本身巨大。**

SpecForge 有(`tp_size`/`tp_device_mesh`),speculators 完全没有。要切 draft 的 attention/MLP/fc/lm_head 并插入 all-reduce,还要和词表映射、markov/confidence 头对齐。draft 只有 5 层、参数量远小于 target,TP 的收益/工作量比很差。**留到 draft 达到数十 B 级别(远期)再评估。** 长序列另有 SP(见 `origin/sp_ulysses` 分支,主线未合)。

## 5. 推荐与排序

| 优先级 | 选项 | 改动量 | 解决什么 |
|---|---|---|---|
| 1 | **B. HSDP** | S–M | 多机 32B 训练的**真正**加速,跨机不退化 |
| 2 | A. ZeRO-2 | S | 内存装得下 + 少 1/3 跨机通信;可与 B 叠加 |
| — | C. TP | L | 远期,draft 巨大化才需要 |

**给用户当前 32B 的落地建议**:

- **单机 8 卡**:现有 `--fsdp-shard`(ZeRO-3)已够,先跑(见前述 OOM 建议)。
- **要多机训练**:上 **HSDP(选项 B)** 才划算;顺带把 ZeRO-2(选项 A)作为一个正交开关加上。两者 FSDP2 都原生支持,改动集中在 `distributed.py` + `trainer.py` + `train.py` 三处,不碰模型代码。

## 6. 验证方法(实施后)

1. 单机 2 卡等价性:ZeRO-3 vs ZeRO-2 vs HSDP(2×1)训同一小数据,loss 曲线一致;
2. 显存:HSDP 每卡 ≈ ZeRO-3 每卡(节点内分片深度相同);ZeRO-2 每卡略高(参数常驻);
3. 吞吐:2 节点 vs 4 节点 s/it,HSDP 近似持平、全局 ZeRO-3 明显劣化——对比确认 HSDP 生效;
4. checkpoint:HSDP 存的 checkpoint 能被单机加载、能喂 vLLM;
5. `scripts/check_norm_canary.py` 确认新路径下 norm 权重正常更新(不被 bf16 舍入冻结)。

## 7. 声明

以上均为**方案评估**,当前分支未改任何分布式代码。需你确认后再按选项落地(建议先 B,叠 A)。C 暂缓。
