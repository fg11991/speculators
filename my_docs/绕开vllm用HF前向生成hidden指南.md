# 绕开 vLLM:用 SpecForge DFlash 全量前向生成 hidden 喂 speculators 训练

> 更新:2026-07-17。规避 vllm-ascend 0.22.1rc1 extract_hidden_states 的坑(prefix caching
> 命中前缀被跳过 → hidden 短一截;chunked prefill 强制关 → 长 prompt NaN/hang),改用
> **SpecForge 本地 fork 的 DFlash 全量前向生成**(多节点 + TP),再用 speculators 的
> `--legacy-data --legacy-data-format specforge` 直接训练 DSpark。关联:[[speculators-npu-dsv4-status]]。
>
> 基于用户本地 `/Users/wuyidong/projects/github_projects/job/specforge/specforge`(NPU 适配 fork)。

## 0. 为什么这样能绕开

SpecForge DFlash 的 `prepare_hidden_states.py` 用 HF transformers 做 `@torch.no_grad()` 全量前向,
teacher forcing 过整条序列——**每个位置都真实前向**,没有 prefix caching 跳过、没有 chunked
prefill 分块,所以 vLLM extract 的两个坑都不存在。而且它**原生多节点(DP)+ 张量并行(TP)**,
32B target 靠 `--tp-size` 摊到多卡。

## 1. 生成(SpecForge 侧,DFlash 模式)

```bash
torchrun --nproc_per_node=8 \
    scripts/prepare_hidden_states.py \
    --model-type dflash \
    --target-model-backend hf \          # DSpark 必须 hf,才会产 last_hidden_state
    --target-model-path Qwen/Qwen3-8B \
    --draft-config-path <draft config.json，含 dflash_config.target_layer_ids> \
    --data-path <sharegpt.jsonl> \
    --output-path <共享盘目录> \
    --max-length 2048 \
    --num-draft-layers 5 \
    --tp-size 1                           # 32B 时设 2/4,把 target 摊到多卡
    # 不要加 --compress(见 §3)
```

要点:
- **`--target-model-backend hf` 必须**——只有 hf 后端会输出 `last_hidden_state`(verifier 末层),
  DSpark 的 TV loss 需要它;sglang 后端 `last_hidden_states=None`,不能用;
- **层号**:用 `--draft-config-path` 指定 `dflash_config.target_layer_ids`,或 `--num-draft-layers`
  自动算。脚本会打印 `[IMPORTANT] Captured target layer IDs: [...]`,**训练必须用同一组 layer ids**;
- 多节点:`torchrun` 的 DP 各节点各扫 1/N 样本(脚本内置 `start_idx`/`samples_per_dp` 切分);
- 32B target 单卡放不下 → `--tp-size 2`(或 4)。

## 2. SpecForge DFlash 存的格式(权威,来自本地 fork)

`prepare_hidden_states.py` DataPoint(存成 `rows_X-Y/data_i.ckpt`):

| key | shape | 含义 | → speculators |
|---|---|---|---|
| `input_ids` | [seq] | token | `input_ids` |
| `loss_mask` | [seq] | 掩码 | `loss_mask` |
| `hidden_state` | [1, seq, **K*h**] | K 个 capture 层拼接(`torch.cat dim=-1`) | `hidden_states`(fc 输入) |
| `last_hidden_state` | [1, seq, h] | verifier 末层(`outputs.hidden_states[-1]`) | `verifier_last_hidden_states` |
| `aux_hidden_state` | None | dflash 不用 | — |

**两样分开给了,K*h 已拼好,不用切分。** 适配器只 squeeze batch 维 + 换 key 名(见 `standardize_data_specforge`)。

## 3. 训练(speculators 侧,已实现)

```bash
torchrun --standalone --nproc_per_node <N> scripts/train.py \
    --legacy-data --legacy-data-format specforge \
    --data-path <SpecForge --output-path 目录> \
    --verifier-name-or-path Qwen/Qwen3-8B \
    --speculator-type dspark \
    --num-layers 5 --block-size 8 \
    --target-layer-ids <和生成时打印的完全一致> \
    --markov-rank 256 --markov-head-type vanilla \
    --enable-confidence-head --confidence-head-with-markov \
    --loss-fn '{"ce": 0.1, "tv": 0.9}' \
    --draft-attn-impl sdpa --total-seq-len 2048 \
    --fsdp-shard   # 32B 必须;多机加 --fsdp-shard-size <每节点卡数>(HSDP)
```

**文件兼容(已在 speculators 侧处理)**:
- `list_files` 收 `.pt` / `.ckpt` / `.ckpt.gz`,`os.walk` 递归 `rows_X-Y/` 子目录 → 直接能发现;
- **`--compress` 现已支持**:`.ckpt.gz` 会 `gzip.open` 解压到内存再 `torch.load`(照搬 SpecForge
  `data/preprocessing.py` 的读法)。权衡:压缩省存储,但读时每样本要解压(吃 CPU,靠 `--num-workers`
  并行摊掉),且 gz 文件失去 mmap、multipack 的长度估计按压缩后大小会略糙(只影响打包效率,不影响
  正确性)。bf16 hidden 是高熵浮点,gzip 压缩比有限,值不值得先拿一批实测压缩率再定。

## 4. 上量前必跑一次校验

```bash
python scripts/check_specforge_hidden.py <一条 data_i.ckpt> \
    --verifier-name-or-path Qwen/Qwen3-8B --num-target-layers 3
```
两件事:①`hidden_state` 宽度 = K*hidden、key/shape 对齐;②用 verifier 自己的
`lm_head(norm(last_hidden_state))` 重建 logits,量 teacher-forcing 下一 token 命中率——
命中率正常说明 `last_hidden_state` 是 **pre-norm**(speculators 需要的);命中率接近 0 说明
存了 post-norm/不匹配,**别放量**,回来处理(让 SpecForge 存 pre-norm、或给 specforge 格式跳过
verifier_norm)。这是源码层面唯一无法证明、必须实测的一点。

## 5. 已实现的 speculators 增量(npu-support 分支)

- `data.py::standardize_data_specforge`:DFlash `hidden_state`→`hidden_states`、
  `last_hidden_state`→`verifier_last_hidden_states`,squeeze batch 维;
- `data.py::list_files`:收 `.ckpt`(除 `.pt`);
- `SampleFileDataset` 可插拔 `standardize_fn`;`LEGACY_STANDARDIZE_FNS = {v1, specforge}`;
- CLI `--legacy-data --legacy-data-format specforge`;
- 单测 `tests/unit/train/test_specforge_adapter.py`(含 v1↔specforge 同源逐位相等);
- 校验脚本 `scripts/check_specforge_hidden.py`。

**两框架核心逻辑都不改**(SpecForge 本就有 DFlash 生成),speculators 只加了适配器+文件发现。

## 6. 一句话

SpecForge DFlash 全量前向(多节点+TP,层号 config 指定)→ `data_i.ckpt` → speculators
`--legacy-data-format specforge` 直接训 DSpark。层号两边一致、生成不压缩、上量前跑一次
norm 校验,就通了。
