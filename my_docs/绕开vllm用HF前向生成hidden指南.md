# 绕开 vLLM:用 HF 全量前向生成 hidden states 喂 speculators 训练

> 更新:2026-07-17。目的:规避 vllm-ascend 0.22.1rc1 extract_hidden_states 的两个坑
> (prefix caching 命中前缀被跳过 → hidden 长度短一截 1244/2048;chunked prefill 强制关 →
> 长 prompt 单发 NaN/hang),改用与 SpecForge 同源的**干净全量前向**生成数据,再用
> speculators 的 `--legacy-data` 通道训练 DSpark。关联:[[speculators-npu-dsv4-status]]。

## 0. 为什么这样能绕开

| | vLLM extract(现在坏的) | HF 全量前向(本方案) |
|---|---|---|
| 机制 | 挂推理引擎前向 + KV connector 抓激活 | `@torch.no_grad()` 直接过整条序列 |
| prefix caching | 默认开 → 命中前缀不重算 → 抽不到 → 长度短 | 无缓存,每个位置都真实前向 → 永远齐全 |
| chunked prefill | 强制关 → 长 prompt 单发 → NaN/hang | 无 serving 分块,不涉及 |
| 依赖 | vLLM + vllm-ascend extract 适配 | 只要 torch_npu 能跑 HF forward |

SpecForge 的 `prepare_hidden_states.py` 就是这么干的(HF transformers backend,`no_grad`)。
本方案取其精简版,**直出 speculators v1 格式**,省掉字段转换,也不引入 SpecForge 依赖。

## 1. speculators 能直接读的 v1 格式(权威,来自 `train/data.py::standardize_data_v1`)

每个样本一个 `.pt`,`torch.save` 一个 dict,**只需三个 key**:

```python
{
    "input_ids":  LongTensor[seq_len],          # 与 prepare_data 的打包序列一致
    "loss_mask":  BoolTensor[seq_len],          # 同上
    "hidden_states": [                          # 一个 list of tensor
        Tensor[seq_len, hidden_size],           #  aux 层 0   ┐
        Tensor[seq_len, hidden_size],           #  aux 层 1   ├ [:-1] → cat 成 fc 输入
        ...                                     #  aux 层 k-1 ┘
        Tensor[seq_len, hidden_size],           #  最后一层  → [-1] verifier_last_hidden
    ],
}
```

- `hidden_states[:-1]` = k 个 aux 层(k = `--num-layers` 对应的 `--target-layer-ids` 个数),
  会被 `torch.cat(..., dim=-1)` 拼成 `[seq_len, k*hidden_size]` 作为 draft 的 fc 输入;
- `hidden_states[-1]` = verifier 最后一层,单独给 verifier_lm_head 算 target logits;
- `lengths` / `position_ids` **不用你存**,`SampleFileDataset.__getitem__` 会自动补。

> ⚠️ **顺序和层对齐是唯一容易错的点**:list 里的 aux 层必须正好是 DSpark 训练时 `--target-layer-ids`
> 指定的那几层、**同样的顺序**;最后一个元素必须是 verifier 的最后一层。

## 2. 生成脚本(模板,放 `scripts/` 或任意位置,不改 speculators 本体)

前置:先跑 speculators 的 `scripts/prepare_data.py` 得到打包好的数据(含 input_ids + loss_mask),
本脚本只负责**加上 hidden_states**。

```python
# gen_hidden_hf.py — HF 全量前向生成 v1 格式 hidden states(NPU)
import argparse, json
from pathlib import Path
import torch, torch_npu  # noqa: F401  (import 即注册 npu 后端)
from transformers import AutoModelForCausalLM

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prepared-data", required=True,
                    help="prepare_data.py 的输出目录(提供 input_ids + loss_mask)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--target-layer-ids", type=int, nargs="+", required=True,
                    help="与训练 --target-layer-ids 完全一致,同顺序")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    device = "npu:0"
    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, output_hidden_states=True,
    ).to(device).eval()

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    # 读取 prepare_data 的样本(input_ids + loss_mask)。按你 prepared-data 的实际
    # 存储改这一段——目标是拿到每条样本的 input_ids(list[int]) 和 loss_mask(list[int/bool])。
    samples = load_prepared_samples(args.prepared_data)  # 见下方说明

    for idx, s in enumerate(samples):
        input_ids = torch.tensor(s["input_ids"], dtype=torch.long, device=device)[None]
        with torch.no_grad():
            hs = model(input_ids).hidden_states   # tuple, 长度 num_layers+1;hs[0]=embedding
        # HF 约定:hs[L] = 第 L 层的输出(hs[0] 是 embedding 输出)。
        # target_layer_ids 的取值口径必须和 vLLM/DSpark 一致 —— 见 §4 校验。
        aux = [hs[L][0].to(dtype).cpu() for L in args.target_layer_ids]
        last = hs[-1][0].to(dtype).cpu()
        data = {
            "input_ids": input_ids[0].cpu(),
            "loss_mask": torch.tensor(s["loss_mask"], dtype=torch.bool),
            "hidden_states": aux + [last],   # [:-1]=aux, [-1]=verifier last
        }
        # NaN 保护(和 SpecForge 一样,坏样本跳过而不是写脏数据)
        if any(torch.isnan(t).any() for t in data["hidden_states"]):
            print(f"skip {idx}: NaN"); continue
        torch.save(data, out / f"data_{idx}.pt")

if __name__ == "__main__":
    main()
```

`load_prepared_samples` 按你 `prepare_data` 的输出格式实现(Arrow/parquet 或 jsonl);
只要能逐条产出 `{"input_ids": [...], "loss_mask": [...]}` 即可。如果不想读它的格式,
也可以直接从原始对话数据 + tokenizer 自己打包(和 prepare_data 用同一套 chat template
和 seq_length,保证 input_ids 一致)。

## 3. 训练(speculators 侧,不改代码)

```bash
torchrun --standalone --nproc_per_node <N> scripts/train.py \
    --legacy-data \
    --data-path <上一步 --output 目录> \
    --verifier-name-or-path Qwen/Qwen3-32B \
    --save-path ./output/dspark/checkpoints \
    --speculator-type dspark \
    --num-layers 5 --block-size 8 \
    --target-layer-ids <和生成时完全一致> \
    --markov-rank 256 --markov-head-type vanilla \
    --enable-confidence-head --confidence-head-with-markov \
    --loss-fn '{"ce": 0.1, "tv": 0.9}' \
    --draft-attn-impl sdpa \
    --total-seq-len 2048 \
    --fsdp-shard   # 32B 必须;多机再加 --fsdp-shard-size <每节点卡数>
```

`--legacy-data` → 走 `SampleFileDataset` 读 `.pt`,完全不碰 vLLM。

## 4. 上量前必做的校验(层口径,决定成败)

生成脚本里 `hs[L]` 的 L 用什么口径,必须和 DSpark 训练时 `--target-layer-ids` 的口径一致,
否则抽的是错层、训出来接受率上不去。两种确认方式任选:

1. **对已知好样本**:如果你手上有一条 vLLM extract 成功(长度正确)的样本,拿同一条
   input_ids 用本脚本生成,`torch.allclose` 比对 aux/last 层——完全对上说明口径正确;
2. **看 vLLM 口径**:`launch_vllm.py` 里 `eagle_aux_hidden_state_layer_ids = target_layer_ids`
   且 `--include-last-layer` 会把 `num_hidden_layers` 追加进去。HF 的 `output_hidden_states`
   中 `hs[i]` 是第 i 层输出(hs[0]=embedding)。先拿 1~2 条核对 shape 和层号,别直接放全量。

冒烟:先生成 500 条、`--legacy-data` 训 1 epoch,确认 loss 正常下降、EAL 与参考同量级,再放量。

## 5. 注意事项

- **`--legacy-data` 在 speculators 标了 DEPRECATED**("will be removed soon"):现在能用,
  pin 住你 npu-support 的 speculators 版本,别跟 main 升到它被删;
- **dtype**:hidden_states 存 bf16(与训练 `--hidden-states-dtype` 一致);
- **32B target 的显存**:HF 全量前向要每卡装下整个 32B(bf16 ~65GB > 64GB 单卡)。
  两个办法:①用 `device_map="auto"` 把 target 切到多卡做前向(生成阶段,纯推理);
  ②或分节点数据并行(每节点各生成一部分样本,和 tige 脚本的生成分片同理)。
  这一步和训练分开,先生成落盘、再训练;
- **速度**:HF 全量前向比 vLLM serving 慢,但正确且简单;GLM 实践也表明瓶颈在特征 I/O
  而非前向本身。可多卡/多节点并行生成来补;
- **NaN**:脚本已按 SpecForge 做法跳过 NaN 样本;若跳过比例高,查 target 权重/精度,
  而不是继续往下训。

## 6. 一句话

生成换成 HF 全量前向(SpecForge 同源),直出 speculators v1 格式,训练用 `--legacy-data` ——
**两边框架核心都不改**,只新增一个生成脚本,彻底甩开 vllm-ascend 的 extract 适配坑。
唯一要盯的是 §4 的层口径校验。
