# 绕开 vLLM 生成 hidden states 喂 speculators 训练(两种方法)

> 更新:2026-07-17。目的:规避 vllm-ascend 0.22.1rc1 extract_hidden_states 的两个坑
> (prefix caching 命中前缀被跳过 → hidden 长度短一截 1244/2048;chunked prefill 强制关 →
> 长 prompt 单发 NaN/hang),改用**干净的 HF 全量前向**生成数据,再用 speculators 的
> `--legacy-data` 通道训练 DSpark。关联:[[speculators-npu-dsv4-status]]。
>
> 两种方法,按需选:
> - **方法 A(推荐,尤其大数据量):用 SpecForge 生成 + speculators 加载时适配**。
>   SpecForge 生成**原生多节点**;几十 TB 数据靠"加载时转换"避免批量重写。
> - **方法 B:自己写独立 HF 前向脚本直出 v1 格式**。依赖最少,但多节点/32B target
>   切分要自己补,当前模板是单进程。

---

## 0. 为什么 HF 全量前向能绕开

| | vLLM extract(现在坏的) | HF 全量前向(A/B 都基于它) |
|---|---|---|
| 机制 | 挂推理引擎前向 + KV connector 抓激活 | `@torch.no_grad()` 直接过整条序列 |
| prefix caching | 默认开 → 命中前缀不重算 → 抽不到 → 长度短 | 无缓存,每位置都真实前向 → 永远齐全 |
| chunked prefill | 强制关 → 长 prompt 单发 → NaN/hang | 无 serving 分块,不涉及 |
| 依赖 | vLLM + vllm-ascend extract 适配(坏) | 只要 torch_npu 能跑 HF forward |

SpecForge 的 `prepare_hidden_states.py` 就是 HF backend + `no_grad` 全量前向(见其 line 2、424)。

## 1. speculators 能读的 v1 格式(权威,来自 `train/data.py::standardize_data_v1`)

`--legacy-data` 走 `SampleFileDataset`,读 `.pt`,每样本一个 dict,**只需三个 key**:

```python
{
    "input_ids":  LongTensor[seq_len],
    "loss_mask":  BoolTensor[seq_len],
    "hidden_states": [                    # list of tensor
        Tensor[seq_len, hidden_size],     #  aux 层 0   ┐
        ...                               #  aux 层 k-1 ├ [:-1] → cat 成 fc 输入
        Tensor[seq_len, hidden_size],     #  最后一层   ┘ [-1] verifier_last_hidden
    ],
}
```

`lengths`/`position_ids` 由 `__getitem__` 自动补,不用你存。
**唯一易错点:list 里 aux 层要正好是训练 `--target-layer-ids` 的那几层、同顺序;最后一个是 verifier 末层。**

---

# 方法 A:SpecForge 生成(多节点)+ speculators 加载时适配

## A.1 为什么选它

- SpecForge `prepare_hidden_states.py` **原生支持多节点**(每节点扫 1/N 样本),你们团队已有流程;
- DSpark 训练在 speculators(SpecForge 无 dspark),所以生成用 SpecForge、训练用 speculators。

## A.2 格式差异(这是唯一的增量)

| | SpecForge DataPoint(torch.save asdict) | speculators v1 |
|---|---|---|
| token | `input_ids` | `input_ids` ✅ |
| mask | `loss_mask` | `loss_mask` ✅ |
| 末层 | `hidden_state` | `hidden_states[-1]` |
| aux | `aux_hidden_state` | `hidden_states[:-1]` |

key 名/结构不同,speculators `--legacy-data` **不能直接读** SpecForge 的原始 `.pt`,需要桥接。

## A.3 桥接怎么做——几十 TB 的关键抉择

**❌ A.3a 批量转换(不推荐,几十 TB 尤其不行)**
逐文件读 SpecForge `.pt` → 重排 key → 写新 `.pt`。问题:hidden states 本身就是那几十 TB,
批量转换 = **全量读一遍 + 全量写一遍**,I/O 翻倍、存储要双份(或转一个删一个但丢原始)。
只在"死活不改 speculators 代码"时才用。转换核心逻辑:

```python
# 每个文件:sf = torch.load(src)
data = {
    "input_ids": sf["input_ids"],
    "loss_mask": sf["loss_mask"],
    # aux_hidden_state 是 stack [k,seq,h] 还是 concat [seq,k*h] 要看实际 shape,
    # 目标是拆成 k 个 [seq,h] 再接上末层:
    "hidden_states": list(sf["aux_hidden_state"].unbind(0)) + [sf["hidden_state"]],
}
torch.save(data, dst)
```
(多节点批量转换可照 tige 脚本的 `--world-size/--rank` 分片并行,转完删原文件保持存储平。)

**✅ A.3b 加载时适配(推荐)—— 不重写任何 TB**
不动那几十 TB,在训练读每个样本时于内存重排 key。读这一遍训练本来就要做,
转换每样本几微秒可忽略,**零额外存储、零额外 I/O**。

代价:`standardize_data_v1` 现在写死在 `_get_raw_data`,`SampleFileDataset` 不接受自定义
standardize 函数。要让 speculators 直接读 SpecForge 格式,需一个 **~15 行的小改动**:

- 给 `SampleFileDataset.__init__` 加 `standardize_fn` 形参,`_get_raw_data` 用它而非写死;
- 加一个 `standardize_data_specforge(sf) -> v1` 函数(逻辑同上面的重排);
- `dataloader.create_train_val_loaders` 按新 CLI(如 `--legacy-data-format specforge`)选函数。

**这是改 speculators 代码,需你批准后再做。** 改完后:SpecForge 多节点生成 → 直接
`--legacy-data --legacy-data-format specforge --data-path <SpecForge输出>` 训练,全程不碰 TB。

## A.4 流程

1. SpecForge `scripts/prepare_hidden_states.py`(多节点,HF 前向,不启 vLLM)生成 `.pt`;
2. 桥接:选 A.3b(改 ~15 行,推荐)或 A.3a(批量转换,大数据量不推荐);
3. speculators `--legacy-data` 训练 DSpark(见下方训练命令)。

---

# 方法 B:独立 HF 前向脚本直出 v1 格式

## B.1 适用场景

不想引入 SpecForge、数据量不大(或 target 单卡放得下,如 Qwen3-8B)时最省事——
直接输出 v1 格式,**连桥接都省了**。

## B.2 生成脚本(模板,单进程,放 `scripts/` 或任意位置)

前置:先跑 speculators `scripts/prepare_data.py` 得到 input_ids + loss_mask,本脚本只加 hidden_states。

```python
# gen_hidden_hf.py — HF 全量前向直出 v1 格式(NPU,单进程模板)
import argparse
from pathlib import Path
import torch, torch_npu  # noqa: F401
from transformers import AutoModelForCausalLM

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prepared-data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--target-layer-ids", type=int, nargs="+", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    device, dtype = "npu:0", getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, output_hidden_states=True,
    ).to(device).eval()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    samples = load_prepared_samples(args.prepared_data)  # 逐条产出 input_ids + loss_mask
    for idx, s in enumerate(samples):
        input_ids = torch.tensor(s["input_ids"], dtype=torch.long, device=device)[None]
        with torch.no_grad():
            hs = model(input_ids).hidden_states   # tuple len=num_layers+1, hs[0]=embedding
        aux = [hs[L][0].to(dtype).cpu() for L in args.target_layer_ids]
        last = hs[-1][0].to(dtype).cpu()
        data = {
            "input_ids": input_ids[0].cpu(),
            "loss_mask": torch.tensor(s["loss_mask"], dtype=torch.bool),
            "hidden_states": aux + [last],
        }
        if any(torch.isnan(t).any() for t in data["hidden_states"]):
            print(f"skip {idx}: NaN"); continue
        torch.save(data, out / f"data_{idx}.pt")

if __name__ == "__main__":
    main()
```

## B.3 ⚠️ 这个模板没做的两件事(诚实说明)

1. **多节点:没有**。模板是单进程单卡循环。要多节点,得自己加 `RANK/WORLD_SIZE` 切片
   (`samples[rank::world]`)+ marker 同步,和 tige 脚本套路一样,十几行——**模板里没写**。
2. **32B target 单卡装不下:没解决**。HF 全量前向要每卡装下整个 32B(bf16 ~65GB > 64GB)。
   模板只在这靠 `device_map="auto"`(未验证)。8B 没问题;**32B 直接用会卡在这一步**,
   要么 device_map 切层(慢)、要么自己做 target 的 TP/FSDP-推理。

→ **所以方法 B 适合 8B / 小数据量;32B + 大数据量优先方法 A(SpecForge 多节点)。**

---

## 训练命令(A / B 通用,speculators 侧不改)

```bash
torchrun --standalone --nproc_per_node <N> scripts/train.py \
    --legacy-data \
    --data-path <生成/桥接后的目录> \
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
    --fsdp-shard   # 32B 必须;多机再加 --fsdp-shard-size <每节点卡数>(HSDP)
# 方法 A 走加载时适配(A.3b)时,额外加 --legacy-data-format specforge(该开关待实现)
```

## 上量前必做的校验(层口径,A/B 都要)

生成时 `hs[L]` / SpecForge aux 层的层号,必须和训练 `--target-layer-ids` 口径一致,否则抽错层。
两种确认:①对一条 vLLM extract 成功的样本 `torch.allclose` 比对;②对齐 `launch_vllm.py` 的
`eagle_aux_hidden_state_layer_ids` + `--include-last-layer` 约定。先核对 1~2 条 shape/层号,
再生成 500 条 `--legacy-data` 训 1 epoch 冒烟,loss 正常下降、EAL 与参考同量级,再放量。

## 注意事项

- **`--legacy-data` 在 speculators 标了 DEPRECATED**("will be removed soon"):pin 住版本;
- **dtype** bf16;**NaN** 样本跳过(A/B 都做);
- 方法 A 的加载时适配(A.3b)和 `--legacy-data-format specforge` 开关**尚未实现**,需批准后加。

## 一句话

- **大数据量 / 32B → 方法 A**:SpecForge 多节点生成 + speculators 加载时适配(~15 行改动,
  避免几十 TB 批量重写);
- **8B / 小数据量 → 方法 B**:独立 HF 脚本直出 v1,最少依赖(但多节点/32B 切分要自己补)。
