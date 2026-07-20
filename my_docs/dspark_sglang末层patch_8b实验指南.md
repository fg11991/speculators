# DSpark 用 SpecForge(sglang)生成 hidden → speculators 训练:8B 端到端实验指南

> 2026-07-20。目标:在 Qwen3-8B 上跑通 **SpecForge sglang 生成 hidden(含 pre-norm 末层)→ speculators 训练 DSpark** 全链,并**验证我们对 sglang / SpecForge 的改动是正确的**,再放量 32B。
> 关联:[[speculators-npu-dsv4-status]]。

---

## 0. 背景:我们改了什么、为什么

DSpark 的 draft 用 `verifier_lm_head(verifier_norm(verifier_last_hidden_states))` 重建 verifier logits 算 TV loss。所以 `verifier_last_hidden_states` 必须是**最后一层 decoder 的 pre-norm 输出**(最终 RMSNorm 之前),这和 HF 后端 `outputs.hidden_states[-1]` 是同一个张量。

但 SpecForge 的 **sglang 后端原本不产末层**(`last_hidden_states=None`):sglang 的 `Qwen2Model.forward` 一到最后就 `self.norm(hidden_states, residual)`,pre-norm 残差流被吃掉;aux(eagle3)capture 又够不到最后一层。为此我们做了**两处最小改动**:

| 改动 | 文件 | 作用 |
|---|---|---|
| **sglang patch** | `scripts/patch_sglang_prenorm.py`(SpecForge 新增) | 在 `Qwen2Model.forward` 最终 norm 前,把 `hidden_states + residual`(pre-norm 末层)存到 `self.captured_last_hidden`。Qwen3 继承此 forward,自动覆盖 |
| **sglang 后端读末层** | `specforge/modeling/target/dflash_target_model.py` | `generate_dflash_data` 读 `captured_last_hidden`,填进 `last_hidden_state` |

**speculators 一行没改**:它的 `standardize_data_specforge` 本就读 `last_hidden_state`,现在 sglang 真的产出这个 key(pre-norm),口径自然对齐。

---

## 1. 前置

- 镜像:内含 **sglang 0.5.9**(生成)+ speculators 训练环境。SpecForge 与 speculators 两个仓库都同步到位。
- 8B 冒烟:sglang 后端 `--tp-size 1` 单卡即可(Qwen3-8B ~16GB)。
- 约定:所有提交用 `w00958190 <wuyidong5@huawei.com>`。

---

## 2. 步骤一:打 sglang patch(容器启动脚本里,必须在拉起 sglang 之前)

patch 改的是**磁盘上的 sglang 源文件**,生成进程再 import 才生效,所以顺序是「先 patch,后生成」。写进容器 entrypoint / 启动脚本:

```bash
# 幂等:每次起容器都可跑,已打过只打印 "already patched"
python /path/to/specforge/scripts/patch_sglang_prenorm.py
```

**如何确认打上了**:

```bash
# ① 脚本自己会报:
#    OK: patched pre-norm final-hidden capture into .../sglang/srt/models/qwen2.py
#    或 OK: already patched, nothing to do -> ...

# ② 直接查文件里有没有注入的那句:
python - <<'PY'
import importlib.util, os
root = importlib.util.find_spec("sglang").submodule_search_locations[0]
p = os.path.join(root, "srt", "models", "qwen2.py")
src = open(p).read()
print("PATCHED" if "self.captured_last_hidden" in src else "NOT PATCHED", "->", p)
PY
```

> ⚠️ 如果脚本报 `ERROR: could not find the expected final-norm block`,说明镜像里的 sglang **不是 0.5.9**(锚点变了)。**不要硬跑**——它会停住保护你。把实际 sglang 版本告诉我,我更新锚点。

---

## 3. 步骤二:生成 hidden(SpecForge,sglang 后端,8B)

先取一小撮(如 64 条)冒烟。**注意:8B 冒烟也要用 sglang 后端**,因为我们要验证的正是这条 sglang 新路径。

```bash
cd /path/to/specforge

# 先打 patch(见步骤一),再生成
python scripts/patch_sglang_prenorm.py

torchrun --nproc_per_node=1 scripts/prepare_hidden_states.py \
    --model-type dflash \
    --target-model-backend sglang \      # ← 走我们改过的 sglang 路径
    --tp-size 1 \                        # 8B 单卡;32B 时设 8
    --target-model-path Qwen/Qwen3-8B \
    --num-draft-layers 5 \
    --chat-template qwen3_no_system_prompt \
    --data-path <小 sharegpt.jsonl> \
    --output-path <冒烟输出目录，如 ./cache/dspark_8b_smoke> \
    --max-length 2048 \
    --num-samples 64
    # 冒烟先不加 --compress，跑通再考虑 .ckpt.gz
```

生成时**记下脚本打印的这一行**,训练要用同一组层号:
```
[IMPORTANT] Captured target layer IDs: [.....]
```

产物:`<output-path>/rows_X-Y/data_i.ckpt`,每个含 `input_ids[seq] / loss_mask[seq] / hidden_state[1,seq,5*4096] / last_hidden_state[1,seq,4096] / aux_hidden_state=None`。

---

## 4. 步骤三:验证改动正确(放量前必做,分两级)

### 4.1 一级:结构 + norm 口径(必做)

```bash
cd /path/to/speculators
python scripts/check_specforge_hidden.py <一条 data_i.ckpt> \
    --verifier-name-or-path Qwen/Qwen3-8B \
    --num-target-layers 5
```
判读:
- `[1] shape/key`:`fc_width` 应 = **5 × 4096 = 20480**,且 `last_hidden_state` 存在、shape `[seq,4096]`;
- `[2] norm convention`:用 verifier 自己的 `lm_head(norm(last))` 重建 logits、量下一 token 命中率。
  - **命中率高(自然文本通常 >0.5)= last 是 pre-norm、全宽、层号对** → **通过**;
  - **接近 0 = 末层口径错了(post-norm / TP 分片 / 层号错)** → **别放量**,回来查(多半是 patch 没生效或 sglang 版本不对)。

这一步同时验掉了 **TP 下 `hidden_states+residual` 是不是全宽 hidden** 这个源码证不了、只能实测的点。

### 4.2 二级:与 HF 后端交叉比对(金标准,强烈建议冒烟阶段做一次)

HF 后端天然产 pre-norm 末层(`outputs.hidden_states[-1]`)、且不 patch。对**同一条样本**分别用 hf 和 sglang-patched 生成,比对 `last_hidden_state`——数值接近 = patch 抓的确实是 pre-norm 末层,铁证。

```bash
# 同一批数据、同一模型,分别生成到两个目录
# (A) HF 后端(不 patch,天然 pre-norm):
torchrun --nproc_per_node=1 scripts/prepare_hidden_states.py \
    --model-type dflash --target-model-backend hf \
    --target-model-path Qwen/Qwen3-8B --num-draft-layers 5 \
    --chat-template qwen3_no_system_prompt \
    --data-path <同一 jsonl> --output-path ./cache/cmp_hf --max-length 2048 --num-samples 4

# (B) sglang-patched(见步骤二,output-path ./cache/cmp_sglang)

# 比对同一条样本的 last_hidden_state
python - <<'PY'
import torch, glob, gzip, io
def load(p):
    f = p.endswith(".gz")
    d = torch.load(io.BytesIO(gzip.open(p,'rb').read()) if f else p, weights_only=False)
    return d
a = load(sorted(glob.glob("./cache/cmp_hf/**/data_0.ckpt*", recursive=True))[0])
b = load(sorted(glob.glob("./cache/cmp_sglang/**/data_0.ckpt*", recursive=True))[0])
la, lb = a["last_hidden_state"].float(), b["last_hidden_state"].float()
print("shapes:", tuple(la.shape), tuple(lb.shape))
print("max abs diff:", (la-lb).abs().max().item())
print("cosine:", torch.nn.functional.cosine_similarity(la.flatten(1), lb.flatten(1)).mean().item())
PY
```
预期:shape 一致,**cosine ≈ 1.0**,max abs diff 很小(bf16 + 后端算子差异,量级 ~1e-2 内可接受)。若 cosine 明显偏离 1,说明 sglang patch 抓错了张量,别放量。

---

## 5. 步骤四:speculators 训练 DSpark(8B)

### 5.1 冒烟(几十条,1 epoch,看 loss 动不动)

```bash
cd /path/to/speculators
torchrun --standalone --nproc_per_node 1 scripts/train.py \
    --legacy-data --legacy-data-format specforge \
    --data-path <步骤二的 output-path，含 rows_X-Y/*.ckpt> \
    --verifier-name-or-path Qwen/Qwen3-8B \
    --speculator-type dspark \
    --num-layers 5 --block-size 8 \
    --target-layer-ids <步骤二打印的那 5 个，空格分隔> \
    --markov-rank 256 --markov-head-type vanilla \
    --enable-confidence-head --confidence-head-with-markov \
    --loss-fn '{"ce":0.1,"tv":0.9}' \
    --confidence-head-alpha 1.0 \
    --draft-attn-impl sdpa \
    --total-seq-len 2048 \
    --epochs 1 --log-freq 1 \
    --save-path ./output/dspark_8b_smoke/checkpoints
```
> `--legacy-data` 路径下 `--data-path` 直接指向 `.ckpt` 目录,**不用 `--hidden-states-path`,也不涉及 `--on-missing`**(`SampleFileDataset` 只 list 存在的文件,天然跳过 vLLM 漏掉的样本)。会有一条 `--legacy-data is deprecated` 警告,忽略即可。

**看的信号**:
1. 数据能加载(`list_files` 递归发现 `rows_X-Y/*.ckpt`);
2. `ce_loss`、`tv_loss` 都是**有限值且在下降**;`tv_loss` 一路 0/NaN = 末层没喂对 → 回步骤三;
3. EAL / accept_len 指标 ≠ 0。

### 5.2 完整训练

冒烟绿了,放大数据、`--epochs 5` 左右、`--save-path` 指定正式目录。训完跑一次 norm canary(确认 #711 混精缺陷没复现):
```bash
python scripts/check_norm_canary.py ./output/dspark_8b_smoke/checkpoints
```

---

## 6. 常见坑

| 现象 | 原因 | 处理 |
|---|---|---|
| `check` 的 [2] 命中率接近 0 | patch 没生效 / sglang 版本不对 / 层号错 | 重跑步骤一验证 patch;确认 sglang 0.5.9;核对 `--target-layer-ids` 与生成打印一致 |
| `.ckpt` 里没有 `last_hidden_state` | 生成时 patch 没打,或用了没改的旧代码 | 先 `patch_sglang_prenorm.py` 再生成;确认 SpecForge 代码已含 `captured_last_hidden` 读取 |
| patch 脚本报 `anchor not found` | sglang 非 0.5.9 | 别硬跑,报版本给我更新锚点 |
| `tv_loss` NaN | 末层口径错 / 层号 fc 宽度不匹配 | 步骤三先过,再训 |
| 训练发现文件为空 | `--data-path` 指错(要指到含 `rows_X-Y/` 的父目录) | 指向 SpecForge `--output-path` |

---

## 7. 一句话流程

```
容器启动: python patch_sglang_prenorm.py   (幂等,验证 "self.captured_last_hidden" 在 qwen2.py)
   ↓
生成: prepare_hidden_states.py --target-model-backend sglang --tp-size 1  → .ckpt(含 pre-norm last_hidden_state)
   ↓
验证: check_specforge_hidden.py  (fc=20480 + norm 命中率高)  [+ 可选 hf 交叉比对 cosine≈1]
   ↓
训练: train.py --legacy-data --legacy-data-format specforge  (ce/tv 有限且降, EAL≠0)
   ↓
32B: 同一套,--tp-size 8,先 check 再放量
```
