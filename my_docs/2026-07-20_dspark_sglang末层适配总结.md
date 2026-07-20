# DSpark × SpecForge(sglang)末层适配总结

**日期:2026-07-20**
关联:[[speculators-npu-dsv4-status]]、`my_docs/dspark_sglang末层patch_8b实验指南.md`

---

## 0. 快速使用(TL;DR)

### (a) SpecForge:打 patch(容器启动、拉 sglang 之前)
```bash
# 打 patch —— 让 sglang 产出 pre-norm 末层(幂等,看到 "OK: patched")
python scripts/patch_sglang_prenorm.py

# 反 patch —— 仅在用 hf 后端做交叉验证 / 容器转回普通推理时
python scripts/depatch_sglang_prenorm.py
```

### (b) SpecForge:生成 hidden(sglang 后端)
```bash
torchrun --nproc_per_node=1 scripts/prepare_hidden_states.py \
    --model-type dflash --target-model-backend sglang \
    --tp-size 1 \                        # 8B=1;32B=8
    --target-model-path Qwen/Qwen3-8B --num-draft-layers 5 \
    --chat-template qwen3_no_system_prompt \
    --data-path <sharegpt.jsonl> --output-path <OUT> --max-length 2048
# 记下打印的 "[IMPORTANT] Captured target layer IDs: [...]"
```

### (c) speculators:训练要加的关键参数
```bash
torchrun --standalone --nproc_per_node <N> scripts/train.py \
    --legacy-data --legacy-data-format specforge \      # ← 读 SpecForge .ckpt
    --data-path <OUT，即含 rows_*/ 的父目录> \           # ← 不要指到 rows_X-Y 里
    --verifier-name-or-path Qwen/Qwen3-8B \
    --speculator-type dspark \
    --num-layers 5 --block-size 8 \
    --target-layer-ids <上面生成打印的那几层，空格分隔> \  # ← 必须和生成一致
    --markov-rank 256 --markov-head-type vanilla \
    --enable-confidence-head --confidence-head-with-markov \
    --loss-fn '{"ce":0.1,"tv":0.9}' \
    --draft-attn-impl sdpa --total-seq-len 2048
# legacy 路径无需 --hidden-states-path / --on-missing;会有 --legacy-data deprecated 警告,忽略
```

---

## 0.5 冒烟测试实证(待填写)

> 用 Qwen3-8B 跑一遍,把实测填进"实测"列。判据见 §六 和 8B 实验指南。

| 检查项 | 期望 | 实测 |
|---|---|---|
| patch 生效 | `qwen2.py` 含 `self.captured_last_hidden` | |
| 生成 .ckpt 文件数 | > 1(否则 train 可能被切空) | |
| `.ckpt` 含 `last_hidden_state` | 是,shape `[1,seq,hidden]` | |
| check `[1]` fc_width | 5 × hidden(8B=20480) | |
| check `[2]` 下一 token 命中率 | 高(自然文本 > 0.5) | |
| hf 交叉比对 cosine | ≈ 1.0 | |
| 训练 `ce_loss` / `tv_loss` | 均有限且下降 | |
| EAL / accept_len | ≠ 0 | |

**结论 / 备注(自填)**:

---

## 一、目标

让 **SpecForge 用 sglang 后端生成的 hidden** 能正确喂给 **speculators 训练 DSpark**。核心难点是 DSpark 的 TV loss 需要 **verifier 最后一层的 pre-norm hidden**,而 sglang 后端原本不产这个张量。

链路:`SpecForge(sglang 生成 .ckpt)→ speculators offline 训练(--legacy-data-format specforge)`,绕开 vllm-ascend 不稳的 extract_hidden_states。

---

## 二、问题的根

speculators 全系(eagle3 / dflash / dspark)重建 verifier logits 都是:
```python
verifier_lm_head(verifier_norm(verifier_last_hidden_states))
```
`verifier_norm` 就是 verifier 自己的 `model.norm`。要重建出正确 logits,`verifier_last_hidden_states` **必须是最终 norm 之前(pre-norm)的最后一层 decoder 输出**。

而 SpecForge sglang 后端原来 `last_hidden_states=None`——沒有这个字段,训练根本跑不了 TV loss。

---

## 三、调研结论(逐一核过源码,不靠猜)

| 来源 | 末层是什么 | 结论 |
|---|---|---|
| **speculators** `dflash/core.py:351` | 需要 pre-norm(要再过 `verifier_norm`) | 基准口径 = **pre-norm** |
| **HF transformers** `outputs.hidden_states[-1]` | `_can_record_outputs={"hidden_states":DecoderLayer}`,记的是 DecoderLayer 输出、不含最终 norm | **pre-norm**,天然兼容;hf 后端直接可用 |
| **sglang 0.5.9** `output.hidden_states` / logits 输入 | 喂 lm_head 的那个 = **post-norm** | 不能直接用 |
| **sglang 0.5.9** aux capture | 层执行前抓 `hidden_states+residual`(pre-norm 输入残差),`layers_to_capture=[id+1]` | **够不到最后一层**(loop index N 不可达) |

**关键洞察**:sglang 的 pre-norm 末层 = `Qwen2Model.forward` 里 `self.norm(hidden_states, residual)` **执行前**的 `hidden_states + residual`。它算出来后立刻被 norm 吃掉、丢弃。这正是我们要抓的那一份,且它 == HF 的 `outputs.hidden_states[-1]`。

（另注:SpecForge `sglang_backend/utils.py` 的 eagle3 wrapper 里 `last_hidden_states` 是 post-norm;0.5.9 已多传 `hidden_states_before_norm` 但该 wrapper 没用它——佐证 pre-norm 在 0.5.9 是能拿到的。）

---

## 四、做的改动

> 说明:**speculators 侧的适配器早先已合入**(见下 4.0);**今天(07-20)的 sglang 末层 patch 之所以无需再动 speculators**,正是因为那个适配器已经在读 `last_hidden_state`。所以"不改 speculators"只针对今天这步,整体 bridge 是两侧都改过的。

### 4.0 speculators 侧适配器(早先合入,`fg11991/speculators`·`npu-support`)
- **`ea8d9aa`** `feat(train): read SpecForge DFlash hidden via --legacy-data-format specforge`:
  `data.py` 加 `standardize_data_specforge`(`hidden_state`→hidden_states、`last_hidden_state`→verifier_last_hidden_states、squeeze batch 维)、`_drop_leading_batch_dim`、`LEGACY_STANDARDIZE_FNS`、`SampleFileDataset` 可插拔 `standardize_fn`、`list_files` 收 `.ckpt`;`dataloader.py` 串 `legacy_data_format`;`train.py` 加 `--legacy-data-format`;新增校验脚本 `scripts/check_specforge_hidden.py` + 单测 `test_specforge_adapter.py`。
- **`2691887`** `feat(train): support gzipped .ckpt.gz`:
  `_load_sample_file` 加 gzip(`gzip.open`+`BytesIO`,照搬 SpecForge preprocessing.py)、`list_files` 收 `.ckpt.gz`。

### 4.1 今天(07-20)的 SpecForge 侧改动(单 commit `a1e1205`)
`fg11991/specforge` · `dspark-npu-offline`,3 个文件:

### 1. `scripts/patch_sglang_prenorm.py`(启动脚本 patch)
- 幂等,往 sglang **`Qwen2Model.forward` 最终 norm 前**插一句:
  ```python
  self.captured_last_hidden = hidden_states + residual if residual is not None else hidden_states
  ```
  Qwen3 继承此 forward,一并覆盖。
- 锚点是 sglang **v0.5.9** 的 final-norm 块(**注意缩进已按实际代码修正为嵌套层级**);版本不符会报错停住,不会静默改错。
- 容器启动脚本里、**拉起 sglang 之前**跑一次。

### 2. `scripts/depatch_sglang_prenorm.py`(反 patch)
- 精确逆操作:直接 import patch 的 `_ANCHOR/_PATCHED`,两者永不漂移;幂等。
- **用途:hf 后端交叉验证时必须先 depatch**——hf 后端天然产 pre-norm 末层,同一容器若还带着 sglang patch 会不干净。跑完验证或把容器转回普通推理前也用它还原。

### 3. `specforge/modeling/target/dflash_target_model.py`(sglang 后端读末层)
- `SGLangDFlashTargetModel.generate_dflash_data` 读 `self.model_runner.model.model.captured_last_hidden`,按 `input_lens` split,填进 `last_hidden_state`。patch 不在时保持 `None`,eagle3/普通流程不受影响。

### 为什么这就和 speculators 一致
`hidden_states + residual`(norm 前)= pre-norm 末层 = hf 的 `outputs.hidden_states[-1]`。speculators 的 `standardize_data_specforge`(4.0 的 `ea8d9aa` 已合入)现成读 `last_hidden_state`(→ `verifier_last_hidden_states`),口径自然对上,**今天这步无需再动 speculators**。

---

## 五、附带核实的两件事

1. **嵌套目录读取**:SpecForge 存成 `output_path/rows_{X}-{Y}/data_{i}.ckpt(.gz)`;speculators `list_files` 用 `os.walk` **递归**,能全部发现(实测 79 文件、train/val=71/8 正常)。`--data-path` 指到含 `rows_*/` 的父目录即可。
2. **训练报 `IndexError: self.data[0]`** 的根因不是目录逻辑(发现正常),而是**训练那次拿到空文件列表**——多为 `--data-path` 少指一层,或启动训练时生成还没跑完。用验证过的绝对路径、生成完成后重跑即可。
   - 边界:若某次只生成出 1 个可用 ckpt(DFlash 过滤 loss token < 2×block_size 的短样本),`int(1×0.9)=0` 会把 train 切空、复现同错。暂未加保护(需改 speculators)。

---

## 六、部署 & 验证流程(详见 8B 实验指南)

```
容器启动: python scripts/patch_sglang_prenorm.py        # 看到 "OK: patched"
生成:     prepare_hidden_states.py --model-type dflash --target-model-backend sglang \
          --tp-size 1(8B)/8(32B) --num-draft-layers 5  → .ckpt 含 pre-norm last_hidden_state
验证①:   scripts/check_specforge_hidden.py <一条.ckpt>  # fc=5×hidden + norm 命中率高
验证②:   depatch → hf 后端同样本生成 → 比对 last_hidden_state cosine≈1(金标准)
训练:     scripts/train.py --legacy-data --legacy-data-format specforge \
          --data-path <output_path 父目录>  # ce/tv 有限且降, EAL≠0
```

---

## 七、提交记录

| 仓库 | 分支 | commit | 内容 |
|---|---|---|---|
| `fg11991/speculators`(wyd) | `npu-support` | `ea8d9aa` | **speculators 适配器**:`--legacy-data-format specforge` + 校验脚本 + 单测(早先) |
| `fg11991/speculators`(wyd) | `npu-support` | `2691887` | **speculators**:`.ckpt.gz` gzip 支持(早先) |
| `fg11991/specforge` | `dspark-npu-offline` | `a1e1205` | SpecForge:patch + depatch + sglang 后端读末层(今天,单 commit) |
| `fg11991/speculators`(wyd) | `npu-support` | `ce3424e` | 8B 端到端实验指南 md(今天) |
| `fg11991/speculators`(wyd) | `npu-support` | `0b79a7b` / 本次 | 本适配总结(今天) |

提交身份统一 `w00958190 <wuyidong5@huawei.com>`,无 Claude 署名。

---

## 八、仍待落实(tige 实测)

1. **TP=8 下** `hidden_states + residual` 是否全宽 hidden(理论上残差流 all-reduce 后全宽,和已工作的 aux 同机制)——放量前用 check 脚本 + hf 交叉验证坐实。
2. patch 锚点绑 **sglang 0.5.9**;升级 sglang 需更新锚点(脚本会报错提醒,不会静默失败)。
3. 8B 冒烟跑通后按同一套上 **32B**(sglang `--tp-size 8`)。
