# DSpark × speculators 工作总结(截至 2026-07-22)

**成文日期:2026-07-22**,覆盖 2026-07-20 ~ 07-22 的工作。
关联:[[speculators-npu-dsv4-status]]、`my_docs/2026-07-20_dspark_sglang末层适配总结.md`、`my_docs/2026-07-20_dspark训练显存与更大模型可行性.md`、`my_docs/dspark_sglang末层patch_8b实验指南.md`

---

## 0. 一句话现状

SpecForge(sglang,pre-norm 末层)生成 hidden → speculators 训练 DSpark 的链路**已跑通**,`full_acc` 200 步到 0.27(比原生 vLLM 路径的 0.146 快很多,佐证末层口径修对了)。本阶段做了大量**诊断 + 一处代码修复(坏数据跳过)**,并厘清了部署侧口径差异。唯一代码改动是坏数据跳过(已合入 `npu-support`,commit `7f2cf59`)。

---

## 一、基础(2026-07-20,已有)

- **sglang pre-norm 末层 patch**(SpecForge 侧)+ speculators `--legacy-data-format specforge` 适配器 + `.ckpt.gz` 支持 + 8B 端到端实验指南。详见 `2026-07-20_dspark_sglang末层适配总结.md`。
- 让 SpecForge sglang 后端产出正确的 pre-norm `verifier_last_hidden_states`,喂 speculators 训 DSpark 的 TV loss。

---

## 二、本阶段做的诊断与结论(2026-07-21 ~ 07-22)

### 2.1 TensorBoard 接入与验证
speculators 训练**本就内建 TensorBoard**(`train.py:461 setup_metric_logger` + `logger.py TensorBoardHandler`),默认没输出只因 `--logger` 默认空。加 `--logger tensorboard --run-name <name> --log-freq 1` 即可。冒烟必看 tag:`train/tv`、`train/ce`、`train/full_acc`、`train/position_*_acc`、`train/eal`;只有 rank0 写、`global_step % log_freq == 0` 才写。

### 2.2 显存杠杆 + 更大模型可行性(仅文档,无代码)
出了 `2026-07-20_dspark训练显存与更大模型可行性.md`(在 `dev/train-mem-scaling` 分支):
- **主导显存 = `[max_anchors·block_size, vocab]` 词表宽张量 + TV loss 的 fp32 中间量**。
- 头号杠杆 **词表裁剪 `--draft-vocab-size`**(现成功能,原生管线零前置,`prepare_data.py` 已产 `token_freq.pt`);其次 **HSDP `--fsdp-shard-size`**(现成,替换裸 `--fsdp-shard` 消跨节点 all-gather)。
- GLM-5.x 量级:离线训 draft 仍能在当前 8×8×910C 上跑(验证模型不在训练显存里);瓶颈转移到 MoE 分布式推理生成 hidden + TB 级 hidden 存储。
- **需改代码的三项**(activation checkpointing / 分块 TV loss / verifier_logits 只在 anchored 位算)**尚未开发**,仅提案。

### 2.3 hidden 生成与词表解耦
核过两侧代码:生成 hidden **与词表无关**(只存 token_ids + hidden 向量);词表裁剪只在**训练时**对 lm_head/verifier_lm_head 切行。**推论:开 `--draft-vocab-size` 不用重生成 hidden**,同一份缓存可扫 `draft_vocab_size × max_anchors`。

### 2.4 gz 压缩现状:两侧就绪,加 `--compress` 即可
SpecForge `prepare_hidden_states.py --compress`(+`--compression-level`)产 `.ckpt.gz`;speculators 读侧 `2691887` 已支持。混放(.ckpt 与 .ckpt.gz)也能一起训。无需再适配。

### 2.5 部署报错诊断(fc 形状不匹配)
`[5120,15360](checkpoint,3 aux) vs [5120,25600](vLLM 建,5 aux)`:
- vLLM-ascend DSpark 的 fc = `len(hf_config.target_layer_ids) × hidden`(`patch_dspark_proposer.py:90`、`model_runner_v1.py:3944`);它加载的 config 里 `target_layer_ids` 是 **5 个**,而 checkpoint 按 **3 个** aux 训的 → 崩。`25600 = 5×hidden = Qwen3-32B intermediate` 是巧合。
- **`Qwen3DSparkModel`(开源发布版,扁平 schema、`target_layer_ids`、block7、full attention)vs `DSparkDraftModel`(speculators 包装 schema、`aux_hidden_state_layer_ids`、block8、sliding)= 同一个 DSpark 的两套 config 打包**,底层数学一致;markov/confidence head 权重名/形状对得上。
- "deepseek" 是误会:发布权重挂在 `deepseek-ai/` org + vllm-ascend 另有独立的 DeepSeek-MTP 路径(`method="mtp"`),与 DSpark(`method="dflash"`)无关。
- **block_size 8 vs 7 = 口径差 1**:speculators 把 anchor 算进 block_size(8=1 anchor+7 draft),开源只数 draft 位(7),两边都投机 7 个 token,等价。DFlash/DSpark 的 "bonus vs anchor+1" 差异在 specforge 内部(label_offsets 起点 0 vs 1)。

### 2.6 训练脚本正确性核查(tige Qwen3-32B)
- **verifier 末层是对的**:`launch_vllm.py --include-last-layer` 默认 True,自动追加真末层;`--target-layer-ids "3 31 61"` 实际抓 `[3,31,61,64]`,前 3 喂 fc(3×5120=15360,对上),第 4(真末层)作 verifier_last。TV target 正确。
- **揪出脚本 bug**:`MARKOV_HEAD_TYPE=${MARKOV_HEAD_TYPE:-vanilla}.` 末尾多一个 `.` → 传成 `vanilla.`,`markov_head_type: Literal["vanilla","gated","rnn"]` 会 `ValueError` 崩训。**删掉那个点。**

### 2.7 acc 涨得快的原因
`full_acc` 是 draft argmax vs target argmax;target = `verifier_lm_head(verifier_norm(verifier_last))`。之前原生 vLLM-ascend 的 verifier_last 不稳/口径错 → target 歪 → acc 卡 0.146;SpecForge 的正确 pre-norm 末层 → target 正确 → 快速收敛。用 `train/tv` 平滑降 + `train/eal` 同步涨确认非虚高。

### 2.8 开源 DFlash 模型增训可行性
speculators 支持:`convert_model(algorithm="dflash", model="z-lab/*-DFlash", verifier=...)` 转成 speculators 格式 → `scripts/train.py --from-pretrained <converted>` 继续训(finetune,新优化器、用小 LR)。对齐约束:**verifier、aux 层号+个数、block_size** 必须与源 checkpoint 一致(否则重演 2.5 的 fc 坑)。加 DSpark 头则新头随机初始化。

---

## 三、本阶段唯一代码改动:坏数据跳过(已合 `npu-support` `7f2cf59`)

**问题**:某个 SpecForge `.ckpt` 缺 `hidden_state` key(生成被截断的半成品),`standardize_data_specforge` 直接 KeyError,在 DataLoader worker 抛出 → 拖垮整个 64 卡任务、每次卡在同一 step。而 `--on-missing skip` 原本只管"文件缺失"、不管"文件损坏",对 legacy specforge 路径是 no-op。

**改动(3 文件,纯加法,默认行为不变)**:
- `train/data.py`:`SampleFileDataset` 加 `skip_bad`(默认 False);`_get_raw_data` 在 skip 模式下 try/except 坏样本 → 返回 None(collate 已过滤)+ 实时 `logger.warning` 打印坏文件名 + 跨 worker 计数;新增 `take_skipped_count()`。
- `train/dataloader.py`:legacy 分支 `skip_bad = (on_missing == "skip")`——复用现有 `--on-missing skip`。
- `train/trainer.py`:每 epoch 末 `_report_skipped_samples`(全卡 all_reduce、rank0 打印 `[epoch N/M] skipped X bad sample(s)`)。

**验证**:本地造缺 `hidden_state` 的坏文件实测——skip 模式跳过并计数=1、好样本正常加载;默认模式仍 raise(严格行为保留)。3 文件 3.11 语法通过。

> ⚠️ 生效需节点拿到新代码;节点拉的是华为 codehub 的 `npu-support`,`wyd`(github)已推,codehub 需另行同步。坏文件说明生成被中断过,skip 只是止损,那批样本已丢,值得确认是否补跑。

---

## 四、待办 / 提案(未开始)

1. **省显存三项代码改动**(需批准):verifier_logits 只在 anchored 位算(最小,先做)、draft 层 activation checkpointing、分块/融合 TV loss。
2. 部署到 vLLM-ascend:把 speculators 产物对齐开源 DSpark 口径(schema 字段名 `target_layer_ids`、aux 层数、block_size、attention),或改 vLLM-ascend 读 `aux_hidden_state_layer_ids`。
3. 词表裁剪 + HSDP 上量(现成功能,直接加参数即可)。

---

提交身份统一 `w00958190 <wuyidong5@huawei.com>`,无 Claude 署名。
