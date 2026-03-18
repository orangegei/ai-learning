# Custom pi0 (Self-Written) + HF Weights + LIBERO

本目录只保留你要的路线：

- 模型实现是**自己写的**（不调用 lerobot 官方 policy 类）
- 从 Hugging Face checkpoint 加载参数
- 在 LIBERO 环境中运行
- 明确打印 `观测 -> 模型输入 batch -> 动作` 的数据流

---

## 目录说明

- `run_local_pi0_libero.py`
  - 主入口：创建 LIBERO 环境，构建自研模型，加载 HF 权重，执行 rollout。
- `custom_pi0/model.py`
  - 自研 pi0 风格模型（视觉/语言/状态编码 + flow-style action denoiser）。
- `custom_pi0/hf_loader.py`
  - 自研权重加载器（不依赖官方 policy）：
    - 解析 HF snapshot
    - 支持 safetensors / 分片权重
    - 参数名映射与形状匹配
    - 输出加载覆盖率报告
- `download_hf_policy.py`
  - 可选：提前把 HF 模型下载到本地目录。
- `scripts/install_lerobot_libero.sh`
  - Linux 依赖安装脚本（名字保留，内容已改为自研路线依赖）。

---

## 安装（Linux）

```bash
bash scripts/install_lerobot_libero.sh
```

脚本会安装：

- `requirements.txt`
- `LIBERO`（从官方 GitHub 安装）

---

## 运行

### 1) 直接从 HF 拉取并运行

```bash
python run_local_pi0_libero.py \
  --policy-path lerobot/pi0_libero_finetuned \
  --suite libero_object \
  --task-id 0 \
  --episodes 1 \
  --steps 200 \
  --trace \
  --debug
```

### 2) 先下载到本地，再离线运行

```bash
python download_hf_policy.py \
  --repo-id lerobot/pi0_libero_finetuned \
  --local-dir /path/to/local/pi0_libero_finetuned
```

```bash
python run_local_pi0_libero.py \
  --policy-path /path/to/local/pi0_libero_finetuned \
  --local-files-only \
  --suite libero_object \
  --task-id 0 \
  --episodes 1 \
  --trace \
  --debug
```

---

## 数据流可观测性

`--debug` 会打印：

- LIBERO 原始观测 key 到模型输入张量的映射
- 图像/状态/语言 token 的 shape

`--trace` 会打印：

- 视觉、语言、状态编码的中间张量 shape
- denoising 关键 step 的中间 shape

---

## 说明

- 由于 checkpoint 架构与自研模型可能不完全一致，`hf_loader` 会输出加载覆盖率（keys/params）。
- 可以用 `--min-load-ratio` 控制最低可接受加载比例，避免“几乎没加载到参数却误以为成功”。
