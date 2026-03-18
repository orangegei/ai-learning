# pi0 学习规划（按 8 步原则）

> 论文：`pi0.pdf`（`π0: A Vision-Language-Action Flow Model for General Robot Control`）

## 1. 学习原则（严格按图执行）

每次学习都走完这 8 步，不跳步：

1. `bounce what to make`：先定义今天要做的最小可运行目标（MVP）。
2. `code it`：先写代码/伪代码，不空读。
3. `debug it so it works`：必须跑通，记录 bug 与修复。
4. `explain parts`：解释每个模块输入/输出和作用。
5. `the intuition why it works`：用直觉解释为什么有效。
6. `the math intuition why`：写出关键公式和变量含义。
7. `explain to me like i'm 12`：用 12 岁能懂的话复述。
8. `go into all details`：补全实验细节、超参、边界条件。

---

## 2. 总体目标（6 周）

- 读懂 pi0 的核心：`VLM backbone + action expert + flow matching + action chunking + pretrain/post-train`
- 能实现一个简化版 `pi0-mini`（非机器人真机，离线数据或 toy 环境）
- 能讲清楚：为什么 flow matching 比离散动作 token 更适合连续控制
- 产出完整学习文档与可复现实验脚本

---

## 3. 每周计划

## Week 1：搭建全景 + 最小原型

- 目标：
  - 画出 pi0 全流程图（输入图像/语言/状态，输出动作 chunk）
  - 跑通一个 toy flow matching demo（1D/2D 连续动作）
- 实操：
  - 用 PyTorch 写 `flow_matching_toy.py`
  - 支持训练 + 采样 + 可视化
- 验收：
  - 你能用 3 分钟解释 pi0 架构
  - toy demo 可收敛，loss 下降

## Week 2：Action Expert + Action Chunking

- 目标：
  - 理解并实现动作 chunk（例如 `H=10/20/50` 可切换）
  - 搭建简化 `action expert`（Transformer block 即可）
- 实操：
  - 写 `action_expert.py`
  - 输入 observation embedding，输出 chunk 动作向量
- 验收：
  - 能解释为何 chunk 比单步动作更稳
  - 能对比不同 `H` 的效果与速度

## Week 3：多模态输入与训练配方

- 目标：
  - 实现图像/文本/状态三路输入融合（可先用假特征代替真 VLM）
  - 复现 pretrain + post-train 两阶段流程
- 实操：
  - 写 `dataset_mixture.py`（多数据源采样权重）
  - 写 `train_pretrain.py` 与 `train_posttrain.py`
- 验收：
  - 能解释“只高质量数据”与“只大杂烩数据”各自缺陷
  - 两阶段训练曲线可对比

## Week 4：语言条件与高层策略

- 目标：
  - 跑通“高层语言指令 -> 低层动作执行”的简化 pipeline
  - 复现论文里的“中间语言子目标”思想
- 实操：
  - 写 `high_level_policy_stub.py`（规则或小模型都可）
  - 写 `eval_language_conditioned.py`
- 验收：
  - 对比三种输入：仅总目标 / 人工中间指令 / 自动中间指令
  - 形成一页实验结论

## Week 5：消融与失败分析

- 目标：
  - 做最关键消融：去掉 VLM 初始化、去掉 flow matching、减小模型
  - 系统化记录失败案例
- 实操：
  - 写 `ablation.yaml` 和统一评测脚本
  - 输出表格：成功率、平均进度、推理速度
- 验收：
  - 你能说清每个组件的必要性
  - 有 5 个以上“失败 -> 修复”案例

## Week 6：讲解闭环与最终文档

- 目标：
  - 完成“专家版 + 小白版”双重讲解
  - 交付完整复现报告
- 实操：
  - 写 `pi0_study_report.md`
  - 录一段 10 分钟讲解（可选）
- 验收：
  - 你可以在不看稿情况下讲清核心公式与直觉
  - 别人按你的文档可复现主要结果

---

## 4. 每天执行模板（90-120 分钟）

1. `15 min`：确定今天的“最小可运行目标”
2. `35 min`：编码实现
3. `20 min`：调试到可运行
4. `10 min`：模块解释（输入/输出/依赖）
5. `10 min`：直觉解释（为什么会有效）
6. `10 min`：数学解释（公式+变量）
7. `5 min`：12 岁解释版
8. `5 min`：细节补档（超参、错误、结论）

---

## 5. 本周就可以开始的任务清单

- [ ] 通读 `pi0.pdf`，提取 15 个关键词（VLA、flow matching、action chunking、cross-embodiment、post-training 等）
- [ ] 手写一页“pi0 架构图”
- [ ] 完成 `flow_matching_toy.py` 第一个可运行版本
- [ ] 记录至少 3 个 bug 与修复
- [ ] 写两版解释：技术版（300 字）+ 12 岁版（120 字）

---

## 6. 学习产出要求（每周固定交付）

- `notes/weekX_summary.md`：本周结论
- `notes/weekX_math.md`：核心公式与推导直觉
- `notes/weekX_eli12.md`：儿童版解释
- `experiments/weekX_results.md`：实验配置、结果、失败案例

---

## 7. 自测标准（判断是否“学会”）

- 你能回答：
  - pi0 为什么用 flow matching 而不是离散动作 token？
  - action chunking 在控制频率和稳定性上带来什么？
  - pretrain/post-train 分工是什么？
  - 高层语言子目标为什么能提升长任务成功率？
- 你能做到：
  - 独立实现简化版训练与推理脚本
  - 复现实验并解释失败样例
  - 给非技术同学讲明白核心思想

