# OPD SQL Agent

### 让学生在自己生成的 SQL 前缀上，向固定教师学习。

[English](README.md) · [CPU 演示](#先在-cpu-上跑通) · [完整结果](docs/results.md) · [实现原理](docs/architecture.md) · [GPU 复现](docs/reproduction.md)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CPU tests](https://github.com/lite93597/opd-sql-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/lite93597/opd-sql-agent/actions/workflows/tests.yml)

**一套可阅读、可改造的 Text-to-SQL 在线策略蒸馏训练与评测工具。** 使用 PyTorch 自定义训练循环、PEFT/LoRA、vLLM、NCCL 和 SQLite，实现当前学生轨迹上的完整词表反向 KL 蒸馏，并保留模型选择、逐题变化和失败分支的证据。

在事先冻结的 **BIRD dev 300 题、11 数据库子集**上，9B 学生的执行正确率达到 **60.67%**，比同一 SFT 起点继续监督微调的 **54.00%** 高 **6.67 个百分点**：新增正确 33 题，丢失正确 13 题。结果来自一个训练 seed 和本地评测器，不是官方全量榜单成绩。

![五组冻结模型的最终结果](assets/results.svg)

## 为什么值得下载

- **读懂蒸馏内部机制。** 当前学生生成轨迹，固定教师读取相同学生前缀，学生优化 `KL(student || teacher)`。训练入口直接展示前向、反向、梯度累积、更新和同步。
- **复用长上下文显存方案。** 每 8 个 token 位置计算一次完整词表 KL，及时获取隐藏状态梯度并释放临时计算图，再反传主干。8192-token 完整反传 / 优化 / 同步 probe 已通过，实际训练最长 7137 tokens。
- **避免 rollout 策略过期。** LoRA 合并、NCCL 完整参数同步、prefix cache 清理和版本检查，连接训练侧学生与 vLLM 副本。
- **检验真实任务效果。** 数据库隔离的内部验证、选择冻结后的五组测试、逐题 gained/lost 和数据库 bootstrap。600 步未改善的结果也公开保留。
- **不必先租 GPU。** 仓库带有纯 CPU 的合成 SQLite 演示，先理解评测器，再准备模型和数据。

适合学习 on-policy 蒸馏、研究 SQL 执行评测、搭建有对照实验的开发者。仓库不附带模型权重、adapter、BIRD 数据库或私人运维日志。

## 先在 CPU 上跑通

核心评测器仅要求 Python 3.10+。这个演示不需要 PyTorch、GPU、BIRD 或模型下载。

```bash
git clone https://github.com/lite93597/opd-sql-agent.git
cd opd-sql-agent
python -m venv .venv
# Linux / macOS：
source .venv/bin/activate
# Windows PowerShell：.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python scripts/demo.py
```

演示在 `work/demo/` 创建一个自造数据库和人工编写的预测 SQL，展示执行结果比较、重复行的影响与逐题配对。**演示分数不是模型成绩或训练收益。** 已存在的输出目录会被拒绝覆盖；再次运行可用 `--output-dir work/demo-2`。

使用公开 CLI 评测演示产生的预测：

```bash
python -m opd_sql.bird_evaluation \
  --records work/demo/records.jsonl \
  --predictions work/demo/candidate-predictions.jsonl \
  --comparison-predictions work/demo/reference-predictions.jsonl \
  --output work/demo/cli-report.json
```

PowerShell 可把选项写在一行。[示例说明](examples/README.md)包含数据格式和 CPU 测试命令。

## 实验结果与边界

学生为 **Qwen3.5-9B**，固定教师为 **Qwen3.8-27B**，使用固定快照与一致的 tokenizer 映射。输入为问题、完整 schema 和给定 evidence；输出一条只读 SQLite SQL，关闭 thinking。

| 最终冻结组 | 正确数 / 300 | 集合执行正确率 | 严格行多重集正确数 / 300 |
|---|---:|---:|---:|
| 原始学生 Base | 150 | 50.00% | 135 |
| 固定教师 | 193 | 64.33% | 177 |
| SFT 起点 S0，300 次更新 | 163 | 54.33% | 147 |
| 同 S0 继续 SFT，300 次更新 | 162 | 54.00% | 148 |
| 同 S0 OPD，LR 1e-5，300 次更新 | **182** | **60.67%** | **165** |

OPD 相对继续 SFT 净增 **20 题 / 6.67pp**，按数据库 cluster-bootstrap 的 95% 区间为 **[2.29, 11.54]pp**。SFT 起点本身相对 Base 提升 4.33pp；Base 到 OPD 的 10.67pp 总提升包含这部分收益，不能全部归因于 OPD。

五组都保留固定分母 300。共同 1 道 gold SQL 超时，五组统一计错。主指标比较结果行集合、忽略重复次数；严格指标保留重复行次数。两者都不是 SQL 字符串匹配，也不证明所有数据库状态下的语义等价。

**600 步负结果：**低学习率 LR 5e-6 的 OPD 在内部 120 题上最终为 67/120，对应 SFT 为 72/120。低 LR 的最佳内部成绩未超过首个 OPD 分支，因此按预定规则保留原 300 步配对，没有追加 600 步模型的最终测试。

**限制：**单 seed、固定子集、本地 SQLite 评测器。两个训练分支匹配起点、题目顺序和更新步数，但学习率、生成 token 数和算力预算不同；统计区间不覆盖训练 seed 与选择过程的不确定性。

[完整结果与选择过程 →](docs/results.md) · [公开聚合证据 →](artifacts/experiment-v1/README.md)

## OPD 的一步在做什么

```text
确认 vLLM 为当前学生版本
    → 生成 4 道题的学生 SQL
    → 教师和学生读取相同的学生生成前缀
    → 只在 completion 预测位置计算完整词表反向 KL
    → 按总 completion token 数归一化、累积梯度
    → 更新 LoRA
    → 合并并同步完整参数、清缓存、验证版本
    → 下一批生成
```

```text
q(v) = 学生的下一 token 概率
p(v) = 固定教师的下一 token 概率
KL(q || p) = sum_v q(v) * (log q(v) - log p(v))
loss = 所有 completion 位置的 KL 之和 / 总 completion token 数
```

离散生成本身不求导；学生在采样 token 上重新做可微分前向，教师关闭梯度。这里优化条件 token 分布，没有额外加入 REINFORCE 的采样梯度项。

SFT 使用标准 SQL 的 completion-only 交叉熵，OPD loss 不包含 gold SQL。分块针对 token 位置，每个位置仍计算完整词表。训练循环由 PyTorch 实现，TRL 提供 vLLM 客户端；本轮未使用 PPO/GRPO。

## GPU 环境与复现要求

记录中的实验运行于 Linux 与 **2 × 96GB RTX PRO 6000 Blackwell**。GPU 0 放教师与训练学生，GPU 1 跑 vLLM；这是职责划分，不是 DDP、tensor parallel 或单个 192GB 设备。

主要参数：LoRA `r=8 / alpha=16 / dropout=0`，5,898,240 个可训练参数；microbatch 1、accumulation 4；训练 context 8192、生成上限 512；KL chunk 8；模型 BF16、概率计算 FP32；每步进行完整参数同步。

精确依赖见 [server requirements](scripts/server/requirements.txt)。原 server 脚本固定了 Linux 目录布局，依赖已校验的模型下载 manifest、准备好的 BIRD 与 S0 checkpoint，**不是 fresh clone 后一条命令就能重跑全部 GPU 实验**。CUDA 13、Blackwell 和单独安装的 causal-conv1d 等要求见 [GPU 复现文档](docs/reproduction.md)。

数据按数据库划分：55 个训练库 / 7231 题，14 个内部验证库 / 2197 题。共同 8K 合格池 6216 题，不截 schema；300 步分支实际使用 1200 道不同题。内部 120 只选 checkpoint；最终 dev300 排除早期 dev120 的题目 ID，选择冻结后才评测。

## 从哪里读代码

| 想了解的部分 | 入口 |
|---|---|
| 分块 KL、同步与恢复 | [onpolicy.py](src/opd_sql/onpolicy.py) |
| SFT 训练 | [supervised.py](src/opd_sql/supervised.py) |
| Prompt / token 构造 | [prompts.py](src/opd_sql/prompts.py) |
| SQL 只读执行和限制 | [sqlite_tools.py](src/opd_sql/sqlite_tools.py) |
| 可移植评测 CLI | [bird_evaluation.py](src/opd_sql/bird_evaluation.py) |
| 下载、数据准备与划分 | [scripts/data/](scripts/data/) |
| 阶段复用与冻结流程 | [run_effect_experiment.py](scripts/server/run_effect_experiment.py) |
| 配对统计和协议审计 | [summarize_effect.py](scripts/analysis/summarize_effect.py) |

`configs/*7b.json` 是早期 Qwen2.5 参考配置，不是本轮 9B/27B 结果的配方，详见 [配置说明](configs/README.md)。

## 参与改进

欢迎提交评测边界问题、GPU 环境适配、带正确性验证的同步优化，或独立多 seed 实验。请保留数据版本、划分 ID、生成协议和负结果，参见 [贡献说明](CONTRIBUTING.md)。

如果实现或评测方法对你有帮助，欢迎 **Star**，帮助更多开发者找到项目。可使用 `git clone`，也可[下载源码 ZIP](https://github.com/lite93597/opd-sql-agent/archive/refs/heads/main.zip)。

项目代码采用 [MIT](LICENSE)。BIRD 与模型遵守上游许可，仓库不分发原始数据和权重，来源与许可边界见 [NOTICE.md](NOTICE.md)。
