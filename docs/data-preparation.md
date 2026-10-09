# BIRD 数据准备

本项目固定使用 BIRD 官方 OSS 的 `train.zip` 与 `dev.zip`。开发集版本是 `dev_20240627`，1,534 题、11 个数据库。官方已另外发布 20251106 清洗版；本实验不混用两版，报告需注明当前版本。原始 gold SQL 保持原样，不能根据是否能执行、是否为空结果或教师是否答对来删题。

## 来源与许可

| 资源 | 官方来源 |
| --- | --- |
| 官网及下载入口 | https://bird-bench.github.io/ |
| 训练集 | https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip |
| 开发集 | https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip |
| 固定 README/许可 | https://github.com/AlibabaResearch/DAMO-ConvAI/blob/188835d4f9948563a6b9c8ac50cd0f3ae4021ed6/bird/README.md |
| 官方执行评测参考 | https://github.com/AlibabaResearch/DAMO-ConvAI/blob/188835d4f9948563a6b9c8ac50cd0f3ae4021ed6/bird/llm/src/evaluation.py |
| 数据许可 | CC BY-SA 4.0：https://creativecommons.org/licenses/by-sa/4.0/ |

官方 README 明确声明 CC BY-SA 4.0。保留 BIRD 来源与署名；如再分发衍生数据，遵守同许可。代码参考和数据许可分别记录，项目自己的数据处理脚本不复制上游训练代码。原文证据在本地 `D:\DATA\Dataset\BIRD\raw\sources\BIRD-official-README.md`，另有 `provenance.json` 记录固定 commit、URL 与 README SHA256。论文：Li et al., *Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs*, arXiv:2305.03111。

## 目录与下载验收

本地根目录是 `D:\DATA\Dataset\BIRD`，服务器根目录是 `/root/autodl-tmp/datasets/BIRD`。两个位置都按以下结构组织：

```text
raw/                 官方 ZIP 与来源证据
extracted/dev/       官方 dev JSON、内嵌 ZIP、SQLite 和原始列描述
extracted/train/     官方 train 内容，待下载完成后生成
processed/           JSONL 和 manifest.json
work/                下载状态 JSON、后台日志
```

`scripts/data/download_bird.py` 仅从官方 OSS 下载。状态先是 `downloading`，之后是 `verifying`，完成长度校验、可用的官方 MD5、全包 SHA256 与 ZIP CRC 后才写 `complete`，并把 `.zip.partial` 原子改名为 `.zip`。中断可用 HTTP Range 续传，服务器必须返回匹配起点的 206 响应。不能用“文件存在”或“文件大小接近”代替完成状态。

2026-10-04 本地开发集实测：346,207,293 bytes；官方 `Content-MD5` 与本地 MD5 都是 `04b4af221c9186361f09b16abfd917ec`，SHA256 为 `cdd6d19faeb45a23970b98d3ef6c40a87987c95459c2cf12076897a60cf5a630`。外层及内层 ZIP CRC 校验通过。数据库包为 345,997,266 bytes，内部解压量 1,493,445,090 bytes；全部 11 个 SQLite 的 `PRAGMA quick_check` 返回 `ok`。

训练集官方外层大小 8,919,543,554 bytes，其内嵌 `train_databases.zip` 解压出来是 9,347,158,408 bytes，数据库实际展开量需下载完成后读取内层目录核实。训练集是 multipart OSS 对象，无官方 `Content-MD5`；multipart ETag 不能当文件 MD5。其验收记录本地计算的 SHA256 和 ZIP CRC，不能宣传成“官方 SHA256 对照验证”。每层解压前会检查剩余空间是否足够容纳 ZIP 的全部解压大小并额外预留 1 GiB。

## 数据合同和划分

每条 JSONL 严格含这 10 个字段：

```json
{"id":"dev:0","db_id":"california_schools","question":"...","evidence":"...","gold_sql":"...","schema":"...","db_path":"...","difficulty":"simple","source_split":"dev"}
```

`schema` 包含该数据库全部用户表/视图的完整 SQLite DDL，以及官方 `database_description` 中的列名、含义、值说明。没有按问题或 gold SQL 筛选表，也没有调用模型生成 schema。CSV 先按 UTF-8 BOM 解码，少量原始 Windows-1252 文件按对应编码解码；保留多行字段并忽略 macOS `._` 资源叉。`db_path` 是实际本地路径；在 Linux 服务器处理时自然生成 Linux 路径。

`gold_sql` 只用于评测和必要的监督基线标签；OPD 的问题提示只能使用 question、evidence、schema，不能拼入 gold。训练仅使用官方 train，验证/原始基线仅使用官方 dev。`prepare_bird.py` 会检查 train/dev 的数据库 ID 交集并在非空时失败。

输出是 `processed/train.jsonl`、`processed/dev.jsonl` 和 `processed/baseline-dev-120.jsonl`。训练包未下载时，manifest 的 train 为 `pending`，不生成替代训练文件；此时 train/dev 交集尚未实证，`split_disjointness_verified=false`。

120 题样本固定 seed=42，以 `(db_id, difficulty)` 为分层，用 Hamilton 最大余数法按比例分配名额，再对每层按 ID 排序并随机打乱取样。选择只读取数据库 ID、难度、样本 ID，不看 SQL、执行结果或模型表现。当前抽样为 72 simple、36 moderate、12 challenging，覆盖全部 11 库。ID 顺序用换行连接后 UTF-8 的 SHA256 为 `1c97485b5c04b2b6917608f13ee1748b6118a4bab9ec2c425487439e4f0f86d6`；Windows/Linux 的 `db_path` 不同导致 JSONL 文件 hash 不同，但 ID 校验值一致。

manifest 记录下载头信息、大小、SHA256、版本目录、样本数、库 ID、每库 SQLite hash/大小/quick_check、schema 长度、分层分布、120 IDs 与文件 hash。当前完整 schema 的 dev 加权平均约 19.1 KB、最大约 41.9 KB；纯 DDL 加权平均约 3.7 KB、最大约 7.3 KB。token 长度必须由真实学生 tokenizer 测量；baseline 不截断 schema，超出上下文的题也保留在固定 120 题分母中并报告失败原因。

服务器 baseline v1 在模型加载前因 Transformers 5.18 的 BatchEncoding 返回格式退出，未产生预测。修复后 baseline v2 使用完善 UTF-8/Windows-1252 解码的最新数据，120 条记录完整快照到 run 的 records.jsonl，并保存实际 prompt token IDs 的 SHA256；蒸馏后复评必须核对同一快照与提示 hash。

## 执行方式

本地默认使用指定 Python：

```powershell
& 'D:\.conda\envs\model\python.exe' 'D:\Codex\opd-sql-agent\scripts\data\download_bird.py' --root 'D:\DATA\Dataset\BIRD' --split dev
& 'D:\.conda\envs\model\python.exe' 'D:\Codex\opd-sql-agent\scripts\data\prepare_bird.py' --root 'D:\DATA\Dataset\BIRD'
```

服务器可直接从官方下载，避免上传整个数据库。以项目 venv 为例：

```bash
/root/autodl-tmp/envs/opd/bin/python /root/autodl-tmp/opd-sql-agent/scripts/data/download_bird.py --root /root/autodl-tmp/datasets/BIRD --split dev
/root/autodl-tmp/envs/opd/bin/python /root/autodl-tmp/opd-sql-agent/scripts/data/download_bird.py --root /root/autodl-tmp/datasets/BIRD --split train
/root/autodl-tmp/envs/opd/bin/python /root/autodl-tmp/opd-sql-agent/scripts/data/prepare_bird.py --root /root/autodl-tmp/datasets/BIRD
```

长下载应由服务器 runner 放入后台并用定时监控读取 `work/dev-download.json`、`work/train-download.json`、日志和 PID。开发集先完成即可先准备/评测，训练包完成后重新 prepare 才能验证全划分。重跑处理会核对 archive hash；已解压且 hash 一致时跳过重复解压，但仍检查数据库和输出记录。若需要仅生成面向另一根目录的路径，`--db-path-root /root/autodl-tmp/datasets/BIRD` 会按相同 relative path 改写记录路径；这些路径只有对应数据库已经迁移时才可使用。
