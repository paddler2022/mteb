# Code-Switching 自定义任务

本文档记录了在 `mteb_old` 中创建的 Code-Switching 变体检索任务。

## 任务概述

这些任务基于官方 MTEB 数据集创建，特点是：
- **queries**: 从本地 jsonl 文件加载（code-switching queries）
- **corpus 和 qrels**: 从官方 HuggingFace 数据集加载

## 任务列表

| 任务名 | 文件路径 | 环境变量 | 原始任务 | eval_split |
|-------|---------|---------|---------|------------|
| `TRECCOVIDCodeSwitching` | `mteb/tasks/retrieval/eng/treccovid_codeswitching_retrieval.py` | `TRECCOVID_QUERY_FILE` | TRECCOVID | test |
| `AILAStatutesCodeSwitching` | `mteb/tasks/retrieval/eng/aila_statutes_codeswitching_retrieval.py` | `AILA_STATUTES_QUERY_FILE` | AILAStatutes | test |
| `HagridRetrievalCodeSwitching` | `mteb/tasks/retrieval/eng/hagrid_codeswitching_retrieval.py` | `HAGRID_QUERY_FILE` | HagridRetrieval | dev |
| `SCIDOCSCodeSwitching` | `mteb/tasks/retrieval/eng/scidocs_codeswitching_retrieval.py` | `SCIDOCS_QUERY_FILE` | SCIDOCS | test |

## 数据格式

### queries.jsonl 格式

```jsonl
{"id": "1", "text": "your code-switching query text here"}
{"id": "2", "text": "another query text"}
```

或使用 `_id` 字段：

```jsonl
{"_id": "1", "text": "your code-switching query text here"}
```

## 使用方法

### 方式一：通过构造函数传参

```python
from mteb import MTEB
from sentence_transformers import SentenceTransformer
from mteb.tasks.retrieval.eng import (
    TRECCOVIDCodeSwitching,
    AILAStatutesCodeSwitching,
    HagridRetrievalCodeSwitching,
    SCIDOCSCodeSwitching,
)

# 加载模型（可选启用 Flash Attention）
model = SentenceTransformer(
    "BAAI/bge-large-en-v1.5",
    model_kwargs={"attn_implementation": "flash_attention_2"},
    device="cuda"
)

# 创建任务，传入 query 文件路径
#所有的CodeSwitching数据都在/root/autodl-tmp/workdir/mteb/Code_Switching_data/中
task = TRECCOVIDCodeSwitching(query_file="/path/to/codeswitching_queries.jsonl")

# 运行评估
evaluation = MTEB(tasks=[task])
results = evaluation.run(model)
```

### 方式二：通过环境变量

```bash
# 设置环境变量
export TRECCOVID_QUERY_FILE=/path/to/treccovid_queries.jsonl
export AILA_STATUTES_QUERY_FILE=/path/to/aila_queries.jsonl
export HAGRID_QUERY_FILE=/path/to/hagrid_queries.jsonl
export SCIDOCS_QUERY_FILE=/path/to/scidocs_queries.jsonl
```

```python
from mteb import MTEB
from sentence_transformers import SentenceTransformer
from mteb.tasks.retrieval.eng import TRECCOVIDCodeSwitching

model = SentenceTransformer("your-model-path")

# 自动从环境变量读取 query 文件路径
task = TRECCOVIDCodeSwitching()

evaluation = MTEB(tasks=[task])
results = evaluation.run(model)
```

### 批量运行多个任务

```python
from mteb import MTEB
from sentence_transformers import SentenceTransformer
from mteb.tasks.retrieval.eng import (
    TRECCOVIDCodeSwitching,
    AILAStatutesCodeSwitching,
    HagridRetrievalCodeSwitching,
    SCIDOCSCodeSwitching,
)

model = SentenceTransformer(
    "BAAI/bge-large-en-v1.5",
    model_kwargs={"attn_implementation": "flash_attention_2"},
    device="cuda"
)

tasks = [
    TRECCOVIDCodeSwitching(query_file="/path/to/treccovid_queries.jsonl"),
    AILAStatutesCodeSwitching(query_file="/path/to/aila_queries.jsonl"),
    HagridRetrievalCodeSwitching(query_file="/path/to/hagrid_queries.jsonl"),
    SCIDOCSCodeSwitching(query_file="/path/to/scidocs_queries.jsonl"),
]

evaluation = MTEB(tasks=tasks)
results = evaluation.run(model, output_folder="results/")
```

## 注意事项

1. **Flash Attention 要求**:
   - 安装: `pip install flash-attn --no-build-isolation`
   - GPU 算力 >= 8.0 (A100, RTX 3090, RTX 4090 等)

2. **HagridRetrieval 使用 `dev` split** 而非 `test`

3. **Query ID 必须与原始数据集中的 ID 匹配**，否则无法找到对应的 qrels
