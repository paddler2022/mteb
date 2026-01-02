from __future__ import annotations

import json
import os
from datasets import Dataset

from mteb.abstasks.sts import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata


def load_jsonl(filepath):
    """加载 JSONL 文件"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


class STSBenchmarkCodeSwitching(AbsTaskSTS):
    """
    STSBenchmark Code-Switching 变体任务
    - 完全从本地 jsonl 文件加载（sentence1, sentence2, score）

    本地数据格式要求：
    {"sentence1": "...", "sentence2": "...", "score": 3.2}
    """

    metadata = TaskMetadata(
        name="STSBenchmarkCodeSwitching",
        description="STSBenchmark Code-Switching variant. Semantic Textual Similarity Benchmark dataset. Sentence pairs are loaded from local file.",
        reference="https://github.com/PhilipMay/stsb-multi-mt/",
        dataset={
            "path": "mteb/stsbenchmark-sts",
            "revision": "b0fddb56ed78048fa8b90373c8a3cfc37b684831",
        },
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cosine_spearman",
        date=None,
        domains=["Blog", "News", "Written"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="machine-translated and verified",
        bibtex_citation=r"""
@inproceedings{huggingface:dataset:stsb_multi_mt,
  author = {Philip May},
  title = {Machine translated multilingual STS benchmark dataset.},
  url = {https://github.com/PhilipMay/stsb-multi-mt},
  year = {2021},
}
""",
    )

    min_score = 0
    max_score = 5

    def __init__(self, data_file: str = None, **kwargs):
        """
        初始化任务

        Args:
            data_file: 本地数据 jsonl 文件路径。如果为 None，则从环境变量 STSBENCHMARK_DATA_FILE 获取
        """
        super().__init__(**kwargs)
        self.data_file = data_file or os.getenv("STSBENCHMARK_DATA_FILE")

    def load_data(self, **kwargs):
        """
        加载数据：完全从本地 jsonl 文件加载 sentence1, sentence2, score
        """
        if self.data_loaded:
            return

        # ========== 1. 验证数据文件路径 ==========
        if not self.data_file:
            raise ValueError(
                "Data file path not provided. "
                "Please pass data_file parameter or set STSBENCHMARK_DATA_FILE environment variable."
            )

        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"Data file not found: {self.data_file}")

        # ========== 2. 从本地加载数据 ==========
        print(f"Loading data from local file: {self.data_file}")
        local_data = load_jsonl(self.data_file)

        # ========== 3. 构建数据集 ==========
        sentences1 = []
        sentences2 = []
        scores = []

        for idx, item in enumerate(local_data):
            try:
                sentences1.append(item['sentence1'])
                sentences2.append(item['sentence2'])
                scores.append(item['score'])
            except KeyError as e:
                raise KeyError(f"Missing key {e} in item {idx}: {item}")

        self.dataset = {
            "test": Dataset.from_dict({
                "sentence1": sentences1,
                "sentence2": sentences2,
                "score": scores,
            })
        }

        print(f"Loaded {len(sentences1)} sentence pairs for test split")
        self.data_loaded = True