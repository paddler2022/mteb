from __future__ import annotations

import json
import os

from datasets import Dataset, DatasetDict

from mteb.abstasks.pair_classification import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


def load_jsonl(filepath):
    """加载 JSONL 文件"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


class TwitterSemEval2015CodeSwitching(AbsTaskPairClassification):
    """
    TwitterSemEval2015 Code-Switching 变体任务
    - 完全从本地 jsonl 文件加载（sentence1, sentence2, labels）

    本地数据格式要求：
    {"sentence1": "...", "sentence2": "...", "labels": 0/1}
    """

    metadata = TaskMetadata(
        name="TwitterSemEval2015CodeSwitching",
        dataset={
            "path": "mteb/twittersemeval2015-pairclassification",
            "revision": "70970daeab8776df92f5ea462b6173c0b46fd2d1",
        },
        description="TwitterSemEval2015 Code-Switching variant. Paraphrase-Pairs of Tweets from the SemEval 2015 workshop. All data loaded from local file.",
        reference="https://alt.qcri.org/semeval2015/task1/",
        category="t2t",
        modalities=["text"],
        type="PairClassification",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=None,
        domains=["Social", "Written"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{xu-etal-2015-semeval,
  address = {Denver, Colorado},
  author = {Xu, Wei  and
Callison-Burch, Chris  and
Dolan, Bill},
  booktitle = {Proceedings of the 9th International Workshop on Semantic Evaluation ({S}em{E}val 2015)},
  doi = {10.18653/v1/S15-2001},
  editor = {Nakov, Preslav  and
Zesch, Torsten  and
Cer, Daniel  and
Jurgens, David},
  month = jun,
  pages = {1--11},
  publisher = {Association for Computational Linguistics},
  title = {{S}em{E}val-2015 Task 1: Paraphrase and Semantic Similarity in {T}witter ({PIT})},
  url = {https://aclanthology.org/S15-2001},
  year = {2015},
}
""",
        prompt="Retrieve tweets that are semantically similar to the given tweet",
    )

    def __init__(self, test_file: str = None, **kwargs):
        """
        初始化任务

        Args:
            test_file: test split 的本地 jsonl 文件路径。如果为 None，则从环境变量 TWITTER_SEMEVAL2015_TEST_FILE 获取
        """
        super().__init__(**kwargs)
        self.test_file = test_file or os.getenv("TWITTER_SEMEVAL2015_TEST_FILE")

    def load_data(self, **kwargs):
        """
        加载数据：完全从本地 jsonl 文件加载 sentence1, sentence2, labels
        """
        if self.data_loaded:
            return

        # ========== 1. 验证数据文件路径 ==========
        if not self.test_file:
            raise ValueError(
                "Data file path not provided. "
                "Please pass test_file parameter or set TWITTER_SEMEVAL2015_TEST_FILE environment variable."
            )

        if not os.path.exists(self.test_file):
            raise FileNotFoundError(f"Data file not found: {self.test_file}")

        # ========== 2. 从本地加载所有数据 ==========
        print(f"Loading data from local file: {self.test_file}")
        local_data = load_jsonl(self.test_file)

        # ========== 3. 构建数据集 ==========
        sentence1 = []
        sentence2 = []
        labels = []
        for idx, item in enumerate(local_data):
            try:
                sentence1.append(item['sentence1'])
                sentence2.append(item['sentence2'])
                labels.append(item['labels'])
            except KeyError as e:
                raise KeyError(f"Missing key {e} in item {idx}: {item}")

        self.dataset = DatasetDict({
            "test": Dataset.from_dict({
                "sentence1": sentence1,
                "sentence2": sentence2,
                "labels": labels,
            })
        })

        print(f"Loaded {len(sentence1)} samples for test split")
        self.data_loaded = True
