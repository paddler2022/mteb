from __future__ import annotations

import json
import os

from datasets import Dataset, DatasetDict

from mteb.abstasks.clustering import AbsTaskClustering
from mteb.abstasks.task_metadata import TaskMetadata

N_SAMPLES = 2048


def load_jsonl(filepath):
    """加载 JSONL 文件"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def split_labels(record: dict) -> dict:
    """将层级标签分割成列表，例如 'quant-ph.cs' -> ['quant-ph', 'cs']"""
    record["labels"] = record["labels"].split(".")
    return record


class ArXivHierarchicalClusteringP2PCodeSwitching(AbsTaskClustering):
    """
    ArXivHierarchicalClusteringP2P Code-Switching 变体任务
    - 完全从本地 jsonl 文件加载（sentences 和 labels）
    - 每行一个样本：{"sentences": "text...", "labels": "quant-ph"}
    """

    metadata = TaskMetadata(
        name="ArXivHierarchicalClusteringP2PCodeSwitching",
        description="ArXivHierarchicalClusteringP2P Code-Switching variant. Clustering of titles+abstract from arxiv. All data loaded from local file.",
        reference="https://www.kaggle.com/Cornell-University/arxiv",
        dataset={
            "path": "mteb/arxiv-clustering-p2p",
            "revision": "0bbdb47bcbe3a90093699aefeed338a0f28a7ee8",
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("1991-01-01", "2021-01-01"),
        domains=["Academic", "Written"],
        task_subtypes=[],
        license="cc0-1.0",
        annotations_creators="derived",
        dialect=["Thematic clustering"],
        sample_creation="found",
        bibtex_citation="",
    )

    def __init__(self, test_file: str = None, **kwargs):
        """
        初始化任务

        Args:
            test_file: test split 的本地 jsonl 文件路径。如果为 None，则从环境变量 ARXIV_P2P_TEST_FILE 获取
        """
        super().__init__(**kwargs)
        self.test_file = test_file or os.getenv("ARXIV_P2P_TEST_FILE")

    def load_data(self, **kwargs):
        """
        加载数据：完全从本地 jsonl 文件加载 sentences 和 labels
        每行一个样本格式：{"sentences": "text...", "labels": "quant-ph"}
        """
        if self.data_loaded:
            return

        # ========== 1. 验证数据文件路径 ==========
        if not self.test_file:
            raise ValueError(
                "Data file path not provided. "
                "Please pass test_file parameter or set ARXIV_P2P_TEST_FILE environment variable."
            )

        if not os.path.exists(self.test_file):
            raise FileNotFoundError(f"Data file not found: {self.test_file}")

        # ========== 2. 从本地加载所有数据 ==========
        print(f"Loading data from local file: {self.test_file}")
        local_data = load_jsonl(self.test_file)

        # ========== 3. 构建数据集（每行一个样本）==========
        sentences = []
        labels = []
        for idx, item in enumerate(local_data):
            try:
                sentences.append(item['sentences'])
                labels.append(item['labels'])
            except KeyError as e:
                raise KeyError(f"Missing key {e} in item {idx}: {item}")

        self.dataset = DatasetDict({
            "test": Dataset.from_dict({
                "sentences": sentences,
                "labels": labels,
            })
        })

        # 分割层级标签
        self.dataset = self.dataset.map(split_labels)

        # 采样
        if len(self.dataset["test"]) > N_SAMPLES:
            self.dataset["test"] = self.dataset["test"].train_test_split(
                test_size=N_SAMPLES, seed=self.seed
            )["test"]

        print(f"Loaded {len(self.dataset['test'])} samples for test split")
        self.data_loaded = True


class ArXivHierarchicalClusteringS2SCodeSwitching(AbsTaskClustering):
    """
    ArXivHierarchicalClusteringS2S Code-Switching 变体任务
    - 完全从本地 jsonl 文件加载（sentences 和 labels）
    - 每行一个样本：{"sentences": "text...", "labels": "quant-ph"}
    """

    metadata = TaskMetadata(
        name="ArXivHierarchicalClusteringS2SCodeSwitching",
        description="ArXivHierarchicalClusteringS2S Code-Switching variant. Clustering of titles from arxiv. All data loaded from local file.",
        reference="https://www.kaggle.com/Cornell-University/arxiv",
        dataset={
            "path": "mteb/arxiv-clustering-s2s",
            "revision": "b73bd54100e5abfa6e3a23dcafb46fe4d2438dc3",
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("1991-01-01", "2021-01-01"),
        domains=["Academic", "Written"],
        task_subtypes=["Thematic clustering"],
        license="cc0-1.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
    )

    def __init__(self, test_file: str = None, **kwargs):
        """
        初始化任务

        Args:
            test_file: test split 的本地 jsonl 文件路径。如果为 None，则从环境变量 ARXIV_S2S_TEST_FILE 获取
        """
        super().__init__(**kwargs)
        self.test_file = test_file or os.getenv("ARXIV_S2S_TEST_FILE")

    def load_data(self, **kwargs):
        """
        加载数据：完全从本地 jsonl 文件加载 sentences 和 labels
        每行一个样本格式：{"sentences": "text...", "labels": "quant-ph"}
        """
        if self.data_loaded:
            return

        # ========== 1. 验证数据文件路径 ==========
        if not self.test_file:
            raise ValueError(
                "Data file path not provided. "
                "Please pass test_file parameter or set ARXIV_S2S_TEST_FILE environment variable."
            )

        if not os.path.exists(self.test_file):
            raise FileNotFoundError(f"Data file not found: {self.test_file}")

        # ========== 2. 从本地加载所有数据 ==========
        print(f"Loading data from local file: {self.test_file}")
        local_data = load_jsonl(self.test_file)

        # ========== 3. 构建数据集（每行一个样本）==========
        sentences = []
        labels = []
        for idx, item in enumerate(local_data):
            try:
                sentences.append(item['sentences'])
                labels.append(item['labels'])
            except KeyError as e:
                raise KeyError(f"Missing key {e} in item {idx}: {item}")

        self.dataset = DatasetDict({
            "test": Dataset.from_dict({
                "sentences": sentences,
                "labels": labels,
            })
        })

        # 分割层级标签
        self.dataset = self.dataset.map(split_labels)

        # 采样
        if len(self.dataset["test"]) > N_SAMPLES:
            self.dataset["test"] = self.dataset["test"].train_test_split(
                test_size=N_SAMPLES, seed=self.seed
            )["test"]

        print(f"Loaded {len(self.dataset['test'])} samples for test split")
        self.data_loaded = True
