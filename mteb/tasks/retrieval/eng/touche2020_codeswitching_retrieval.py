from __future__ import annotations

import json
import os
from datasets import load_dataset
from tqdm import tqdm

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


def load_jsonl(filepath):
    """加载 JSONL 文件"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


class Touche2020CodeSwitching(AbsTaskRetrieval):
    """
    Touche2020 Code-Switching 变体任务
    - queries: 从本地 jsonl 文件加载（code-switching queries）
    - corpus 和 qrels: 从官方 HuggingFace 库加载

    本地数据格式要求：
    {"id": "1", "text": "query text here"}
    """

    metadata = TaskMetadata(
        name="Touche2020CodeSwitching",
        description="Touche2020 Code-Switching variant. Argument Retrieval for Controversial Questions. Queries are loaded from local file, corpus and qrels from official dataset.",
        reference="https://webis.de/events/touche-20/shared-task-1.html",
        dataset={
            "path": "mteb/touche2020",
            "revision": "a34f9a33db75fa0cbb21bb5cfc3dae8dc8bec93f",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2020-09-23", "2020-09-23"),
        domains=["Academic"],
        task_subtypes=["Question answering"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@dataset{potthast_2022_6862281,
  author = {Potthast, Martin and
Gienapp, Lukas and
Wachsmuth, Henning and
Hagen, Matthias and
Fröbe, Maik and
Bondarenko, Alexander and
Ajjour, Yamen and
Stein, Benno},
  doi = {10.5281/zenodo.6862281},
  month = jul,
  publisher = {Zenodo},
  title = {{Touché20-Argument-Retrieval-for-Controversial-
Questions}},
  url = {https://doi.org/10.5281/zenodo.6862281},
  year = {2022},
}
""",
        prompt={
            "query": "Given a question, retrieve detailed and persuasive arguments that answer the question"
        },
    )

    def __init__(self, query_file: str = None, **kwargs):
        """
        初始化任务

        Args:
            query_file: 本地 queries jsonl 文件路径。如果为 None，则从环境变量 TOUCHE2020_QUERY_FILE 获取
        """
        super().__init__(**kwargs)
        self.query_file = query_file or os.getenv("TOUCHE2020_QUERY_FILE")

    def load_data(self, **kwargs):
        """
        加载数据：
        - queries: 从本地 jsonl 文件
        - corpus 和 qrels: 从官方 HuggingFace 数据集
        """
        if self.data_loaded:
            return

        # ========== 1. 验证 query 文件路径 ==========
        if not self.query_file:
            raise ValueError(
                "Query file path not provided. "
                "Please pass query_file parameter or set TOUCHE2020_QUERY_FILE environment variable."
            )

        if not os.path.exists(self.query_file):
            raise FileNotFoundError(f"Query file not found: {self.query_file}")

        # ========== 2. 从本地加载 queries ==========
        print(f"Loading queries from local file: {self.query_file}")
        query_lines = load_jsonl(self.query_file)

        # ========== 3. 从官方库加载 corpus 和 qrels ==========
        dataset_path = self.metadata.dataset["path"]
        revision = self.metadata.dataset["revision"]

        print(f"Loading corpus from HuggingFace: {dataset_path}")
        corpus_dataset = load_dataset(dataset_path, "corpus", revision=revision)
        corpus_lines = list(corpus_dataset['corpus'])

        print(f"Loading qrels from HuggingFace: {dataset_path}")
        qrels_dataset = load_dataset(dataset_path, "default", revision=revision)
        qrels_lines = list(qrels_dataset['test'])

        # ========== 4. 初始化数据结构 ==========
        self.queries = {"test": {}}
        self.corpus = {"test": {}}
        self.relevant_docs = {"test": {}}

        # ========== 5. 填充 queries（从本地文件）==========
        for item in query_lines:
            qid = str(item.get('id') or item.get('_id'))
            text = item.get('text', '')
            self.queries["test"][qid] = text

        # ========== 6. 填充 corpus（从官方库）==========
        for item in tqdm(corpus_lines, desc="Loading corpus"):
            doc_id = str(item.get('_id') or item.get('id'))
            self.corpus["test"][doc_id] = {
                "title": item.get('title', ''),
                "text": item.get('text', '')
            }

        # ========== 7. 填充 relevant_docs（从官方库）==========
        for item in qrels_lines:
            qid = str(item.get('query-id'))
            doc_id = str(item.get('corpus-id'))
            score = int(item.get('score', 1))

            # 只加载在自定义 queries 中存在的 qrels
            if qid in self.queries["test"]:
                if qid not in self.relevant_docs["test"]:
                    self.relevant_docs["test"][qid] = {}
                self.relevant_docs["test"][qid][doc_id] = score

        # ========== 8. 统计信息 ==========
        print(f"Loaded {len(self.queries['test'])} queries")
        print(f"Loaded {len(self.corpus['test'])} documents")
        print(f"Loaded {len(self.relevant_docs['test'])} query-document relevance pairs")

        self.data_loaded = True


class Touche2020v3RetrievalCodeSwitching(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="Touche2020Retrieval.v3CodeSwitching",
        description="Touché Task 1: Argument Retrieval for Controversial Questions",
        reference="https://github.com/castorini/touche-error-analysis",
        dataset={
            "path": "mteb/webis-touche2020-v3",
            "revision": "431886eaecc48f067a3975b70d0949ea2862463c",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2020-09-23", "2020-09-23"),
        domains=["Academic"],
        task_subtypes=["Question answering"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{Thakur_etal_SIGIR2024,
  address_ = {Washington, D.C.},
  author = {Nandan Thakur and Luiz Bonifacio and Maik {Fr\"{o}be} and Alexander Bondarenko and Ehsan Kamalloo and Martin Potthast and Matthias Hagen and Jimmy Lin},
  booktitle = {Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  title = {Systematic Evaluation of Neural Retrieval Models on the {Touch\'{e}} 2020 Argument Retrieval Subset of {BEIR}},
  year = {2024},
}
""",
        adapted_from=["Touche2020"],
    )

    def __init__(self, query_file: str = None, ** kwargs):
        """
        初始化任务

        Args:
            query_file: 本地 queries jsonl 文件路径。如果为 None，则从环境变量 TOUCHE2020_QUERY_FILE 获取
        """
        super().__init__(**kwargs)
        self.query_file = query_file or os.getenv("TOUCHE2020V3_QUERY_FILE")


    def load_data(self, **kwargs):
        """
        加载数据：
        - queries: 从本地 jsonl 文件
        - corpus 和 qrels: 从官方 HuggingFace 数据集
        """
        if self.data_loaded:
            return

        # ========== 1. 验证 query 文件路径 ==========
        if not self.query_file:
            raise ValueError(
                "Query file path not provided. "
                "Please pass query_file parameter or set TOUCHE2020_QUERY_FILE environment variable."
            )

        if not os.path.exists(self.query_file):
            raise FileNotFoundError(f"Query file not found: {self.query_file}")

        # ========== 2. 从本地加载 queries ==========
        print(f"Loading queries from local file: {self.query_file}")
        query_lines = load_jsonl(self.query_file)

        # ========== 3. 从官方库加载 corpus 和 qrels ==========
        dataset_path = self.metadata.dataset["path"]
        revision = self.metadata.dataset["revision"]

        print(f"Loading corpus from HuggingFace: {dataset_path}")
        corpus_dataset = load_dataset(dataset_path, "corpus", revision=revision)
        corpus_lines = list(corpus_dataset['corpus'])

        print(f"Loading qrels from HuggingFace: {dataset_path}")
        qrels_dataset = load_dataset(dataset_path, "default", revision=revision)
        qrels_lines = list(qrels_dataset['test'])

        # ========== 4. 初始化数据结构 ==========
        self.queries = {"test": {}}
        self.corpus = {"test": {}}
        self.relevant_docs = {"test": {}}

        # ========== 5. 填充 queries（从本地文件）==========
        for item in query_lines:
            qid = str(item.get('id') or item.get('_id'))
            text = item.get('text', '')
            self.queries["test"][qid] = text

        # ========== 6. 填充 corpus（从官方库）==========
        for item in tqdm(corpus_lines, desc="Loading corpus"):
            doc_id = str(item.get('_id') or item.get('id'))
            self.corpus["test"][doc_id] = {
                "title": item.get('title', ''),
                "text": item.get('text', '')
            }

        # ========== 7. 填充 relevant_docs（从官方库）==========
        for item in qrels_lines:
            qid = str(item.get('query-id'))
            doc_id = str(item.get('corpus-id'))
            score = int(item.get('score', 1))

            # 只加载在自定义 queries 中存在的 qrels
            if qid in self.queries["test"]:
                if qid not in self.relevant_docs["test"]:
                    self.relevant_docs["test"][qid] = {}
                self.relevant_docs["test"][qid][doc_id] = score

        # ========== 8. 统计信息 ==========
        print(f"Loaded {len(self.queries['test'])} queries")
        print(f"Loaded {len(self.corpus['test'])} documents")
        print(f"Loaded {len(self.relevant_docs['test'])} query-document relevance pairs")

        self.data_loaded = True