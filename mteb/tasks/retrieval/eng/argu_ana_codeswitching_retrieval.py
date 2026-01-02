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


class ArguAnaCodeSwitching(AbsTaskRetrieval):
    """
    ArguAna Code-Switching 变体任务
    - queries: 从本地 jsonl 文件加载（code-switching queries）
    - corpus 和 qrels: 从官方 HuggingFace 库加载
    """

    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="ArguAnaCodeSwitching",
        description="ArguAna Code-Switching variant. Retrieval of the Best Counterargument without Prior Topic Knowledge. Corpus and qrels are loaded from the official dataset.",
        reference="http://argumentation.bplaced.net/arguana/data",
        dataset={
            "path": "mteb/arguana",
            "revision": "c22ab2a51041ffd869aaddef7af8d8215647e41a",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=["2018-01-01", "2018-07-01"],
        domains=["Social", "Web", "Written"],
        task_subtypes=["Discourse coherence"],
        license="cc-by-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{wachsmuth2018retrieval,
  author = {Wachsmuth, Henning and Syed, Shahbaz and Stein, Benno},
  booktitle = {ACL},
  title = {Retrieval of the Best Counterargument without Prior Topic Knowledge},
  year = {2018},
}
""",
        prompt={"query": "Given a claim, find documents that refute the claim"},
    )

    def __init__(self, query_file: str = None, **kwargs):
        """
        初始化任务

        Args:
            query_file: 本地 queries jsonl 文件路径。如果为 None，则从环境变量 ARGUANA_QUERY_FILE 获取
        """
        super().__init__(**kwargs)
        self.query_file = query_file or os.getenv("ARGUANA_QUERY_FILE")

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
                "Please pass query_file parameter or set ARGUANA_QUERY_FILE environment variable."
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
        for idx, item in enumerate(tqdm(query_lines, desc="Loading queries")):
            try:
                qid = str(item.get('_id') or item['id'])
                text = item['text']
                self.queries["test"][qid] = text
            except KeyError as e:
                raise KeyError(f"Missing key {e} in query item {idx}: {item}")

        # ========== 6. 填充 corpus（从官方库）==========
        for item in tqdm(corpus_lines, desc="Loading corpus"):
            doc_id = str(item.get('_id') or item.get('id'))
            self.corpus["test"][doc_id] = {
                "title": item.get('title', ''),
                "text": item.get('text', '')
            }

        # ========== 7. 填充 relevant_docs（从官方库）==========
        for item in tqdm(qrels_lines, desc="Loading qrels"):
            qid = str(item.get('query-id'))
            doc_id = str(item.get('corpus-id'))
            score = int(item.get('score'))

            if qid in self.queries["test"]:
                if qid not in self.relevant_docs["test"]:
                    self.relevant_docs["test"][qid] = {}
                self.relevant_docs["test"][qid][doc_id] = score

        # ========== 8. 统计信息 ==========
        print(f"Loaded {len(self.queries['test'])} queries")
        print(f"Loaded {len(self.corpus['test'])} documents")
        print(f"Loaded {len(self.relevant_docs['test'])} query-document relevance pairs")

        self.data_loaded = True
