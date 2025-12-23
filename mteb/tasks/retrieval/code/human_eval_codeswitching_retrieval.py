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


class HumanEvalRetrievalCodeSwitching(AbsTaskRetrieval):
    """
    HumanEval Code-Switching 变体任务
    - queries: 从本地 jsonl 文件加载（code-switching queries）
    - corpus 和 qrels: 从官方 HuggingFace 库加载

    本地数据格式要求：
    {"id": "HumanEval/0", "text": "query text here"}
    """

    metadata = TaskMetadata(
        name="HumanEvalRetrievalCodeSwitching",
        description="HumanEval Code-Switching variant. A code retrieval task based on Python programming problems. Queries are loaded from local file, corpus and qrels from official dataset.",
        reference="https://huggingface.co/datasets/embedding-benchmark/HumanEval",
        dataset={
            "path": "embedding-benchmark/HumanEval",
            "revision": "ed1f48aca747f10bac146795328e2f03326e7625",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn", "python-Code"],
        main_score="ndcg_at_10",
        date=("2021-01-01", "2021-12-31"),
        domains=["Programming"],
        task_subtypes=["Code retrieval"],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@article{chen2021evaluating,
  archiveprefix = {arXiv},
  author = {Chen, Mark and Tworek, Jerry and Jun, Heewoo and Yuan, Qiming and Pinto, Henrique Ponde de Oliveira and Kaplan, Jared and Edwards, Harri and Burda, Yuri and Joseph, Nicholas and Brockman, Greg and Ray, Alex and Puri, Raul and Krueger, Gretchen and Petrov, Michael and Khlaaf, Heidy and Sastry, Girish and Mishkin, Pamela and Chan, Brooke and Gray, Scott and Ryder, Nick and Pavlov, Mikhail and Power, Alethea and Kaiser, Lukasz and Bavarian, Mohammad and Winter, Clemens and Tillet, Philippe and Such, Felipe Petroski and Cummings, Dave and Plappert, Matthias and Chantzis, Fotios and Barnes, Elizabeth and Herbert-Voss, Ariel and Guss, William Hebgen and Nichol, Alex and Paino, Alex and Tezak, Nikolas and Tang, Jie and Babuschkin, Igor and Balaji, Suchir and Jain, Shantanu and Saunders, William and Hesse, Christopher and Carr, Andrew N. and Leike, Jan and Achiam, Joshua and Misra, Vedant and Morikawa, Evan and Radford, Alec and Knight, Matthew and Brundage, Miles and Murati, Mira and Mayer, Katie and Welinder, Peter and McGrew, Bob and Amodei, Dario and McCandlish, Sam and Sutskever, Ilya and Zaremba, Wojciech},
  eprint = {2107.03374},
  primaryclass = {cs.LG},
  title = {Evaluating Large Language Models Trained on Code},
  year = {2021},
}""",
    )

    def __init__(self, query_file: str = None, **kwargs):
        """
        初始化任务

        Args:
            query_file: 本地 queries jsonl 文件路径。如果为 None，则从环境变量 HUMANEVAL_QUERY_FILE 获取
        """
        super().__init__(**kwargs)
        self.query_file = query_file or os.getenv("HUMANEVAL_QUERY_FILE")

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
                "Please pass query_file parameter or set HUMANEVAL_QUERY_FILE environment variable."
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
        corpus_ds = load_dataset(dataset_path, "corpus", revision=revision)["corpus"]

        print(f"Loading qrels from HuggingFace: {dataset_path}")
        qrels_ds = load_dataset(dataset_path, "default", revision=revision)["test"]

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
        for item in tqdm(corpus_ds, desc="Loading corpus"):
            doc_id = str(item.get('id') or item.get('_id'))
            self.corpus["test"][doc_id] = {
                "title": "",
                "text": item.get('text', '')
            }

        # ========== 7. 填充 relevant_docs（从官方库）==========
        for item in qrels_ds:
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
