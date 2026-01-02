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


class ClimateFEVERHardNegativesV2CodeSwitching(AbsTaskRetrieval):
    """
    ClimateFEVER HardNegatives V2 Code-Switching 变体任务
    - queries: 从本地 jsonl 文件加载（code-switching queries）
    - corpus 和 qrels: 从官方 HuggingFace 库加载
    """

    metadata = TaskMetadata(
        name="ClimateFEVERHardNegatives.v2CodeSwitching",
        description=(
            "ClimateFEVER HardNegatives V2 Code-Switching variant. "
            "CLIMATE-FEVER is a dataset adopting the FEVER methodology that consists of 1,535 real-world claims regarding climate-change. "
            "The hard negative version has been created by pooling the 250 top documents per query from BM25, e5-multilingual-large and e5-mistral-instruct. "
            "V2 uses a more appropriate prompt rather than the default prompt for retrieval. "
            "Corpus and qrels are loaded from the official dataset."
        ),
        reference="https://www.sustainablefinance.uzh.ch/en/research/climate-fever.html",
        dataset={
            "path": "mteb/ClimateFEVER_test_top_250_only_w_correct-v2",
            "revision": "3a309e201f3c2c4b13bd4a367a8f37eee2ec1d21",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2001-01-01", "2020-12-31"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Claim verification"],
        license="cc-by-sa-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        adapted_from=["ClimateFEVER"],
        bibtex_citation=r"""
@misc{diggelmann2021climatefever,
  archiveprefix = {arXiv},
  author = {Thomas Diggelmann and Jordan Boyd-Graber and Jannis Bulian and Massimiliano Ciaramita and Markus Leippold},
  eprint = {2012.00614},
  primaryclass = {cs.CL},
  title = {CLIMATE-FEVER: A Dataset for Verification of Real-World Climate Claims},
  year = {2021},
}
""",
        prompt={
            "query": "Given a claim about climate change, retrieve documents that support or refute the claim"
        },
    )

    def __init__(self, query_file: str = None, **kwargs):
        """
        初始化任务

        Args:
            query_file: 本地 queries jsonl 文件路径。如果为 None，则从环境变量 CLIMATEFEVER_QUERY_FILE 获取
        """
        super().__init__(**kwargs)
        self.query_file = query_file or os.getenv("CLIMATEFEVER_QUERY_FILE")

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
                "Please pass query_file parameter or set CLIMATEFEVER_QUERY_FILE environment variable."
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
        corpus_lines = list(corpus_dataset['test'])

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
