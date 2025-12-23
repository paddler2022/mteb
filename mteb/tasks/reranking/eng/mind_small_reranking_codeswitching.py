from __future__ import annotations

import json
import logging
import os
from datasets import load_dataset
from tqdm import tqdm

from mteb._evaluators.retrieval_metrics import max_over_subqueries
from mteb.abstasks import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


logger = logging.getLogger(__name__)


def load_jsonl(filepath):

    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


class MindSmallRerankingCodeSwitching(AbsTaskRetrieval):
    """
    MindSmallReranking Code-Switching Task
    """

    metadata = TaskMetadata(
        name="MindSmallRerankingCodeSwitching",
        description="MindSmallReranking Code-Switching variant. Microsoft News Dataset: A Large-Scale English Dataset for News Recommendation Research. Corpus and qrels are loaded from the official dataset.",
        reference="https://msnews.github.io/assets/doc/ACL2020_MIND.pdf",
        dataset={
            "path": "mteb/MindSmallReranking",
            "revision": "227478e3235572039f4f7661840e059f31ef6eb1",
        },
        type="Reranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_over_subqueries_map_at_1000",
        date=("2019-10-12", "2019-11-22"),
        domains=["News", "Written"],
        task_subtypes=[],
        license="https://github.com/msnews/MIND/blob/master/MSR%20License_Data.pdf",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        prompt="Retrieve relevant news articles based on user browsing history",
        bibtex_citation=r"""
@inproceedings{wu-etal-2020-mind,
  address = {Online},
  author = {Wu, Fangzhao  and Qiao, Ying  and Chen, Jiun-Hung  and Wu, Chuhan  and Qi,
Tao  and Lian, Jianxun  and Liu, Danyang  and Xie, Xing  and Gao, Jianfeng  and Wu, Winnie  and Zhou, Ming},
  booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  doi = {10.18653/v1/2020.acl-main.331},
  editor = {Jurafsky, Dan  and Chai, Joyce  and Schluter, Natalie  and Tetreault, Joel},
  month = jul,
  pages = {3597--3606},
  publisher = {Association for Computational Linguistics},
  title = {{MIND}: A Large-scale Dataset for News
Recommendation},
  url = {https://aclanthology.org/2020.acl-main.331},
  year = {2020},
}
""",
    )

    def __init__(self, query_file: str = None, **kwargs):

        super().__init__(**kwargs)
        self.query_file = query_file or os.getenv("MINDSMALL_QUERY_FILE")

    def load_data(self, **kwargs):
        """
        Load data:
        - queries: from local file
        - corpus and qrels: from Huggingface
        """
        if self.data_loaded:
            return

        if not self.query_file:
            raise ValueError(
                "Query file path not provided. "
                "Please pass query_file parameter or set MINDSMALL_QUERY_FILE environment variable."
            )

        if not os.path.exists(self.query_file):
            raise FileNotFoundError(f"Query file not found: {self.query_file}")

        print(f"Loading queries from local file: {self.query_file}")
        query_lines = load_jsonl(self.query_file)

        dataset_path = self.metadata.dataset["path"]
        revision = self.metadata.dataset["revision"]

        print(f"Loading corpus from HuggingFace: {dataset_path}")
        corpus_dataset = load_dataset(dataset_path, "corpus", revision=revision)
        corpus_lines = list(corpus_dataset['corpus'])

        print(f"Loading qrels from HuggingFace: {dataset_path}")
        qrels_dataset = load_dataset(dataset_path, "default", revision=revision)
        qrels_lines = list(qrels_dataset['test'])

        self.queries = {"test": {}}
        self.corpus = {"test": {}}
        self.relevant_docs = {"test": {}}

        for item in tqdm(query_lines, desc="Loading queries"):
            qid = str(item.get('id') or item.get('_id'))
            text = item.get('text', '')
            self.queries["test"][qid] = text

        for item in tqdm(corpus_lines, desc="Loading corpus"):
            doc_id = str(item.get('_id') or item.get('id'))
            self.corpus["test"][doc_id] = {
                "title": item.get('title', ''),
                "text": item.get('text', '')
            }

        for item in tqdm(qrels_lines, desc="Loading qrels"):
            qid = str(item.get('query-id'))
            doc_id = str(item.get('corpus-id'))
            score = item.get('score', 1)

            if qid in self.queries["test"]:
                if qid not in self.relevant_docs["test"]:
                    self.relevant_docs["test"][qid] = {}
                self.relevant_docs["test"][qid][doc_id] = score

        print(f"Loaded {len(self.queries['test'])} queries")
        print(f"Loaded {len(self.corpus['test'])} documents")
        print(f"Loaded {len(self.relevant_docs['test'])} query-document relevance pairs")

        self.data_loaded = True

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        return max_over_subqueries(qrels, results, self.k_values)
