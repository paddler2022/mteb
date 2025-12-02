from __future__ import annotations

import json
import os
from datasets import load_dataset, Dataset
from tqdm import tqdm

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


def load_jsonl(filepath):
    """Load json file"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


class TRECCOVIDCodeSwitching(AbsTaskRetrieval):
    """
    TREC-COVID Code-Switching Customed Task
    - queries: queries(code switching version) jsonl file path
    - corpus and qrels: corpus and queries from huggingface
    """

    metadata = TaskMetadata(
        name="TRECCOVIDCodeSwitching",
        description="TREC-COVID variant with code-switching queries. Corpus and qrels are loaded from the official dataset.",
        reference="https://ir.nist.gov/covidSubmit/index.html",
        dataset={
            "path": "mteb/trec-covid",
            "revision": "bb9466bac8153a0349341eb1b22e06409e78ef4e",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=["Medical", "Academic", "Written"],
        task_subtypes=["Question answering"],
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=r"""
@misc{roberts2021searching,
  archiveprefix = {arXiv},
  author = {Kirk Roberts and Tasmeer Alam and Steven Bedrick and Dina Demner-Fushman and Kyle Lo and Ian Soboroff and Ellen Voorhees and Lucy Lu Wang and William R Hersh},
  eprint = {2104.09632},
  primaryclass = {cs.IR},
  title = {Searching for Scientific Evidence in a Pandemic: An Overview of TREC-COVID},
  year = {2021},
}
""",
        prompt={
            "query": "Given a query on COVID-19, retrieve documents that answer the query"
        },
    )

    def __init__(self, query_file: str = None, **kwargs):
        """
        Initialization

        Args:
            query_file: local code switching queries jsonl file path
        """
        super().__init__(**kwargs)
        self.query_file = query_file or os.getenv("TRECCOVID_QUERY_FILE")

    def load_data(self, **kwargs):
        """
        Load data:
        - queries: queries(code switching version) jsonl file path
        - corpus and qrels: corpus and queries from huggingface
        """
        if self.data_loaded:
            return

        # ========== check whether query file path is valid ==========
        if not self.query_file:
            raise ValueError(
                "Query file path not provided. "
                "Please pass query_file parameter or set TRECCOVID_QUERY_FILE environment variable."
            )

        if not os.path.exists(self.query_file):
            raise FileNotFoundError(f"Query file not found: {self.query_file}")

        # ========== load queries from local file ==========
        print(f"Loading queries from local file: {self.query_file}")
        query_lines = load_jsonl(self.query_file)

        # ========== Load corpus and qrels from hf ==========
        dataset_path = self.metadata.dataset["path"]
        revision = self.metadata.dataset["revision"]

        print(f"Loading corpus from HuggingFace: {dataset_path}")
        corpus_dataset = load_dataset(dataset_path, "corpus", revision=revision)
        corpus_lines = list(corpus_dataset['corpus'])

        print(f"Loading qrels from HuggingFace: {dataset_path}")
        qrels_dataset = load_dataset(dataset_path, "default", revision=revision)
        qrels_lines = list(qrels_dataset['test'])

        # ========== data preparation ==========
        self.queries = {"test": {}}
        self.corpus = {"test": {}}
        self.relevant_docs = {"test": {}}

        # ========== modify data structure==========
        # Support 2 formats
        # Format 1: {"id": "q1", "text": "query text"}
        # Format 2: {"_id": "q1", "text": "query text"}
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

        # qrels format: {"query-id": "q1", "corpus-id": "doc1", "score": 1}
        for item in tqdm(qrels_lines, desc="Loading qrels"):
            qid = str(item.get('query-id'))
            doc_id = str(item.get('corpus-id'))
            score = int(item.get('score', 1))

            # Load qrels only existing in queries_code_switching
            if qid in self.queries["test"]:
                if qid not in self.relevant_docs["test"]:
                    self.relevant_docs["test"][qid] = {}
                self.relevant_docs["test"][qid][doc_id] = score

        # ========== 8. 统计信息 ==========
        print(f"Loaded {len(self.queries['test'])} queries from local file")
        print(f"Loaded {len(self.corpus['test'])} documents")
        print(f"Loaded {len(self.relevant_docs['test'])} query-document relevance pairs")

        self.data_loaded = True
