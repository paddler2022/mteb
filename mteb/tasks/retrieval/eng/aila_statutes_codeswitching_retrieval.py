from __future__ import annotations

import json
import os
from datasets import load_dataset
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


class AILAStatutesCodeSwitching(AbsTaskRetrieval):
    """
    AILA Statutes Code-Switching Code-Switching Task
    - queries: queries(code switching version) jsonl file path
    - corpus and qrels: corpus and queries from huggingface
    """

    metadata = TaskMetadata(
        name="AILAStatutesCodeSwitching",
        description="AILA Statutes variant with code-switching queries. Corpus and qrels are loaded from the official dataset.",
        reference="https://zenodo.org/records/4063986",
        dataset={
            "path": "mteb/AILA_statutes",
            "revision": "ebfcd844eadd3d667efa3c57fc5c8c87f5c2867e",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=["Legal", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=None,
        sample_creation="found",
        bibtex_citation=r"""
@dataset{paheli_bhattacharya_2020_4063986,
  author = {Paheli Bhattacharya and
Kripabandhu Ghosh and
Saptarshi Ghosh and
Arindam Pal and
Parth Mehta and
Arnab Bhattacharya and
Prasenjit Majumder},
  doi = {10.5281/zenodo.4063986},
  month = oct,
  publisher = {Zenodo},
  title = {AILA 2019 Precedent \& Statute Retrieval Task},
  url = {https://doi.org/10.5281/zenodo.4063986},
  year = {2020},
}
""",
    )

    def __init__(self, query_file: str = None, **kwargs):
        """
        Initialization

        Args:
            query_file: local code switching queries jsonl file path
        """
        super().__init__(**kwargs)
        self.query_file = query_file or os.getenv("AILA_STATUTES_QUERY_FILE")

    def load_data(self, **kwargs):
        """
            Load data:
            - queries: queries(code switching version) jsonl file path
            - corpus and qrels: corpus and queries from huggingface
        """
        if self.data_loaded:
            return

        if not self.query_file:
            raise ValueError(
                "Query file path not provided. "
                "Please pass query_file parameter or set AILA_STATUTES_QUERY_FILE environment variable."
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
            score = int(item.get('score', 1))

            if qid in self.queries["test"]:
                if qid not in self.relevant_docs["test"]:
                    self.relevant_docs["test"][qid] = {}
                self.relevant_docs["test"][qid][doc_id] = score

        print(f"Loaded {len(self.queries['test'])} queries")
        print(f"Loaded {len(self.corpus['test'])} documents")
        print(f"Loaded {len(self.relevant_docs['test'])} query-document relevance pairs")

        self.data_loaded = True
