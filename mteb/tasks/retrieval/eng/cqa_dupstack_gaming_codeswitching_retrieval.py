from __future__ import annotations

import json
import os
from datasets import load_dataset
from tqdm import tqdm

from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


def load_jsonl(filepath):

    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


class CQADupstackGamingRetrievalCodeSwitching(AbsTaskRetrieval):

    metadata = TaskMetadata(
        name="CQADupstackGamingRetrievalCodeSwitching",
        description="CQADupstackGamingRetrieval Code-Switching variant. CQADupStack: A Benchmark Data Set for Community Question-Answering Research. Corpus and qrels are loaded from the official dataset.",
        reference="http://nlp.cis.unimelb.edu.au/resources/cqadupstack/",
        dataset={
            "path": "mteb/cqadupstack-gaming",
            "revision": "4885aa143210c98657558c04aaf3dc47cfb54340",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=None,
        domains=["Web", "Written"],
        task_subtypes=["Question answering", "Duplicate Detection"],
        license="apache-2.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{hoogeveen2015,
  acmid = {2838934},
  address = {New York, NY, USA},
  articleno = {3},
  author = {Hoogeveen, Doris and Verspoor, Karin M. and Baldwin, Timothy},
  booktitle = {Proceedings of the 20th Australasian Document Computing Symposium (ADCS)},
  doi = {10.1145/2838931.2838934},
  isbn = {978-1-4503-4040-3},
  location = {Parramatta, NSW, Australia},
  numpages = {8},
  pages = {3:1--3:8},
  publisher = {ACM},
  series = {ADCS '15},
  title = {CQADupStack: A Benchmark Data Set for Community Question-Answering Research},
  url = {http://doi.acm.org/10.1145/2838931.2838934},
  year = {2015},
}
""",
    )

    def __init__(self, query_file: str = None, **kwargs):

        super().__init__(**kwargs)
        self.query_file = query_file or os.getenv("CQADUPSTACK_GAMING_QUERY_FILE")

    def load_data(self, **kwargs):

        if self.data_loaded:
            return

        if not self.query_file:
            raise ValueError(
                "Query file path not provided. "
                "Please pass query_file parameter or set CQADUPSTACK_GAMING_QUERY_FILE environment variable."
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
