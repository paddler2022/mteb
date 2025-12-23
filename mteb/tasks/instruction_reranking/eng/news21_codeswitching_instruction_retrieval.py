from __future__ import annotations

import json
import os
from datasets import load_dataset, Dataset
from tqdm import tqdm

from mteb._evaluators.retrieval_metrics import evaluate_p_mrr_change
from mteb.abstasks import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


def load_jsonl(filepath):
    """加载 JSONL 文件"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


class News21InstructionRetrievalCodeSwitching(AbsTaskRetrieval):
    """
    News21 Instruction Retrieval Code-Switching 变体任务
    - queries: 从本地 jsonl 文件加载，使用分离的 query 和 instruction 字段
    - corpus, qrels 和 qrel_diff: 从官方 HuggingFace 库加载

    本地数据格式要求：
    {"id": "301", "query_og": "...", "query_changed": "...", "instruction_og": "...", "instruction_changed": "..."}

    数据处理流程与官方一致：
    - queries 包含 id, text 列
    - instructions 包含 query-id, instruction 列
    - 框架会自动用空格连接：text + " " + instruction

    注意：为了正确计算 p-MRR，需要同时加载 og 和 changed 版本，
    并为每个 query 创建两个条目：{qid}-og 和 {qid}-changed
    """

    metadata = TaskMetadata(
        name="News21InstructionRetrievalCodeSwitching",
        description="News21 Instruction Retrieval Code-Switching variant. Measuring retrieval instruction following ability on News21 narratives for the FollowIR benchmark.",
        reference="https://arxiv.org/abs/2403.15246",
        dataset={
            "path": "jhu-clsp/news21-instructions-mteb",
            "revision": "39db677749b3b783bb277d0e2d4712f5f133f52b",
        },
        type="InstructionReranking",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="p-MRR",
        date=("2023-08-01", "2024-04-01"),
        domains=["News", "Written"],
        task_subtypes=[],
        license="mit",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{weller2024followir,
  archiveprefix = {arXiv},
  author = {Orion Weller and Benjamin Chang and Sean MacAvaney and Kyle Lo and Arman Cohan and Benjamin Van Durme and Dawn Lawrie and Luca Soldaini},
  eprint = {2403.15246},
  primaryclass = {cs.IR},
  title = {FollowIR: Evaluating and Teaching Information Retrieval Models to Follow Instructions},
  year = {2024},
}
""",
    )

    def __init__(self, query_file: str = None, instruction_file: str = None, **kwargs):
        """
        初始化任务

        Args:
            query_file: 本地 queries jsonl 文件路径。如果为 None，则从环境变量 NEWS21_QUERY_FILE 获取
            instruction_file: 本地 instructions jsonl 文件路径。如果为 None，则从环境变量 NEWS21_INSTRUCTION_FILE 获取
        """
        super().__init__(**kwargs)
        self.query_file = query_file or os.getenv("NEWS21_QUERY_FILE")
        self.instruction_file = instruction_file or os.getenv("NEWS21_INSTRUCTIONS_FILE")

    def load_data(self, **kwargs):
        """
        加载数据：
        - queries 和 instructions: 从本地 jsonl 文件，分离加载
        - corpus, qrels 和 top_ranked: 从官方 HuggingFace 数据集

        本地数据格式：
        {"id": "301", "query_og": "...", "query_changed": "...", "instruction_og": "...", "instruction_changed": "..."}

        为了正确计算 p-MRR，每个 query 会创建两个条目：
        - {qid}-og: 使用 query_og + instruction_og
        - {qid}-changed: 使用 query_changed + instruction_changed
        """
        if self.data_loaded:
            return

        # ========== 1. 验证 query 文件路径 ==========
        if not self.query_file:
            raise ValueError(
                "Query file path not provided. "
                "Please pass query_file parameter or set NEWS21_QUERY_FILE environment variable."
            )

        if not os.path.exists(self.query_file):
            raise FileNotFoundError(f"Query file not found: {self.query_file}")

        # ========== 2. 从本地加载 queries ==========
        print(f"Loading queries from local file: {self.query_file}")
        print(f"Loading instructions from local file: {self.instruction_file}")
        query_lines = load_jsonl(self.query_file)
        instruction_lines = load_jsonl(self.instruction_file)
        # ========== 3. 从官方库加载 corpus, qrels 和 top_ranked ==========
        dataset_path = self.metadata.dataset["path"]
        revision = self.metadata.dataset["revision"]

        print(f"Loading corpus from HuggingFace: {dataset_path}")
        corpus_dataset = load_dataset(dataset_path, "corpus", revision=revision)
        corpus_lines = list(corpus_dataset['corpus'])

        print(f"Loading qrels from HuggingFace: {dataset_path}")
        qrels_dataset = load_dataset(dataset_path, "default", revision=revision)
        qrels_lines = list(qrels_dataset['test'])

        print(f"Loading top_ranked from HuggingFace: {dataset_path}")
        top_ranked_dataset = load_dataset(dataset_path, "top_ranked", revision=revision)
        top_ranked_lines = list(top_ranked_dataset['top_ranked'])

        # ========== 4. 初始化数据结构 ==========
        # 使用分离的 queries 和 instructions，与官方流程一致
        self.queries = {"test": {}}
        self.instructions = {"test": {}}  # 新增：单独的 instructions
        self.corpus = {"test": {}}
        self.relevant_docs = {"test": {}}
        self.top_ranked = {"test": {}}

        # ========== 5. 填充 queries 和 instructions（从本地文件）==========
        for item in tqdm(query_lines, desc="Loading queries"):
            qid = str(item.get('_id'))
            self.queries["test"][qid] = item.get('text')

        for item in tqdm(instruction_lines, desc="Loading queries"):
            qid = str(item.get('query-id'))
            self.instructions["test"][qid] = item.get('instruction')

        # ========== 6. 填充 corpus（从官方库）==========
        for item in tqdm(corpus_lines, desc="Loading corpus"):
            doc_id = str(item.get('_id') or item.get('id'))
            self.corpus["test"][doc_id] = {
                "title": item.get('title'),
                "text": item.get('text')
            }

        # ========== 7. 填充 relevant_docs（从官方库）==========
        # 注意：官方 qrels 的 query-id 已经带有 -og/-changed 后缀
        # 例如 "310-og", "310-changed"，所以直接使用即可
        for item in tqdm(qrels_lines, desc="Loading qrels"):
            qid = str(item.get('query-id'))  # 已经带有 -og/-changed 后缀
            doc_id = str(item.get('corpus-id'))
            score = int(item.get('score', 1))  # 转换为整数

            # 直接使用 qid（已经带有后缀），不需要再添加
            if qid in self.queries["test"]:
                if qid not in self.relevant_docs["test"]:
                    self.relevant_docs["test"][qid] = {}
                self.relevant_docs["test"][qid][doc_id] = score

        # ========== 8. 填充 top_ranked（从官方库）==========
        # 官方 top_ranked 的 query-id 已经带有 -og/-changed 后缀
        for item in tqdm(top_ranked_lines, desc="Loading top_ranked"):
            qid = str(item.get('query-id'))
            corpus_ids = item.get('corpus-ids', [])
            if qid in self.queries["test"]:
                self.top_ranked["test"][qid] = corpus_ids

        # ========== 9. 统计信息 ==========
        print(f"Loaded {len(self.queries['test'])} queries (og + changed)")
        print(f"Loaded {len(self.instructions['test'])} instructions (og + changed)")
        print(f"Loaded {len(self.corpus['test'])} documents")
        print(f"Loaded {len(self.relevant_docs['test'])} query-document relevance pairs")
        print(f"Loaded {len(self.top_ranked['test'])} top_ranked entries")

        self.data_loaded = True

    def task_specific_scores(
        self,
        scores: dict[str, dict[str, float]],
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        hf_split: str,
        hf_subset: str,
    ) -> dict[str, float]:
        qrel_diff_ds = load_dataset(
            self.metadata.dataset["path"],
            "qrel_diff",
            split="qrel_diff",
            revision=self.metadata.dataset["revision"],
        )
        changed_qrels = {item["query-id"]: item["corpus-ids"] for item in qrel_diff_ds}

        return evaluate_p_mrr_change(
            qrels,
            results,
            changed_qrels,
            self.k_values,
        )
