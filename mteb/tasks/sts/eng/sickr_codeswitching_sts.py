from __future__ import annotations

import json
import os
from datasets import Dataset

from mteb.abstasks.sts import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata


def load_jsonl(filepath):
    """加载 JSONL 文件"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


class SickrCodeSwitchingSTS(AbsTaskSTS):
    """
    SICK-R Code-Switching 变体任务
    - 完全从本地 jsonl 文件加载（sentence1, sentence2, score）

    本地数据格式要求：
    {"sentence1": "...", "sentence2": "...", "score": 3.2}
    """

    metadata = TaskMetadata(
        name="SICK-RCodeSwitching",
        description="SICK-R Code-Switching variant. Semantic Textual Similarity SICK-R dataset. Sentence pairs are loaded from local file, scores from official dataset.",
        reference="https://aclanthology.org/L14-1314/",
        dataset={
            "path": "mteb/sickr-sts",
            "revision": "20a6d6f312dd54037fe07a32d58e5e168867909d",
        },
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cosine_spearman",
        date=None,
        domains=["Web", "Written"],
        task_subtypes=["Textual Entailment"],
        license="cc-by-nc-sa-3.0",
        annotations_creators="human-annotated",
        dialect=None,
        sample_creation=None,
        bibtex_citation=r"""
@inproceedings{marelli-etal-2014-sick,
  address = {Reykjavik, Iceland},
  author = {Marelli, Marco  and
Menini, Stefano  and
Baroni, Marco  and
Bentivogli, Luisa  and
Bernardi, Raffaella  and
Zamparelli, Roberto},
  booktitle = {Proceedings of the Ninth International Conference on Language Resources and Evaluation ({LREC}'14)},
  editor = {Calzolari, Nicoletta  and
Choukri, Khalid  and
Declerck, Thierry  and
Loftsson, Hrafn  and
Maegaard, Bente  and
Mariani, Joseph  and
Moreno, Asuncion  and
Odijk, Jan  and
Piperidis, Stelios},
  month = may,
  pages = {216--223},
  publisher = {European Language Resources Association (ELRA)},
  title = {A {SICK} cure for the evaluation of compositional distributional semantic models},
  url = {http://www.lrec-conf.org/proceedings/lrec2014/pdf/363_Paper.pdf},
  year = {2014},
}
""",
    )

    min_score = 0
    max_score = 5

    def __init__(self, data_file: str = None, **kwargs):
        """
        初始化任务

        Args:
            data_file: 本地数据 jsonl 文件路径。如果为 None，则从环境变量 SICKR_DATA_FILE 获取
        """
        super().__init__(**kwargs)
        self.data_file = data_file or os.getenv("SICKR_DATA_FILE")

    def load_data(self, **kwargs):
        """
        加载数据：完全从本地 jsonl 文件加载 sentence1, sentence2, score
        """
        if self.data_loaded:
            return

        # ========== 1. 验证数据文件路径 ==========
        if not self.data_file:
            raise ValueError(
                "Data file path not provided. "
                "Please pass data_file parameter or set SICKR_DATA_FILE environment variable."
            )

        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"Data file not found: {self.data_file}")

        # ========== 2. 从本地加载数据 ==========
        print(f"Loading data from local file: {self.data_file}")
        local_data = load_jsonl(self.data_file)

        # ========== 3. 构建数据集 ==========
        sentences1 = []
        sentences2 = []
        scores = []

        for item in local_data:
            sentences1.append(item['sentence1'])
            sentences2.append(item['sentence2'])
            scores.append(item['score'])

        self.dataset = {
            "test": Dataset.from_dict({
                "sentence1": sentences1,
                "sentence2": sentences2,
                "score": scores,
            })
        }

        print(f"Loaded {len(sentences1)} sentence pairs for test split")
        self.data_loaded = True
