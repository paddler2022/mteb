from __future__ import annotations

import json
import os
from datasets import Dataset, DatasetDict

from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


def load_jsonl(filepath):
    """加载 JSONL 文件"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


class MassiveIntentClassificationCodeSwitching(AbsTaskClassification):
    """
    MassiveIntentClassification Code-Switching 变体任务
    - 完全从本地 jsonl 文件加载（text 和 label）
    - 只支持英语 (en)
    """

    fast_loading = True
    metadata = TaskMetadata(
        name="MassiveIntentClassificationCodeSwitching",
        dataset={
            "path": "mteb/amazon_massive_intent",
            "revision": "4672e20407010da34463acc759c162ca9734bca6",
        },
        description="MassiveIntentClassification Code-Switching variant. All data loaded from local file. English only.",
        reference="https://arxiv.org/abs/2204.08582",
        category="t2c",
        modalities=["text"],
        type="Classification",
        eval_splits=["validation", "test"],
        eval_langs={"en": ["eng-Latn"]},
        main_score="accuracy",
        date=("2022-01-01", "2022-04-22"),
        domains=["Spoken"],
        task_subtypes=[],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="human-translated and localized",
        bibtex_citation=r"""
@misc{fitzgerald2022massive,
  archiveprefix = {arXiv},
  author = {Jack FitzGerald and Christopher Hench and Charith Peris and Scott Mackie and Kay Rottmann and Ana Sanchez and Aaron Nash and Liam Urbach and Vishesh Kakarala and Richa Singh and Swetha Ranganath and Laurie Crist and Misha Britan and Wouter Leeuwis and Gokhan Tur and Prem Natarajan},
  eprint = {2204.08582},
  primaryclass = {cs.CL},
  title = {MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages},
  year = {2022},
}
""",
        prompt="Given a user utterance as query, find the user intents",
    )

    def __init__(
        self,
        validation_file: str = None,
        test_file: str = None,
        **kwargs
    ):
        """
        初始化任务

        Args:
            validation_file: validation split 的本地 jsonl 文件路径。如果为 None，则从环境变量 MASSIVE_INTENT_VALIDATION_FILE 获取
            test_file: test split 的本地 jsonl 文件路径。如果为 None，则从环境变量 MASSIVE_INTENT_TEST_FILE 获取
        """
        super().__init__(**kwargs)
        self.validation_file = validation_file or os.getenv("MASSIVE_INTENT_VALIDATION_FILE")
        self.test_file = test_file or os.getenv("MASSIVE_INTENT_TEST_FILE")

    def load_data(self, **kwargs):
        """
        加载数据：train 从 HuggingFace 加载，validation/test 从本地 jsonl 文件加载
        """
        if self.data_loaded:
            return

        # split 名称与文件路径的映射
        split_files = {
            "validation": self.validation_file,
            "test": self.test_file,
        }

        dataset_dict = {}

        # 从 HuggingFace 加载 train split
        print("Loading train data from HuggingFace...")
        from datasets import load_dataset
        hf_dataset = load_dataset(
            self.metadata.dataset["path"],
            "en",
            revision=self.metadata.dataset["revision"],
            trust_remote_code=True,
        )
        dataset_dict["train"] = hf_dataset["train"]
        print(f"Loaded {len(dataset_dict['train'])} samples for train split from HuggingFace")

        # 从本地文件加载 validation 和 test
        for split, filepath in split_files.items():
            if not filepath:
                print(f"Warning: No file provided for {split} split, skipping.")
                continue

            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Data file not found for {split}: {filepath}")

            print(f"Loading {split} data from: {filepath}")
            local_data = load_jsonl(filepath)

            texts = [item.get('text', '') for item in local_data]
            labels = [item.get('label', 0) for item in local_data]

            dataset_dict[split] = Dataset.from_dict({
                "text": texts,
                "label": labels,
            })
            print(f"Loaded {len(texts)} samples for {split} split")

        if not dataset_dict:
            raise ValueError(
                "No data files provided. "
                "Please pass validation_file/test_file parameters or set "
                "MASSIVE_INTENT_VALIDATION_FILE/MASSIVE_INTENT_TEST_FILE environment variables."
            )

        # 将数据集包装在 "en" key 下，以匹配 eval_langs={"en": ["eng-Latn"]} 的结构
        self.dataset = {"en": DatasetDict(dataset_dict)}
        self.data_loaded = True
