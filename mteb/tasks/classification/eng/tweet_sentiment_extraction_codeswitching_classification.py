from __future__ import annotations

import json
import os
from datasets import Dataset, DatasetDict, load_dataset

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


class TweetSentimentExtractionClassificationCodeSwitching(AbsTaskClassification):
    """
    TweetSentimentExtractionClassification Code-Switching 变体任务
    - train: 从 HuggingFace 加载
    - test: 从本地 jsonl 文件加载（code-switching text）

    本地数据格式要求：
    {"text": "...", "label": "positive/negative/neutral", ...}
    """

    metadata = TaskMetadata(
        name="TweetSentimentExtractionClassificationCodeSwitching",
        description="TweetSentimentExtractionClassification Code-Switching variant. Test data loaded from local file.",
        reference="https://www.kaggle.com/competitions/tweet-sentiment-extraction/overview",
        dataset={
            "path": "mteb/tweet_sentiment_extraction",
            "revision": "d604517c81ca91fe16a244d1248fc021f9ecee7a",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=(
            "2020-01-01",
            "2020-12-31",
        ),
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{tweet-sentiment-extraction,
  author = {Maggie, Phil Culliton, Wei Chen},
  publisher = {Kaggle},
  title = {Tweet Sentiment Extraction},
  url = {https://kaggle.com/competitions/tweet-sentiment-extraction},
  year = {2020},
}
""",
        prompt="Classify the sentiment of a given tweet as either positive, negative, or neutral",
    )

    samples_per_label = 32

    def __init__(self, test_file: str = None, **kwargs):
        """
        初始化任务

        Args:
            test_file: test split 的本地 jsonl 文件路径。如果为 None，则从环境变量 TWEET_SENTIMENT_TEST_FILE 获取
        """
        super().__init__(**kwargs)
        self.test_file = test_file or os.getenv("TWEET_SENTIMENT_TEST_FILE")

    def load_data(self, **kwargs):
        """
        加载数据：train 从 HuggingFace 加载，test 从本地 jsonl 文件加载
        """
        if self.data_loaded:
            return

        # ========== 1. 验证 test 文件路径 ==========
        if not self.test_file:
            raise ValueError(
                "Test file path not provided. "
                "Please pass test_file parameter or set TWEET_SENTIMENT_TEST_FILE environment variable."
            )

        if not os.path.exists(self.test_file):
            raise FileNotFoundError(f"Test file not found: {self.test_file}")

        # ========== 2. 从 HuggingFace 加载 train split ==========
        print("Loading train data from HuggingFace...")
        hf_dataset = load_dataset(
            self.metadata.dataset["path"],
            revision=self.metadata.dataset["revision"],
            trust_remote_code=True,
        )
        train_data = hf_dataset["train"]
        print(f"Loaded {len(train_data)} samples for train split from HuggingFace")

        # ========== 3. 从本地加载 test split ==========
        print(f"Loading test data from: {self.test_file}")
        local_data = load_jsonl(self.test_file)

        texts = []
        labels = []
        for idx, item in enumerate(local_data):
            try:
                texts.append(item['text'])
                labels.append(item['label'])
            except KeyError as e:
                raise KeyError(f"Missing key {e} in item {idx}: {item}")

        test_data = Dataset.from_dict({
            "text": texts,
            "label": labels,
        })
        print(f"Loaded {len(texts)} samples for test split")

        # ========== 4. 构建数据集 ==========
        self.dataset = DatasetDict({
            "train": train_data,
            "test": test_data,
        })

        self.data_loaded = True