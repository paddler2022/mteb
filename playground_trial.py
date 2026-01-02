import mteb
import os
import json
from mteb import MTEB
from mteb.types import PromptType
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper
from mteb.models.instruct_wrapper import InstructSentenceTransformerModel
from sentence_transformers import SentenceTransformer
from mteb.tasks.retrieval.eng import TRECCOVIDCodeSwitching, ArguAna, ArguAnaCodeSwitching, ClimateFEVERHardNegativesV2, ClimateFEVERHardNegativesV2CodeSwitching, TRECCOVID, Touche2020v3Retrieval, Touche2020v3RetrievalCodeSwitching
from mteb.tasks.instruction_reranking.eng import Core17InstructionRetrievalCodeSwitching, News21InstructionRetrievalCodeSwitching, Robust04InstructionRetrievalCodeSwitching, Core17InstructionRetrieval, News21InstructionRetrieval, Robust04InstructionRetrieval
from mteb.tasks.classification.eng import TweetSentimentExtractionClassificationV2, TweetSentimentExtractionClassificationCodeSwitching
from mteb.tasks.reranking.eng import AskUbuntuDupQuestions, AskUbuntuDupQuestionsCodeSwitching
from mteb.tasks.sts.eng import STSBenchmarkSTS, STSBenchmarkCodeSwitching
from mteb.tasks.pair_classification.eng import TwitterSemEval2015PC, TwitterSemEval2015CodeSwitching
from mteb.tasks.clustering.eng import ArXivHierarchicalClusteringP2P, ArXivHierarchicalClusteringP2PCodeSwitching
from mteb.tasks.retrieval.code import HumanEvalRetrievalCodeSwitching, HumanEvalRetrieval
import torch
import argparse
from datetime import datetime


# ========== Prompt 配置 ==========

# E5-large-v2 模型的 prompt (来自 e5_models.py)
# 注意: e5-large-v2 是基础模型，只使用 query:/passage: 前缀
# 如果使用 e5-instruct 模型，需要用 E5_INSTRUCT_PROMPTS
E5_PROMPTS = {
    PromptType.query.value: "query: ",
    PromptType.document.value: "passage: ",
}

# Qwen3 模型的任务特定 instruction
QWEN3_PROMPTS = {
    # Classification tasks
    "AmazonCounterfactualClassification": "Classify a given Amazon customer review text as either counterfactual or not-counterfactual",
    "AmazonCounterfactualClassificationCodeSwitching": "Classify a given Amazon customer review text as either counterfactual or not-counterfactual",
    "AmazonPolarityClassification": "Classify Amazon reviews into positive or negative sentiment",
    "AmazonReviewsClassification": "Classify the given Amazon review into its appropriate rating category",
    "Banking77Classification": "Given a online banking query, find the corresponding intents",
    "EmotionClassification": "Classify the emotion expressed in the given Twitter message into one of the six emotions: anger, fear, joy, love, sadness, and surprise",
    "ImdbClassification": "Classify the sentiment expressed in the given movie review text from the IMDB dataset",
    "MassiveIntentClassification": "Given a user utterance as query, find the user intents",
    "MassiveIntentClassificationCodeSwitching": "Given a user utterance as query, find the user intents",
    "MassiveScenarioClassification": "Given a user utterance as query, find the user scenarios",
    "MTOPDomainClassification": "Classify the intent domain of the given utterance in task-oriented conversation",
    "MTOPIntentClassification": "Classify the intent of the given utterance in task-oriented conversation",
    "ToxicConversationsClassification": "Classify the given comments as either toxic or not toxic",
    "TweetSentimentExtractionClassification": "Classify the sentiment of a given tweet as either positive, negative, or neutral",
    "TweetSentimentExtractionClassification.v2": "Classify the sentiment of a given tweet as either positive, negative, or neutral",
    "TweetSentimentExtractionClassification.v2CodeSwitching": "Classify the sentiment of a given tweet as either positive, negative, or neutral",
    "TweetSentimentExtractionClassificationCodeSwitching": "Classify the sentiment of a given tweet as either positive, negative, or neutral",
    "TNews": "Classify the fine-grained category of the given news title",
    "IFlyTek": "Given an App description text, find the appropriate fine-grained category",
    "MultilingualSentiment": "Classify sentiment of the customer review into positive, neutral, or negative",
    "JDReview": "Classify the customer review for iPhone on e-commerce platform into positive or negative",
    "OnlineShopping": "Classify the customer review for online shopping into positive or negative",
    "Waimai": "Classify the customer review from a food takeaway platform into positive or negative",
    # Clustering tasks
    "ArxivClusteringP2P": "Identify the main and secondary category of Arxiv papers based on the titles and abstracts",
    "ArxivClusteringS2S": "Identify the main and secondary category of Arxiv papers based on the titles",
    "ArXivHierarchicalClusteringP2P": "Identify the main and secondary category of Arxiv papers based on the titles and abstracts",
    "ArXivHierarchicalClusteringP2PCodeSwitching": "Identify the main and secondary category of Arxiv papers based on the titles and abstracts",
    "BiorxivClusteringP2P": "Identify the main category of Biorxiv papers based on the titles and abstracts",
    "BiorxivClusteringS2S": "Identify the main category of Biorxiv papers based on the titles",
    "MedrxivClusteringP2P": "Identify the main category of Medrxiv papers based on the titles and abstracts",
    "MedrxivClusteringS2S": "Identify the main category of Medrxiv papers based on the titles",
    "RedditClustering": "Identify the topic or theme of Reddit posts based on the titles",
    "RedditClusteringP2P": "Identify the topic or theme of Reddit posts based on the titles and posts",
    "StackExchangeClustering": "Identify the topic or theme of StackExchange posts based on the titles",
    "StackExchangeClusteringP2P": "Identify the topic or theme of StackExchange posts based on the given paragraphs",
    "TwentyNewsgroupsClustering": "Identify the topic or theme of the given news articles",
    "CLSClusteringS2S": "Identify the main category of scholar papers based on the titles",
    "CLSClusteringP2P": "Identify the main category of scholar papers based on the titles and abstracts",
    "ThuNewsClusteringS2S": "Identify the topic or theme of the given news articles based on the titles",
    "ThuNewsClusteringP2P": "Identify the topic or theme of the given news articles based on the titles and contents",
    # Reranking tasks
    "AskUbuntuDupQuestions": "Retrieve duplicate questions from AskUbuntu forum",
    "AskUbuntuDupQuestionsCodeSwitching": "Retrieve duplicate questions from AskUbuntu forum",
    "MindSmallReranking": "Retrieve relevant news articles based on user browsing history",
    "SciDocsRR": "Given a title of a scientific paper, retrieve the titles of other relevant papers",
    "StackOverflowDupQuestions": "Retrieve duplicate questions from StackOverflow forum",
    "SprintDuplicateQuestions": "Retrieve duplicate questions from Sprint forum",
    "TwitterSemEval2015": "Retrieve tweets that are semantically similar to the given tweet",
    "TwitterURLCorpus": "Retrieve tweets that are semantically similar to the given tweet",
    "T2Reranking": "Given a Chinese search query, retrieve web passages that answer the question",
    "MmarcoReranking": "Given a Chinese search query, retrieve web passages that answer the question",
    "CMedQAv1": "Given a Chinese community medical question, retrieve replies that best answer the question",
    "CMedQAv2": "Given a Chinese community medical question, retrieve replies that best answer the question",
    "Ocnli": "Retrieve semantically similar text.",
    "Cmnli": "Retrieve semantically similar text.",
    # Retrieval tasks
    "ArguAna": "Given a claim, find documents that refute the claim",
    "ArguAnaCodeSwitching": "Given a claim, find documents that refute the claim",
    "ClimateFEVER": "Given a claim about climate change, retrieve documents that support or refute the claim",
    "ClimateFEVERHardNegatives": "Given a claim about climate change, retrieve documents that support or refute the claim",
    "ClimateFEVERHardNegativesCodeSwitching": "Given a claim about climate change, retrieve documents that support or refute the claim",
    "ClimateFEVERHardNegatives.v2": "Given a claim about climate change, retrieve documents that support or refute the claim",
    "ClimateFEVERHardNegatives.v2CodeSwitching": "Given a claim about climate change, retrieve documents that support or refute the claim",
    "DBPedia": "Given a query, retrieve relevant entity descriptions from DBPedia",
    "FEVER": "Given a claim, retrieve documents that support or refute the claim",
    "FEVERHardNegatives": "Given a claim, retrieve documents that support or refute the claim",
    "FiQA2018": "Given a financial question, retrieve user replies that best answer the question",
    "FiQA2018CodeSwitching": "Given a financial question, retrieve user replies that best answer the question",
    "HotpotQA": "Given a multi-hop question, retrieve documents that can help answer the question",
    "HotpotQAHardNegatives": "Given a multi-hop question, retrieve documents that can help answer the question",
    "HotpotQAHardNegativesCodeSwitching": "Given a multi-hop question, retrieve documents that can help answer the question",
    "MSMARCO": "Given a web search query, retrieve relevant passages that answer the query",
    "NFCorpus": "Given a question, retrieve relevant documents that best answer the question",
    "NQ": "Given a question, retrieve Wikipedia passages that answer the question",
    "QuoraRetrieval": "Given a question, retrieve questions that are semantically equivalent to the given question",
    "SCIDOCS": "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper",
    "SCIDOCSCodeSwitching": "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper",
    "SciFact": "Given a scientific claim, retrieve documents that support or refute the claim",
    "Touche2020": "Given a question, retrieve detailed and persuasive arguments that answer the question",
    "Touche2020Retrieval.v3": "Given a question, retrieve detailed and persuasive arguments that answer the question",
    "Touche2020Retrieval.v3CodeSwitching": "Given a question, retrieve detailed and persuasive arguments that answer the question",
    "TRECCOVID": "Given a query on COVID-19, retrieve documents that answer the query",
    "TRECCOVIDCodeSwitching": "Given a query on COVID-19, retrieve documents that answer the query",
    "T2Retrieval": "Given a Chinese search query, retrieve web passages that answer the question",
    "MMarcoRetrieval": "Given a web search query, retrieve relevant passages that answer the query",
    "DuRetrieval": "Given a Chinese search query, retrieve web passages that answer the question",
    "CovidRetrieval": "Given a question on COVID-19, retrieve news articles that answer the question",
    "CmedqaRetrieval": "Given a Chinese community medical question, retrieve replies that best answer the question",
    "EcomRetrieval": "Given a user query from an e-commerce website, retrieve description sentences of relevant products",
    "MedicalRetrieval": "Given a medical question, retrieve user replies that best answer the question",
    "VideoRetrieval": "Given a video search query, retrieve the titles of relevant videos",
    "HumanEvalRetrieval": "Given a question about code problem, retrieval code that can solve user's problem",
    "HumanEvalRetrievalCodeSwitching": "Given a question about code problem, retrieval code that can solve user's problem",
    # Instruction Retrieval tasks (FollowIR)
    "Core17InstructionRetrieval": "Retrieval the relevant passage for the given query",
    "News21InstructionRetrieval": "Retrieval the relevant passage for the given query",
    "Robust04InstructionRetrieval": "Retrieval the relevant passage for the given query",
    "Core17InstructionRetrievalCodeSwitching": "Retrieval the relevant passage for the given query",
    "News21InstructionRetrievalCodeSwitching": "Retrieval the relevant passage for the given query",
    "Robust04InstructionRetrievalCodeSwitching": "Retrieval the relevant passage for the given query",
    # STS tasks
    "STSBenchmarkMultilingualSTS": "Retrieve semantically similar text",
    "SICKFr": "Retrieve semantically similar text",
    "SummEvalFr": "Given a news summary, retrieve other semantically similar summaries",
    "STS12": "Retrieve semantically similar text",
    "STS13": "Retrieve semantically similar text",
    "STS14": "Retrieve semantically similar text",
    "STS15": "Retrieve semantically similar text",
    "STS16": "Retrieve semantically similar text",
    "STS17": "Retrieve semantically similar text",
    "STS22": "Retrieve semantically similar text",
    "STS22CodeSwitching": "Retrieve semantically similar text",
    "STSBenchmark": "Retrieve semantically similar text",
    "STSBenchmarkCodeSwitching": "Retrieve semantically similar text",
    "BIOSSES": "Retrieve semantically similar text",
    "SICK-R": "Retrieve semantically similar text",
    "SickrCodeSwitchingSTS": "Retrieve semantically similar text",
    "SummEval": "Retrieve semantically similar text",
    # Pair Classification tasks
    "PawsX": "Retrieve semantically similar text",
    "XNLI": "Retrieve semantically similar text",
    "TwitterSemEval2015PC": "Retrieve tweets that are semantically similar to the given tweet",
    "TwitterSemEval2015CodeSwitching": "Retrieve tweets that are semantically similar to the given tweet",
    # CQADupstack tasks
    "CQADupstackRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question",
    "CQADupstackGamingRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question",
    "CQADupstackGamingRetrievalCodeSwitching": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question",
    "CQADupstackUnixRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question",
    # Code retrieval tasks
    "AppsRetrieval": "Given a question about code problem, retrieval code that can solve user's problem",
    "COIRCodeSearchNetRetrieval": "Given a code snippet, retrieve the comment corresponding to that code.",
    "CodeEditSearchRetrieval": "Given a piece of code, retrieval code that in the ",
    "CodeFeedbackMT": "Given a question about coding, retrieval code or passage that can solve user's question",
    "CodeFeedbackST": "Given a question about coding, retrieval code or passage that can solve user's question",
    "CodeSearchNetCCRetrieval": "Given a code comment, retrieve the code snippet corresponding to that comment.",
    "CodeSearchNetRetrieval": "Given a code snippet, retrieve the comment corresponding to that code.",
    "CodeTransOceanContest": "Given a piece for code, retrieval semantically similar code",
    "CodeTransOceanDL": "Given a piece for code, retrieval semantically similar code",
    "CosQA": "Given a question about coding, retrieval code or passage that can solve user's question",
    "StackOverflowQA": "Given a question about coding, retrieval code or passage that can solve user's question",
    "SyntheticText2SQL": "Given a user's question, retrieve SQL queries that are appropriate responses to the question",
    # Bitext Mining tasks
    "BibleNLPBitextMining": "Retrieve parallel sentences",
    "BUCC.v2": "Retrieve parallel sentences",
    "DiaBlaBitextMining": "Retrieve parallel sentences",
    "FloresBitextMining": "Retrieve parallel sentences",
    "Tatoeba": "Retrieve parallel sentences",
    # Other retrieval tasks
    "HagridRetrieval": "Retrieval the relevant passage for the given query",
    "LegalBenchCorporateLobbying": "Retrieval the relevant passage for the given query",
    "LEMBPasskeyRetrieval": "Retrieval the relevant passage for the given query",
    "BelebeleRetrieval": "Retrieval the relevant passage for the given query",
    "MLQARetrieval": "Retrieval the relevant passage for the given query",
    "StatcanDialogueDatasetRetrieval": "Retrieval the relevant passage for the given query",
    "WikipediaRetrievalMultilingual": "Retrieval the relevant passage for the given query",
    "WebLINXCandidatesReranking": "Retrieval the relevant passage for the given query",
    "WikipediaRerankingMultilingual": "Retrieval the relevant passage for the given query",
    "MIRACLRetrievalHardNegatives": "Retrieval relevant passage for the given query",
}


def qwen3_instruction_template(instruction: str, prompt_type: PromptType | None = None) -> str:
    """Qwen3 Embedding 模型的 instruction 模板 (参考 qwen3_models.py)"""
    if not instruction or prompt_type == PromptType.document:
        return ""
    if isinstance(instruction, dict):
        if prompt_type is None:
            instruction = list(instruction.values())[0]
        else:
            instruction = instruction[prompt_type]
    return f"Instruct: {instruction}\nQuery:"


# ========== 模型加载函数 ==========


def get_model_kwargs(model_path):
    """根据环境返回合适的 model_kwargs"""
    kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "cuda"
    }
    if "qwen" in model_path.lower() or "llama" in model_path.lower() or "SFR-Embedding-2_R" in model_path:
        kwargs["attn_implementation"] = "flash_attention_2"
        print("[INFO] 使用 Flash Attention 2")
    else:
        print("[INFO] 使用默认 Attention 实现")
    return kwargs


def load_model(model_path):
    """加载 HuggingFace 模型（使用 MTEB 内置配置）"""
    if "SFR-Embedding-2_R" in model_path:
        model = mteb.get_model(
            "Salesforce/SFR-Embedding-2_R",
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16
        )

    elif "llama" in model_path.lower():
        model = mteb.get_model(model_path)
    else:

        model_kwargs = get_model_kwargs(model_path)
        model = mteb.get_model(
            model_path,
            model_kwargs=model_kwargs,
            trust_remote_code=True,
        )
    return model


def load_model_ST(model_path):
    """
    加载本地模型并支持 prompt
    自动检测并使用 Flash Attention 2（如果可用）
    """
    model_path_lower = model_path.lower()
    model_kwargs = get_model_kwargs(model_path)

    # E5 系列模型
    if "e5" in model_path_lower:
        print(f"[INFO] 检测到 E5 模型，使用 query:/passage: prompt")
        model = SentenceTransformerEncoderWrapper(
            model=model_path,
            revision=None,
            model_prompts=E5_PROMPTS,
            trust_remote_code=True
        )
        return model
    # Qwen3 Embedding 系列模型
    elif "qwen" in model_path_lower and "embedding" in model_path_lower:
        print(f"[INFO] 检测到 Qwen Embedding 模型，使用 Instruct 模板")
        model = InstructSentenceTransformerModel(
            model_name=model_path,
            revision=None,
            instruction_template=qwen3_instruction_template,
            apply_instruction_to_passages=False,
            prompts_dict=QWEN3_PROMPTS,
            model_kwargs=model_kwargs,
            trust_remote_code=True,
        )
        # Qwen3 在 flash_attention_2 模式下需要 left padding
        if hasattr(model, 'model') and hasattr(model.model, 'tokenizer'):
            model.model.tokenizer.padding_side = 'left'
        return model
    # MiniLM 系列模型 - 使用与 mteb.get_model() 一致的默认配置
    elif "minilm" in model_path_lower:
        print(f"[INFO] 检测到 MiniLM 模型，使用默认配置（与 mteb.get_model 一致）")
        model = SentenceTransformer(
            model_path,
            device="cuda"
        )
        return model
    # 其他模型
    else:
        print(f"[INFO] 未检测到特殊 prompt 需求，使用默认配置")
        model = SentenceTransformer(
            model_path,
            model_kwargs=model_kwargs,
            device="cuda"
        )
        model.tokenizer.padding_side = 'left'
        return model


# ========== 任务加载函数 ==========

def load_IR_code_switching_task():
    task_core17_cs = Core17InstructionRetrievalCodeSwitching(
        query_file="./Code_Switching_dataset/IR/core17/jhu-clsp_core17-instructions-mteb_required_format_queries_gpt-5-mini.jsonl",
        instruction_file="./Code_Switching_dataset/IR/core17/jhu-clsp_core17-instructions-mteb_required_format_instructions_gpt-5-mini.jsonl"
    )
    task_news21_cs = News21InstructionRetrievalCodeSwitching(
        query_file="./Code_Switching_dataset/IR/news21/jhu-clsp_news21-instructions-mteb_required_format_queries_gpt-5-mini.jsonl",
        instruction_file="./Code_Switching_dataset/IR/news21/jhu-clsp_news21-instructions-mteb_required_format_instructions_gpt-5-mini.jsonl"
    )
    task_robust04_cs = Robust04InstructionRetrievalCodeSwitching(
        query_file="./Code_Switching_dataset/IR/robust04/jhu-clsp_robust04-instructions-mteb_required_format_queries_gpt-5-mini.jsonl",
        instruction_file="./Code_Switching_dataset/IR/robust04/jhu-clsp_robust04-instructions-mteb_required_format_instructions_gpt-5-mini.jsonl"
    )
    IR_tasks_cs = [task_core17_cs, task_news21_cs, task_robust04_cs]
    return IR_tasks_cs


def load_IR_original_task():
    task_core17 = Core17InstructionRetrieval()
    task_news21 = News21InstructionRetrieval()
    task_robust04 = Robust04InstructionRetrieval()
    IR_task_orginal = [task_core17, task_news21, task_robust04]
    return IR_task_orginal


def load_Retrieval_And_IR_fixed_task_Chinese():
    task_HumanEval_ch = HumanEvalRetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLChinese_fixed/Chinese_embedding-benchmark_HumanEval_queries_queries_gpt-5-mini.jsonl")
    task_core17_ch = Core17InstructionRetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/IR_Chinese_fixed/core17/fixed_core17_queries_cn.jsonl",
        instruction_file="./CodeSwitching_Dataset_fixed/IR_Chinese_fixed/core17/fixed_core17_instructions_cn.jsonl"
    )
    task_news21_ch = News21InstructionRetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/IR_Chinese_fixed/news21/fixed_news21_queries_cn.jsonl",
        instruction_file="./CodeSwitching_Dataset_fixed/IR_Chinese_fixed/news21/fixed_news21_instructions_cn.jsonl"
    )
    task_robust04_ch = Robust04InstructionRetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/IR_Chinese_fixed/robust04/fixed_robust04_queries_cn.jsonl",
        instruction_file="./CodeSwitching_Dataset_fixed/IR_Chinese_fixed/robust04/fixed_robust04_instructions_cn.jsonl"
    )
    tasks = [task_HumanEval_ch, task_core17_ch, task_news21_ch, task_robust04_ch]
    return tasks


def load_Retrieval_Chinese():
    task_touchev3_ch = Touche2020v3RetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLChinese_fixed/mteb_webis-touche2020-v3_queries_chinese.jsonl")
    task_trec_covid_ch = TRECCOVIDCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLChinese_fixed/mteb_trec-covid_queries_queries_gpt-5-mini.jsonl")
    tasks = [task_trec_covid_ch, task_touchev3_ch]
    return tasks


def load_Retrieval_And_IR_fixed_task_Japanese():
    task_HumanEval_jp = HumanEvalRetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLJapanese_fixed/Japanese_embedding-benchmark_HumanEval_queries_queries_gpt-5-mini.jsonl")
    task_core17_jp = Core17InstructionRetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/IR_Japanese_fixed/core17/core17_queries_jp.jsonl",
        instruction_file="./CodeSwitching_Dataset_fixed/IR_Japanese_fixed/core17/core17_instructions_jp.jsonl"
    )
    task_news21_jp = News21InstructionRetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/IR_Japanese_fixed/news21/news21_queries_jp.jsonl",
        instruction_file="./CodeSwitching_Dataset_fixed/IR_Japanese_fixed/news21/news21_instructions_jp.jsonl"
    )
    task_robust04_jp = Robust04InstructionRetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/IR_Japanese_fixed/robust04/robust04_queries_jp.jsonl",
        instruction_file="./CodeSwitching_Dataset_fixed/IR_Japanese_fixed/robust04/robust04_instructions_jp.jsonl"
    )
    tasks = [task_HumanEval_jp, task_core17_jp, task_news21_jp, task_robust04_jp]
    return tasks


def load_Retrieval_Japanese():
    task_trec_covid_jp = TRECCOVIDCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLJapanese_fixed/Japanese_mteb_trec-covid_queries_queries_gpt-5-mini.jsonl")
    task_touchev3_jp = Touche2020v3RetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLJapanese_fixed/mteb_webis-touche2020-v3_queries_japanese.jsonl")
    tasks = [task_trec_covid_jp, task_touchev3_jp]
    return tasks


def load_Touche2020RetrievalV3_ch():
    touchev3_ch = Touche2020v3RetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLChinese_fixed/mteb_webis-touche2020-v3_queries_chinese.jsonl")
    # touchev3 = Touche2020v3Retrieval()
    return [touchev3_ch]


def load_Touche2020RetrievalV3_jp():
    touchev3_jp = Touche2020v3RetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLJapanese_fixed/mteb_webis-touche2020-v3_queries_japanese.jsonl")
    return [touchev3_jp]

def load_touche2020RetrievalV3():
    task_touche2020 = Touche2020v3Retrieval()
    return [task_touche2020]

def load_original_tasks():
    task_core17 = Core17InstructionRetrieval()
    task_news21 = News21InstructionRetrieval()
    task_robust04 = Robust04InstructionRetrieval()
    task_touche2020 = Touche2020v3Retrieval()
    task_humaneval = HumanEvalRetrieval()
    task_treccovid = TRECCOVID()
    tasks = [task_humaneval, task_core17, task_news21, task_robust04,]
    return tasks


def load_og_Retrieval_tasks():
    task_touche2020 = Touche2020v3Retrieval()
    task_treccovid = TRECCOVID()
    return [task_treccovid, task_touche2020]

def load_TRECCOVID_tasks():
    task_treccovid = TRECCOVID()
    return [task_treccovid]

def load_TRECCOVID_Chinese_tasks():
    task_treccovid_ch = TRECCOVIDCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLChinese_fixed/mteb_trec-covid_queries_queries_gpt-5-mini.jsonl")
    return [task_treccovid_ch]

def load_TRECCOVID_Japanese_tasks():
    task_treccovid_jp = TRECCOVIDCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLJapanese_fixed/Japanese_mteb_trec-covid_queries_queries_gpt-5-mini.jsonl")
    return [task_treccovid_jp]


# ========== 新增 Code-Switching 任务加载函数 ==========

# ----- Classification -----
def load_TweetSentiment_Chinese():
    task = TweetSentimentExtractionClassificationCodeSwitching(
        test_file="./CodeSwitching_Dataset_fixed/Classification_Chinese_fixed/Chinese_mteb_tweet_sentiment_extraction_None_test_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


def load_TweetSentiment_Japanese():
    task = TweetSentimentExtractionClassificationCodeSwitching(
        test_file="./CodeSwitching_Dataset_fixed/Classification_Japanese_fixed/Japanese_mteb_tweet_sentiment_extraction_None_test_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


# ----- Clustering -----
def load_ArxivClustering_Chinese():
    task = ArXivHierarchicalClusteringP2PCodeSwitching(
        test_file="./CodeSwitching_Dataset_fixed/Clustering_Chinese_fixed/Chinese_mteb_arxiv-clustering-p2p_None_test_sampled_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


def load_ArxivClustering_Japanese():
    task = ArXivHierarchicalClusteringP2PCodeSwitching(
        test_file="./CodeSwitching_Dataset_fixed/Clustering_Japanese_fixed/Japanese_mteb_arxiv-clustering-p2p_None_test_sampled_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


# ----- PairClassification -----
def load_TwitterSemEval_Chinese():
    task = TwitterSemEval2015CodeSwitching(
        test_file="./CodeSwitching_Dataset_fixed/PairClassification_Chinese_fixed/Chinese_mteb_twittersemeval2015-pairclassification_None_test_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


def load_TwitterSemEval_Japanese():
    task = TwitterSemEval2015CodeSwitching(
        test_file="./CodeSwitching_Dataset_fixed/PairClassification_Japanese_fixed/Japanese_mteb_twittersemeval2015-pairclassification_None_test_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


# ----- STS -----
def load_STSBenchmark_Chinese():
    task = STSBenchmarkCodeSwitching(
        data_file="./CodeSwitching_Dataset_fixed/STS_Chinese_fixed/Chinese_mteb_stsbenchmark-sts_None_test_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


def load_STSBenchmark_Japanese():
    task = STSBenchmarkCodeSwitching(
        data_file="./CodeSwitching_Dataset_fixed/STS_Japanese_fixed/Japanese_mteb_stsbenchmark-sts_None_test_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


# ----- Reranking -----
def load_AskUbuntu_Chinese():
    task = AskUbuntuDupQuestionsCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/Reranking_Chinese_fixed/Chinese_mteb_AskUbuntuDupQuestions_queries_test_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


def load_AskUbuntu_Japanese():
    task = AskUbuntuDupQuestionsCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/Reranking_Japanese_fixed/Japanese_mteb_AskUbuntuDupQuestions_queries_test_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


# ----- Retrieval -----
def load_ArguAna_Chinese():
    task = ArguAnaCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLChinese_fixed/Chinese_mteb_arguana_queries_queries_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


def load_ArguAna_Japanese():
    task = ArguAnaCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLJapanese_fixed/Japanese_mteb_arguana_queries_queries_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


def load_ClimateFEVER_Chinese():
    task = ClimateFEVERHardNegativesV2CodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLChinese_fixed/Chinese_mteb_ClimateFEVER_test_top_250_only_w_correct-v2_queries_test_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


def load_ClimateFEVER_Japanese():
    task = ClimateFEVERHardNegativesV2CodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLJapanese_fixed/Japanese_mteb_ClimateFEVER_test_top_250_only_w_correct-v2_queries_test_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


# ========== French Code-Switching 任务加载函数 ==========

# ----- Classification -----
def load_TweetSentiment_French():
    task = TweetSentimentExtractionClassificationCodeSwitching(
        test_file="./CodeSwitching_Dataset_fixed/Classification_French_fixed/French_mteb_tweet_sentiment_extraction_None_test_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


# ----- Clustering -----
def load_ArxivClustering_French():
    task = ArXivHierarchicalClusteringP2PCodeSwitching(
        test_file="./CodeSwitching_Dataset_fixed/Clustering_French_fixed/French_mteb_arxiv-clustering-p2p_None_test_sampled_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


# ----- PairClassification -----
def load_TwitterSemEval_French():
    task = TwitterSemEval2015CodeSwitching(
        test_file="./CodeSwitching_Dataset_fixed/PairClassification_French_fixed/French_mteb_twittersemeval2015-pairclassification_None_test_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


# ----- STS -----
def load_STSBenchmark_French():
    task = STSBenchmarkCodeSwitching(
        data_file="./CodeSwitching_Dataset_fixed/STS_French_fixed/French_mteb_stsbenchmark-sts_None_test_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


# ----- Reranking -----
def load_AskUbuntu_French():
    task = AskUbuntuDupQuestionsCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/Reranking_French_fixed/French_mteb_AskUbuntuDupQuestions_queries_test_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


# ----- Retrieval -----
def load_ArguAna_French():
    task = ArguAnaCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLFrench_fixed/French_mteb_arguana_queries_queries_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


def load_ClimateFEVER_French():
    task = ClimateFEVERHardNegativesV2CodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLFrench_fixed/French_mteb_ClimateFEVER_test_top_250_only_w_correct-v2_queries_test_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


def load_Touche2020RetrievalV3_French():
    task = Touche2020v3RetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLFrench_fixed/French_mteb_webis-touche2020-v3_queries_train_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


def load_HumanEval_French():
    task = HumanEvalRetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLFrench_fixed/French_embedding-benchmark_HumanEval_queries_queries_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


def load_TRECCOVID_French():
    task = TRECCOVIDCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLFrench_fixed/French_mteb_trec-covid_queries_queries_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


def load_Retrieval_French():
    """French Retrieval tasks: TRECCOVID + Touche2020v3"""
    task_treccovid_fr = TRECCOVIDCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLFrench_fixed/French_mteb_trec-covid_queries_queries_xiaomi--mimo-v2-flash_free.jsonl"
    )
    task_touchev3_fr = Touche2020v3RetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLFrench_fixed/French_mteb_webis-touche2020-v3_queries_train_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task_treccovid_fr, task_touchev3_fr]


# ----- IR -----
def load_Retrieval_And_IR_fixed_task_French():
    task_HumanEval_fr = HumanEvalRetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLFrench_fixed/French_embedding-benchmark_HumanEval_queries_queries_xiaomi--mimo-v2-flash_free.jsonl"
    )
    task_core17_fr = Core17InstructionRetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/IR_French_fixed/core17/French_jhu-clsp_core17-instructions-mteb_required_format_queries_xiaomi--mimo-v2-flash_free.jsonl",
        instruction_file="./CodeSwitching_Dataset_fixed/IR_French_fixed/core17/French_jhu-clsp_core17-instructions-mteb_required_format_instructions_xiaomi--mimo-v2-flash_free.jsonl"
    )
    task_news21_fr = News21InstructionRetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/IR_French_fixed/news21/French_jhu-clsp_news21-instructions-mteb_required_format_queries_xiaomi--mimo-v2-flash_free.jsonl",
        instruction_file="./CodeSwitching_Dataset_fixed/IR_French_fixed/news21/French_jhu-clsp_news21-instructions-mteb_required_format_instructions_xiaomi--mimo-v2-flash_free.jsonl"
    )
    task_robust04_fr = Robust04InstructionRetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/IR_French_fixed/robust04/French_jhu-clsp_robust04-instructions-mteb_required_format_queries_xiaomi--mimo-v2-flash_free.jsonl",
        instruction_file="./CodeSwitching_Dataset_fixed/IR_French_fixed/robust04/French_jhu-clsp_robust04-instructions-mteb_required_format_instructions_xiaomi--mimo-v2-flash_free.jsonl"
    )
    tasks = [task_HumanEval_fr, task_core17_fr, task_news21_fr, task_robust04_fr]
    return tasks


# ========== German Code-Switching 任务加载函数 ==========

# ----- Classification -----
def load_TweetSentiment_German():
    task = TweetSentimentExtractionClassificationCodeSwitching(
        test_file="./CodeSwitching_Dataset_fixed/Classification_German_fixed/German_mteb_tweet_sentiment_extraction_None_test_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


# ----- Clustering -----
def load_ArxivClustering_German():
    task = ArXivHierarchicalClusteringP2PCodeSwitching(
        test_file="./CodeSwitching_Dataset_fixed/Clustering_German_fixed/German_mteb_arxiv-clustering-p2p_None_test_sampled_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


# ----- PairClassification -----
def load_TwitterSemEval_German():
    task = TwitterSemEval2015CodeSwitching(
        test_file="./CodeSwitching_Dataset_fixed/PairClassification_German_fixed/German_mteb_twittersemeval2015-pairclassification_None_test_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


# ----- STS -----
def load_STSBenchmark_German():
    task = STSBenchmarkCodeSwitching(
        data_file="./CodeSwitching_Dataset_fixed/STS_German_fixed/German_mteb_stsbenchmark-sts_None_test_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


# ----- Reranking -----
def load_AskUbuntu_German():
    task = AskUbuntuDupQuestionsCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/Reranking_German_fixed/German_mteb_AskUbuntuDupQuestions_queries_test_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


# ----- Retrieval -----
def load_ArguAna_German():
    task = ArguAnaCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLGerman_fixed/German_mteb_arguana_queries_queries_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


def load_ClimateFEVER_German():
    task = ClimateFEVERHardNegativesV2CodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLGerman_fixed/German_mteb_ClimateFEVER_test_top_250_only_w_correct-v2_queries_test_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


def load_Touche2020RetrievalV3_German():
    task = Touche2020v3RetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLGerman_fixed/German_mteb_webis-touche2020-v3_queries_train_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


def load_HumanEval_German():
    task = HumanEvalRetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLGerman_fixed/German_embedding-benchmark_HumanEval_queries_queries_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


def load_TRECCOVID_German():
    task = TRECCOVIDCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLGerman_fixed/German_mteb_trec-covid_queries_queries_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


def load_Retrieval_German():
    """German Retrieval tasks: TRECCOVID + Touche2020v3"""
    task_treccovid_de = TRECCOVIDCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLGerman_fixed/German_mteb_trec-covid_queries_queries_xiaomi--mimo-v2-flash_free.jsonl"
    )
    task_touchev3_de = Touche2020v3RetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLGerman_fixed/German_mteb_webis-touche2020-v3_queries_train_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task_treccovid_de, task_touchev3_de]


# ----- IR -----
def load_Retrieval_And_IR_fixed_task_German():
    task_HumanEval_de = HumanEvalRetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLGerman_fixed/German_embedding-benchmark_HumanEval_queries_queries_xiaomi--mimo-v2-flash_free.jsonl"
    )
    task_core17_de = Core17InstructionRetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/IR_German_fixed/core17/German_jhu-clsp_core17-instructions-mteb_required_format_queries_xiaomi--mimo-v2-flash_free.jsonl",
        instruction_file="./CodeSwitching_Dataset_fixed/IR_German_fixed/core17/German_jhu-clsp_core17-instructions-mteb_required_format_instructions_xiaomi--mimo-v2-flash_free.jsonl"
    )
    task_news21_de = News21InstructionRetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/IR_German_fixed/news21/German_jhu-clsp_news21-instructions-mteb_required_format_queries_xiaomi--mimo-v2-flash_free.jsonl",
        instruction_file="./CodeSwitching_Dataset_fixed/IR_German_fixed/news21/German_jhu-clsp_news21-instructions-mteb_required_format_instructions_xiaomi--mimo-v2-flash_free.jsonl"
    )
    task_robust04_de = Robust04InstructionRetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/IR_German_fixed/robust04/German_jhu-clsp_robust04-instructions-mteb_required_format_queries_xiaomi--mimo-v2-flash_free.jsonl",
        instruction_file="./CodeSwitching_Dataset_fixed/IR_German_fixed/robust04/German_jhu-clsp_robust04-instructions-mteb_required_format_instructions_xiaomi--mimo-v2-flash_free.jsonl"
    )
    tasks = [task_HumanEval_de, task_core17_de, task_news21_de, task_robust04_de]
    return tasks


# ========== Dutch Code-Switching 任务加载函数 ==========

# ----- Classification -----
def load_TweetSentiment_Dutch():
    task = TweetSentimentExtractionClassificationCodeSwitching(
        test_file="./CodeSwitching_Dataset_fixed/Classification_Dutch_fixed/Dutch_mteb_tweet_sentiment_extraction_None_test_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


# ----- Clustering -----
def load_ArxivClustering_Dutch():
    task = ArXivHierarchicalClusteringP2PCodeSwitching(
        test_file="./CodeSwitching_Dataset_fixed/Clustering_Dutch_fixed/Dutch_mteb_arxiv-clustering-p2p_None_test_sampled_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


# ----- PairClassification -----
def load_TwitterSemEval_Dutch():
    task = TwitterSemEval2015CodeSwitching(
        test_file="./CodeSwitching_Dataset_fixed/PairClassification_Dutch_fixed/Dutch_mteb_twittersemeval2015-pairclassification_None_test_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


# ----- STS -----
def load_STSBenchmark_Dutch():
    task = STSBenchmarkCodeSwitching(
        data_file="./CodeSwitching_Dataset_fixed/STS_Dutch_fixed/Dutch_mteb_stsbenchmark-sts_None_test_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


# ----- Reranking -----
def load_AskUbuntu_Dutch():
    task = AskUbuntuDupQuestionsCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/Reranking_Dutch_fixed/Dutch_mteb_AskUbuntuDupQuestions_queries_test_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


# ----- Retrieval -----
def load_ArguAna_Dutch():
    task = ArguAnaCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLDutch_fixed/Dutch_mteb_arguana_queries_queries_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


def load_ClimateFEVER_Dutch():
    task = ClimateFEVERHardNegativesV2CodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLDutch_fixed/Dutch_mteb_ClimateFEVER_test_top_250_only_w_correct-v2_queries_test_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


def load_TRECCOVID_Dutch():
    task = TRECCOVIDCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLDutch_fixed/Dutch_mteb_trec-covid_queries_queries_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


def load_Touche2020RetrievalV3_Dutch():
    task = Touche2020v3RetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLDutch_fixed/Dutch_mteb_touche2020_queries_queries_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


def load_Touche2020v3_Dutch():
    task = Touche2020v3RetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLDutch_fixed/Dutch_mteb_webis-touche2020-v3_queries_train_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


def load_HumanEval_Dutch():
    task = HumanEvalRetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLDutch_fixed/Dutch_embedding-benchmark_HumanEval_queries_queries_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


# ----- IR -----
def load_Retrieval_And_IR_fixed_task_Dutch():
    task_HumanEval_nl = HumanEvalRetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLDutch_fixed/Dutch_embedding-benchmark_HumanEval_queries_queries_xiaomi--mimo-v2-flash_free.jsonl"
    )
    task_core17_nl = Core17InstructionRetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/IR_Dutch_fixed/core17/Dutch_jhu-clsp_core17-instructions-mteb_required_format_queries_xiaomi--mimo-v2-flash_free.jsonl",
        instruction_file="./CodeSwitching_Dataset_fixed/IR_Dutch_fixed/core17/Dutch_jhu-clsp_core17-instructions-mteb_required_format_instructions_xiaomi--mimo-v2-flash_free.jsonl"
    )
    task_news21_nl = News21InstructionRetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/IR_Dutch_fixed/news21/Dutch_jhu-clsp_news21-instructions-mteb_required_format_queries_xiaomi--mimo-v2-flash_free.jsonl",
        instruction_file="./CodeSwitching_Dataset_fixed/IR_Dutch_fixed/news21/Dutch_jhu-clsp_news21-instructions-mteb_required_format_instructions_xiaomi--mimo-v2-flash_free.jsonl"
    )
    task_robust04_nl = Robust04InstructionRetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/IR_Dutch_fixed/robust04/Dutch_jhu-clsp_robust04-instructions-mteb_required_format_queries_xiaomi--mimo-v2-flash_free.jsonl",
        instruction_file="./CodeSwitching_Dataset_fixed/IR_Dutch_fixed/robust04/Dutch_jhu-clsp_robust04-instructions-mteb_required_format_instructions_xiaomi--mimo-v2-flash_free.jsonl"
    )
    tasks = [task_HumanEval_nl, task_core17_nl, task_news21_nl, task_robust04_nl]
    return tasks


# ========== Korean Code-Switching 任务加载函数 ==========

# ----- Classification -----
def load_TweetSentiment_Korean():
    task = TweetSentimentExtractionClassificationCodeSwitching(
        test_file="./CodeSwitching_Dataset_fixed/Classification_Korean_fixed/Korean_mteb_tweet_sentiment_extraction_None_test_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


# ----- Clustering -----
def load_ArxivClustering_Korean():
    task = ArXivHierarchicalClusteringP2PCodeSwitching(
        test_file="./CodeSwitching_Dataset_fixed/Clustering_Korean_fixed/Korean_mteb_arxiv-clustering-p2p_None_test_sampled_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


# ----- PairClassification -----
def load_TwitterSemEval_Korean():
    task = TwitterSemEval2015CodeSwitching(
        test_file="./CodeSwitching_Dataset_fixed/PairClassification_Korean_fixed/Korean_mteb_twittersemeval2015-pairclassification_None_test_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


# ----- STS -----
def load_STSBenchmark_Korean():
    task = STSBenchmarkCodeSwitching(
        data_file="./CodeSwitching_Dataset_fixed/STS_Korean_fixed/Korean_mteb_stsbenchmark-sts_None_test_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


# ----- Reranking -----
def load_AskUbuntu_Korean():
    task = AskUbuntuDupQuestionsCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/Reranking_Korean_fixed/Korean_mteb_AskUbuntuDupQuestions_queries_test_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


# ----- Retrieval -----
def load_ArguAna_Korean():
    task = ArguAnaCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLKorean_fixed/Korean_mteb_arguana_queries_queries_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


def load_ClimateFEVER_Korean():
    task = ClimateFEVERHardNegativesV2CodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLKorean_fixed/Korean_mteb_ClimateFEVER_test_top_250_only_w_correct-v2_queries_test_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


def load_TRECCOVID_Korean():
    task = TRECCOVIDCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLKorean_fixed/Korean_mteb_trec-covid_queries_queries_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


def load_Touche2020RetrievalV3_Korean():
    task = Touche2020v3RetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLKorean_fixed/Korean_mteb_touche2020_queries_queries_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


def load_Touche2020v3_Korean():
    task = Touche2020v3RetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLKorean_fixed/Korean_mteb_webis-touche2020-v3_queries_train_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


def load_HumanEval_Korean():
    task = HumanEvalRetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLKorean_fixed/Korean_embedding-benchmark_HumanEval_queries_queries_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


# ----- IR -----
def load_Retrieval_And_IR_fixed_task_Korean():
    task_HumanEval_ko = HumanEvalRetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLKorean_fixed/Korean_embedding-benchmark_HumanEval_queries_queries_xiaomi--mimo-v2-flash_free.jsonl"
    )
    task_core17_ko = Core17InstructionRetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/IR_Korean_fixed/core17/Korean_jhu-clsp_core17-instructions-mteb_required_format_queries_xiaomi--mimo-v2-flash_free.jsonl",
        instruction_file="./CodeSwitching_Dataset_fixed/IR_Korean_fixed/core17/Korean_jhu-clsp_core17-instructions-mteb_required_format_instructions_xiaomi--mimo-v2-flash_free.jsonl"
    )
    task_news21_ko = News21InstructionRetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/IR_Korean_fixed/news21/Korean_jhu-clsp_news21-instructions-mteb_required_format_queries_xiaomi--mimo-v2-flash_free.jsonl",
        instruction_file="./CodeSwitching_Dataset_fixed/IR_Korean_fixed/news21/Korean_jhu-clsp_news21-instructions-mteb_required_format_instructions_xiaomi--mimo-v2-flash_free.jsonl"
    )
    task_robust04_ko = Robust04InstructionRetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/IR_Korean_fixed/robust04/Korean_jhu-clsp_robust04-instructions-mteb_required_format_queries_xiaomi--mimo-v2-flash_free.jsonl",
        instruction_file="./CodeSwitching_Dataset_fixed/IR_Korean_fixed/robust04/Korean_jhu-clsp_robust04-instructions-mteb_required_format_instructions_xiaomi--mimo-v2-flash_free.jsonl"
    )
    tasks = [task_HumanEval_ko, task_core17_ko, task_news21_ko, task_robust04_ko]
    return tasks


# ========== Portuguese Code-Switching 任务加载函数 ==========

# ----- Classification -----
def load_TweetSentiment_Portuguese():
    task = TweetSentimentExtractionClassificationCodeSwitching(
        test_file="./CodeSwitching_Dataset_fixed/Classification_Portuguese_fixed/Portuguese_mteb_tweet_sentiment_extraction_None_test_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


# ----- Clustering -----
def load_ArxivClustering_Portuguese():
    task = ArXivHierarchicalClusteringP2PCodeSwitching(
        test_file="./CodeSwitching_Dataset_fixed/Clustering_Portuguese_fixed/Portuguese_mteb_arxiv-clustering-p2p_None_test_sampled_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


# ----- PairClassification -----
def load_TwitterSemEval_Portuguese():
    task = TwitterSemEval2015CodeSwitching(
        test_file="./CodeSwitching_Dataset_fixed/PairClassification_Portuguese_fixed/Portuguese_mteb_twittersemeval2015-pairclassification_None_test_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


# ----- STS -----
def load_STSBenchmark_Portuguese():
    task = STSBenchmarkCodeSwitching(
        data_file="./CodeSwitching_Dataset_fixed/STS_Portuguese_fixed/Portuguese_mteb_stsbenchmark-sts_None_test_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


# ----- Reranking -----
def load_AskUbuntu_Portuguese():
    task = AskUbuntuDupQuestionsCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/Reranking_Portuguese_fixed/Portuguese_mteb_AskUbuntuDupQuestions_queries_test_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


# ----- Retrieval -----
def load_ArguAna_Portuguese():
    task = ArguAnaCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLPortuguese_fixed/Portuguese_mteb_arguana_queries_queries_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


def load_ClimateFEVER_Portuguese():
    task = ClimateFEVERHardNegativesV2CodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLPortuguese_fixed/Portuguese_mteb_ClimateFEVER_test_top_250_only_w_correct-v2_queries_test_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


def load_TRECCOVID_Portuguese():
    task = TRECCOVIDCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLPortuguese_fixed/Portuguese_mteb_trec-covid_queries_queries_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


def load_Touche2020RetrievalV3_Portuguese():
    task = Touche2020v3RetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLPortuguese_fixed/Portuguese_mteb_touche2020_queries_queries_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


# ----- IR -----
def load_Retrieval_And_IR_fixed_task_Portuguese():
    task_core17_pt = Core17InstructionRetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/IR_Portuguese_fixed/core17/Portuguese_jhu-clsp_core17-instructions-mteb_required_format_queries_xiaomi--mimo-v2-flash_free.jsonl",
        instruction_file="./CodeSwitching_Dataset_fixed/IR_Portuguese_fixed/core17/Portuguese_jhu-clsp_core17-instructions-mteb_required_format_instructions_xiaomi--mimo-v2-flash_free.jsonl"
    )
    task_news21_pt = News21InstructionRetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/IR_Portuguese_fixed/news21/Portuguese_jhu-clsp_news21-instructions-mteb_required_format_queries_xiaomi--mimo-v2-flash_free.jsonl",
        instruction_file="./CodeSwitching_Dataset_fixed/IR_Portuguese_fixed/news21/Portuguese_jhu-clsp_news21-instructions-mteb_required_format_instructions_xiaomi--mimo-v2-flash_free.jsonl"
    )
    task_robust04_pt = Robust04InstructionRetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/IR_Portuguese_fixed/robust04/Portuguese_jhu-clsp_robust04-instructions-mteb_required_format_queries_xiaomi--mimo-v2-flash_free.jsonl",
        instruction_file="./CodeSwitching_Dataset_fixed/IR_Portuguese_fixed/robust04/Portuguese_jhu-clsp_robust04-instructions-mteb_required_format_instructions_xiaomi--mimo-v2-flash_free.jsonl"
    )
    tasks = [task_core17_pt, task_news21_pt, task_robust04_pt]
    return tasks


def load_Touche2020v3_Portuguese():
    task = Touche2020v3RetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLPortuguese_fixed/Portuguese_mteb_webis-touche2020-v3_queries_train_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


def load_HumanEval_Portuguese():
    task = HumanEvalRetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLPortuguese_fixed/Portuguese_embedding-benchmark_HumanEval_queries_queries_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


# ========== Italian Code-Switching 任务加载函数 ==========

# ----- Classification -----
def load_TweetSentiment_Italian():
    task = TweetSentimentExtractionClassificationCodeSwitching(
        test_file="./CodeSwitching_Dataset_fixed/Classification_Italian_fixed/Italian_mteb_tweet_sentiment_extraction_None_test_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


# ----- Clustering -----
def load_ArxivClustering_Italian():
    task = ArXivHierarchicalClusteringP2PCodeSwitching(
        test_file="./CodeSwitching_Dataset_fixed/Clustering_Italian_fixed/Italian_mteb_arxiv-clustering-p2p_None_test_sampled_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


# ----- PairClassification -----
def load_TwitterSemEval_Italian():
    task = TwitterSemEval2015CodeSwitching(
        test_file="./CodeSwitching_Dataset_fixed/PairClassification_Italian_fixed/Italian_mteb_twittersemeval2015-pairclassification_None_test_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


# ----- STS -----
def load_STSBenchmark_Italian():
    task = STSBenchmarkCodeSwitching(
        data_file="./CodeSwitching_Dataset_fixed/STS_Italian_fixed/Italian_mteb_stsbenchmark-sts_None_test_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


# ----- Reranking -----
def load_AskUbuntu_Italian():
    task = AskUbuntuDupQuestionsCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/Reranking_Italian_fixed/Italian_mteb_AskUbuntuDupQuestions_queries_test_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


# ----- Retrieval -----
def load_ArguAna_Italian():
    task = ArguAnaCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLItalian_fixed/Italian_mteb_arguana_queries_queries_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


def load_ClimateFEVER_Italian():
    task = ClimateFEVERHardNegativesV2CodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLItalian_fixed/Italian_mteb_ClimateFEVER_test_top_250_only_w_correct-v2_queries_test_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


def load_TRECCOVID_Italian():
    task = TRECCOVIDCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLItalian_fixed/Italian_mteb_trec-covid_queries_queries_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


def load_Touche2020RetrievalV3_Italian():
    task = Touche2020v3RetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLItalian_fixed/Italian_mteb_touche2020_queries_queries_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


# ----- IR -----
def load_Retrieval_And_IR_fixed_task_Italian():
    task_core17_it = Core17InstructionRetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/IR_Italian_fixed/core17/Italian_jhu-clsp_core17-instructions-mteb_required_format_queries_xiaomi--mimo-v2-flash_free.jsonl",
        instruction_file="./CodeSwitching_Dataset_fixed/IR_Italian_fixed/core17/Italian_jhu-clsp_core17-instructions-mteb_required_format_instructions_xiaomi--mimo-v2-flash_free.jsonl"
    )
    task_news21_it = News21InstructionRetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/IR_Italian_fixed/news21/Italian_jhu-clsp_news21-instructions-mteb_required_format_queries_xiaomi--mimo-v2-flash_free.jsonl",
        instruction_file="./CodeSwitching_Dataset_fixed/IR_Italian_fixed/news21/Italian_jhu-clsp_news21-instructions-mteb_required_format_instructions_xiaomi--mimo-v2-flash_free.jsonl"
    )
    task_robust04_it = Robust04InstructionRetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/IR_Italian_fixed/robust04/Italian_jhu-clsp_robust04-instructions-mteb_required_format_queries_xiaomi--mimo-v2-flash_free.jsonl",
        instruction_file="./CodeSwitching_Dataset_fixed/IR_Italian_fixed/robust04/Italian_jhu-clsp_robust04-instructions-mteb_required_format_instructions_xiaomi--mimo-v2-flash_free.jsonl"
    )
    tasks = [task_core17_it, task_news21_it, task_robust04_it]
    return tasks


def load_Touche2020v3_Italian():
    task = Touche2020v3RetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLItalian_fixed/Italian_mteb_webis-touche2020-v3_queries_train_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


def load_HumanEval_Italian():
    task = HumanEvalRetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLItalian_fixed/Italian_embedding-benchmark_HumanEval_queries_queries_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


# ========== Spanish Code-Switching 任务加载函数 ==========

# ----- Classification -----
def load_TweetSentiment_Spanish():
    task = TweetSentimentExtractionClassificationCodeSwitching(
        test_file="./CodeSwitching_Dataset_fixed/Classification_Spanish_fixed/Spanish_mteb_tweet_sentiment_extraction_None_test_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]

def load_ArxivClustering_Spanish():
    task = ArXivHierarchicalClusteringP2PCodeSwitching(
        test_file="./CodeSwitching_Dataset_fixed/Clustering_Spanish_fixed/Spanish_mteb_arxiv-clustering-p2p_None_test_sampled_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]

# ----- PairClassification -----
def load_TwitterSemEval_Spanish():
    task = TwitterSemEval2015CodeSwitching(
        test_file="./CodeSwitching_Dataset_fixed/PairClassification_Spanish_fixed/Spanish_mteb_twittersemeval2015-pairclassification_None_test_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


# ----- STS -----
def load_STSBenchmark_Spanish():
    task = STSBenchmarkCodeSwitching(
        data_file="./CodeSwitching_Dataset_fixed/STS_Spanish_fixed/Spanish_mteb_stsbenchmark-sts_None_test_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


# ----- Reranking -----
def load_AskUbuntu_Spanish():
    task = AskUbuntuDupQuestionsCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/Reranking_Spanish_fixed/Spanish_mteb_AskUbuntuDupQuestions_queries_test_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


# ----- Retrieval -----
def load_ArguAna_Spanish():
    task = ArguAnaCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLSpanish_fixed/Spanish_mteb_arguana_queries_queries_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


def load_ClimateFEVER_Spanish():
    task = ClimateFEVERHardNegativesV2CodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLSpanish_fixed/Spanish_mteb_ClimateFEVER_test_top_250_only_w_correct-v2_queries_test_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


def load_TRECCOVID_Spanish():
    task = TRECCOVIDCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLSpanish_fixed/Spanish_mteb_trec-covid_queries_queries_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


def load_Touche2020RetrievalV3_Spanish():
    task = Touche2020v3RetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLSpanish_fixed/Spanish_mteb_touche2020_queries_queries_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


def load_Touche2020v3_Spanish():
    task = Touche2020v3RetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLSpanish_fixed/Spanish_mteb_webis-touche2020-v3_queries_train_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


def load_HumanEval_Spanish():
    task = HumanEvalRetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLSpanish_fixed/Spanish_embedding-benchmark_HumanEval_queries_queries_xiaomi--mimo-v2-flash_free.jsonl"
    )
    return [task]


# ----- IR -----
def load_Retrieval_And_IR_fixed_task_Spanish():
    task_HumanEval_es = HumanEvalRetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/RetrievaLSpanish_fixed/Spanish_embedding-benchmark_HumanEval_queries_queries_xiaomi--mimo-v2-flash_free.jsonl"
    )
    task_core17_es = Core17InstructionRetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/IR_Spanish_fixed/core17/Spanish_jhu-clsp_core17-instructions-mteb_required_format_queries_xiaomi--mimo-v2-flash_free.jsonl",
        instruction_file="./CodeSwitching_Dataset_fixed/IR_Spanish_fixed/core17/Spanish_jhu-clsp_core17-instructions-mteb_required_format_instructions_xiaomi--mimo-v2-flash_free.jsonl"
    )
    task_news21_es = News21InstructionRetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/IR_Spanish_fixed/news21/Spanish_jhu-clsp_news21-instructions-mteb_required_format_queries_xiaomi--mimo-v2-flash_free.jsonl",
        instruction_file="./CodeSwitching_Dataset_fixed/IR_Spanish_fixed/news21/Spanish_jhu-clsp_news21-instructions-mteb_required_format_instructions_xiaomi--mimo-v2-flash_free.jsonl"
    )
    task_robust04_es = Robust04InstructionRetrievalCodeSwitching(
        query_file="./CodeSwitching_Dataset_fixed/IR_Spanish_fixed/robust04/Spanish_jhu-clsp_robust04-instructions-mteb_required_format_queries_xiaomi--mimo-v2-flash_free.jsonl",
        instruction_file="./CodeSwitching_Dataset_fixed/IR_Spanish_fixed/robust04/Spanish_jhu-clsp_robust04-instructions-mteb_required_format_instructions_xiaomi--mimo-v2-flash_free.jsonl"
    )
    tasks = [task_HumanEval_es, task_core17_es, task_news21_es, task_robust04_es]
    return tasks


# ----- French 大整合函数 -----
def load_all_new_tasks_French():
    """加载所有新增的法语 Code-Switching 任务"""
    tasks = []
    tasks.extend(load_TweetSentiment_French())
    tasks.extend(load_ArxivClustering_French())
    tasks.extend(load_TwitterSemEval_French())
    tasks.extend(load_STSBenchmark_French())
    tasks.extend(load_AskUbuntu_French())
    tasks.extend(load_ArguAna_French())
    tasks.extend(load_ClimateFEVER_French())
    return tasks


# ----- German 大整合函数 -----
def load_all_new_tasks_German():
    """加载所有新增的德语 Code-Switching 任务"""
    tasks = []
    tasks.extend(load_TweetSentiment_German())
    tasks.extend(load_ArxivClustering_German())
    tasks.extend(load_TwitterSemEval_German())
    tasks.extend(load_STSBenchmark_German())
    tasks.extend(load_AskUbuntu_German())
    tasks.extend(load_ArguAna_German())
    tasks.extend(load_ClimateFEVER_German())
    return tasks


# ========== 原始任务加载函数（对应 Code-Switching 版本）==========

# ----- Classification -----
def load_TweetSentiment_Original():
    task = TweetSentimentExtractionClassificationV2()
    return [task]


# ----- Clustering -----
def load_ArxivClustering_Original():
    # Since we used the sampled dataset there, we load the sampled file through the code-switching class
    # Note that the code-switching version class only changed the data loader, so this is valid.
    task = ArXivHierarchicalClusteringP2PCodeSwitching(test_file="./CodeSwitching_Dataset_fixed/Clustering_OG/mteb_arxiv-clustering-p2p_None_test_sampled.jsonl")
    return [task]


# ----- PairClassification -----
def load_TwitterSemEval_Original():
    task = TwitterSemEval2015PC()
    return [task]


# ----- STS -----
def load_STSBenchmark_Original():
    task = STSBenchmarkSTS()
    return [task]


# ----- Reranking -----
def load_AskUbuntu_Original():
    task = AskUbuntuDupQuestions()
    return [task]


# ----- Retrieval -----
def load_ArguAna_Original():
    task = ArguAna()
    return [task]


def load_ClimateFEVER_Original():
    task = ClimateFEVERHardNegativesV2()
    return [task]


# ----- 大整合函数 -----
def load_all_new_tasks_Original():
    """加载所有新增任务的原始版本"""
    tasks = []
    tasks.extend(load_TweetSentiment_Original())
    tasks.extend(load_ArxivClustering_Original())
    tasks.extend(load_TwitterSemEval_Original())
    tasks.extend(load_STSBenchmark_Original())
    tasks.extend(load_AskUbuntu_Original())
    tasks.extend(load_ArguAna_Original())
    tasks.extend(load_ClimateFEVER_Original())
    return tasks


def load_all_new_tasks_Chinese():
    """加载所有新增的中文 Code-Switching 任务"""
    tasks = []
    tasks.extend(load_TweetSentiment_Chinese())
    tasks.extend(load_ArxivClustering_Chinese())
    tasks.extend(load_TwitterSemEval_Chinese())
    tasks.extend(load_STSBenchmark_Chinese())
    tasks.extend(load_AskUbuntu_Chinese())
    tasks.extend(load_ArguAna_Chinese())
    tasks.extend(load_ClimateFEVER_Chinese())
    return tasks


def load_all_new_tasks_Japanese():
    """加载所有新增的日文 Code-Switching 任务"""
    tasks = []
    tasks.extend(load_TweetSentiment_Japanese())
    tasks.extend(load_ArxivClustering_Japanese())
    tasks.extend(load_TwitterSemEval_Japanese())
    tasks.extend(load_STSBenchmark_Japanese())
    tasks.extend(load_AskUbuntu_Japanese())
    tasks.extend(load_ArguAna_Japanese())
    tasks.extend(load_ClimateFEVER_Japanese())
    return tasks


# ========== 主函数 ==========

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="MTEB configuration selection"
    )
    parser.add_argument("--mode", choices=["batch", "single"], default="batch")
    parser.add_argument("--tasks", nargs='+', help="list of the tasks in types you need to run", type=str, required=True)
    parser.add_argument("--model_path", required=True, help="Either hf path or local path", type=str)
    parser.add_argument("--batch_size", required=True, help="batch_size", type=int, default=32)
    parser.add_argument("--output_subfolder_name", help="subfolder name for output", type=str, default=None)
    parser.add_argument(
        "--evaluation_output_dir",
        required=True,
        help="Evaluation output directory",
        type=str,
    )

    args = parser.parse_args()

    current_time = datetime.now()
    date = current_time.strftime("%Y%m%d")

    if args.output_subfolder_name:
        output_dir = os.path.join(args.evaluation_output_dir, "20251231", args.output_subfolder_name)
    else:
        output_dir = os.path.join(args.evaluation_output_dir, date)

    All_tasks_names = []

    if args.mode == "batch":
        for task in args.tasks:
            if task == "Original":
                curr_tasks = load_original_tasks()
            elif task == "OG_Retrieval":
                curr_tasks = load_og_Retrieval_tasks()
            elif task == "Fixed_Chinese":
                curr_tasks = load_Retrieval_And_IR_fixed_task_Chinese()
            elif task == "Fixed_Japanese":
                curr_tasks = load_Retrieval_And_IR_fixed_task_Japanese()
            elif task == "Fixed_R_Chinese":
                curr_tasks = load_Retrieval_Chinese()
            elif task == "Fixed_R_Japanese":
                curr_tasks = load_Retrieval_Japanese()
            elif task == "IR_cs":
                curr_tasks = load_IR_code_switching_task()
            elif task == "touchev3_ch":
                curr_tasks = load_Touche2020RetrievalV3_ch()
            elif task == "touchev3_jp":
                curr_tasks = load_Touche2020RetrievalV3_jp()
            elif task == "touchev3":
                curr_tasks = load_touche2020RetrievalV3()
            elif task == "treccovid":
                curr_tasks = load_TRECCOVID_tasks()
            elif task == "treccovid_ch":
                curr_tasks = load_TRECCOVID_Chinese_tasks()
            elif task == "treccovid_jp":
                curr_tasks = load_TRECCOVID_Japanese_tasks()
            # ----- 新增任务 -----
            elif task == "tweet_sentiment_ch":
                curr_tasks = load_TweetSentiment_Chinese()
            elif task == "tweet_sentiment_jp":
                curr_tasks = load_TweetSentiment_Japanese()
            elif task == "arxiv_clustering_ch":
                curr_tasks = load_ArxivClustering_Chinese()
            elif task == "arxiv_clustering_jp":
                curr_tasks = load_ArxivClustering_Japanese()
            elif task == "twitter_semeval_ch":
                curr_tasks = load_TwitterSemEval_Chinese()
            elif task == "twitter_semeval_jp":
                curr_tasks = load_TwitterSemEval_Japanese()
            elif task == "stsbenchmark_ch":
                curr_tasks = load_STSBenchmark_Chinese()
            elif task == "stsbenchmark_jp":
                curr_tasks = load_STSBenchmark_Japanese()
            elif task == "askubuntu_ch":
                curr_tasks = load_AskUbuntu_Chinese()
            elif task == "askubuntu_jp":
                curr_tasks = load_AskUbuntu_Japanese()
            elif task == "arguana_ch":
                curr_tasks = load_ArguAna_Chinese()
            elif task == "arguana_jp":
                curr_tasks = load_ArguAna_Japanese()
            elif task == "climatefever_ch":
                curr_tasks = load_ClimateFEVER_Chinese()
            elif task == "climatefever_jp":
                curr_tasks = load_ClimateFEVER_Japanese()
            # ----- French 任务 -----
            elif task == "Fixed_French":
                curr_tasks = load_Retrieval_And_IR_fixed_task_French()
            elif task == "Fixed_R_French":
                curr_tasks = load_Retrieval_French()
            elif task == "treccovid_fr":
                curr_tasks = load_TRECCOVID_French()
            elif task == "touchev3_fr":
                curr_tasks = load_Touche2020RetrievalV3_French()
            elif task == "humaneval_fr":
                curr_tasks = load_HumanEval_French()
            elif task == "tweet_sentiment_fr":
                curr_tasks = load_TweetSentiment_French()
            elif task == "arxiv_clustering_fr":
                curr_tasks = load_ArxivClustering_French()
            elif task == "twitter_semeval_fr":
                curr_tasks = load_TwitterSemEval_French()
            elif task == "stsbenchmark_fr":
                curr_tasks = load_STSBenchmark_French()
            elif task == "askubuntu_fr":
                curr_tasks = load_AskUbuntu_French()
            elif task == "arguana_fr":
                curr_tasks = load_ArguAna_French()
            elif task == "climatefever_fr":
                curr_tasks = load_ClimateFEVER_French()
            # ----- German 任务 -----
            elif task == "Fixed_German":
                curr_tasks = load_Retrieval_And_IR_fixed_task_German()
            elif task == "Fixed_R_German":
                curr_tasks = load_Retrieval_German()
            elif task == "treccovid_de":
                curr_tasks = load_TRECCOVID_German()
            elif task == "touchev3_de":
                curr_tasks = load_Touche2020RetrievalV3_German()
            elif task == "humaneval_de":
                curr_tasks = load_HumanEval_German()
            elif task == "tweet_sentiment_de":
                curr_tasks = load_TweetSentiment_German()
            elif task == "arxiv_clustering_de":
                curr_tasks = load_ArxivClustering_German()
            elif task == "twitter_semeval_de":
                curr_tasks = load_TwitterSemEval_German()
            elif task == "stsbenchmark_de":
                curr_tasks = load_STSBenchmark_German()
            elif task == "askubuntu_de":
                curr_tasks = load_AskUbuntu_German()
            elif task == "arguana_de":
                curr_tasks = load_ArguAna_German()
            elif task == "climatefever_de":
                curr_tasks = load_ClimateFEVER_German()
            # ----- Dutch 任务 -----
            elif task == "treccovid_nl":
                curr_tasks = load_TRECCOVID_Dutch()
            elif task == "touchev3_nl":
                curr_tasks = load_Touche2020RetrievalV3_Dutch()
            elif task == "tweet_sentiment_nl":
                curr_tasks = load_TweetSentiment_Dutch()
            elif task == "arxiv_clustering_nl":
                curr_tasks = load_ArxivClustering_Dutch()
            elif task == "twitter_semeval_nl":
                curr_tasks = load_TwitterSemEval_Dutch()
            elif task == "stsbenchmark_nl":
                curr_tasks = load_STSBenchmark_Dutch()
            elif task == "askubuntu_nl":
                curr_tasks = load_AskUbuntu_Dutch()
            elif task == "arguana_nl":
                curr_tasks = load_ArguAna_Dutch()
            elif task == "climatefever_nl":
                curr_tasks = load_ClimateFEVER_Dutch()
            elif task == "Fixed_Dutch":
                curr_tasks = load_Retrieval_And_IR_fixed_task_Dutch()
            elif task == "humaneval_nl":
                curr_tasks = load_HumanEval_Dutch()
            elif task == "touche2020v3_nl":
                curr_tasks = load_Touche2020v3_Dutch()
            # ----- Korean 任务 -----
            elif task == "Fixed_Korean":
                curr_tasks = load_Retrieval_And_IR_fixed_task_Korean()
            elif task == "treccovid_ko":
                curr_tasks = load_TRECCOVID_Korean()
            elif task == "touchev3_ko":
                curr_tasks = load_Touche2020RetrievalV3_Korean()
            elif task == "touche2020v3_ko":
                curr_tasks = load_Touche2020v3_Korean()
            elif task == "humaneval_ko":
                curr_tasks = load_HumanEval_Korean()
            elif task == "tweet_sentiment_ko":
                curr_tasks = load_TweetSentiment_Korean()
            elif task == "arxiv_clustering_ko":
                curr_tasks = load_ArxivClustering_Korean()
            elif task == "twitter_semeval_ko":
                curr_tasks = load_TwitterSemEval_Korean()
            elif task == "stsbenchmark_ko":
                curr_tasks = load_STSBenchmark_Korean()
            elif task == "askubuntu_ko":
                curr_tasks = load_AskUbuntu_Korean()
            elif task == "arguana_ko":
                curr_tasks = load_ArguAna_Korean()
            elif task == "climatefever_ko":
                curr_tasks = load_ClimateFEVER_Korean()
            # ----- Portuguese 任务 -----
            elif task == "Fixed_Portuguese":
                curr_tasks = load_Retrieval_And_IR_fixed_task_Portuguese()
            elif task == "treccovid_pt":
                curr_tasks = load_TRECCOVID_Portuguese()
            elif task == "touchev3_pt":
                curr_tasks = load_Touche2020RetrievalV3_Portuguese()
            elif task == "tweet_sentiment_pt":
                curr_tasks = load_TweetSentiment_Portuguese()
            elif task == "arxiv_clustering_pt":
                curr_tasks = load_ArxivClustering_Portuguese()
            elif task == "twitter_semeval_pt":
                curr_tasks = load_TwitterSemEval_Portuguese()
            elif task == "stsbenchmark_pt":
                curr_tasks = load_STSBenchmark_Portuguese()
            elif task == "askubuntu_pt":
                curr_tasks = load_AskUbuntu_Portuguese()
            elif task == "arguana_pt":
                curr_tasks = load_ArguAna_Portuguese()
            elif task == "climatefever_pt":
                curr_tasks = load_ClimateFEVER_Portuguese()
            elif task == "humaneval_pt":
                curr_tasks = load_HumanEval_Portuguese()
            elif task == "touche2020v3_pt":
                curr_tasks = load_Touche2020v3_Portuguese()
            # ----- Italian 任务 -----
            elif task == "Fixed_Italian":
                curr_tasks = load_Retrieval_And_IR_fixed_task_Italian()
            elif task == "treccovid_it":
                curr_tasks = load_TRECCOVID_Italian()
            elif task == "touchev3_it":
                curr_tasks = load_Touche2020RetrievalV3_Italian()
            elif task == "tweet_sentiment_it":
                curr_tasks = load_TweetSentiment_Italian()
            elif task == "arxiv_clustering_it":
                curr_tasks = load_ArxivClustering_Italian()
            elif task == "twitter_semeval_it":
                curr_tasks = load_TwitterSemEval_Italian()
            elif task == "stsbenchmark_it":
                curr_tasks = load_STSBenchmark_Italian()
            elif task == "askubuntu_it":
                curr_tasks = load_AskUbuntu_Italian()
            elif task == "arguana_it":
                curr_tasks = load_ArguAna_Italian()
            elif task == "climatefever_it":
                curr_tasks = load_ClimateFEVER_Italian()
            elif task == "humaneval_it":
                curr_tasks = load_HumanEval_Italian()
            elif task == "touche2020v3_it":
                curr_tasks = load_Touche2020v3_Italian()
            # ----- Spanish 任务 -----
            elif task == "Fixed_Spanish":
                curr_tasks = load_Retrieval_And_IR_fixed_task_Spanish()
            elif task == "arxiv_clustering_es":
                curr_tasks = load_ArxivClustering_Spanish()
            elif task == "treccovid_es":
                curr_tasks = load_TRECCOVID_Spanish()
            elif task == "touchev3_es":
                curr_tasks = load_Touche2020RetrievalV3_Spanish()
            elif task == "touche2020v3_es":
                curr_tasks = load_Touche2020v3_Spanish()
            elif task == "humaneval_es":
                curr_tasks = load_HumanEval_Spanish()
            elif task == "tweet_sentiment_es":
                curr_tasks = load_TweetSentiment_Spanish()
            elif task == "twitter_semeval_es":
                curr_tasks = load_TwitterSemEval_Spanish()
            elif task == "stsbenchmark_es":
                curr_tasks = load_STSBenchmark_Spanish()
            elif task == "askubuntu_es":
                curr_tasks = load_AskUbuntu_Spanish()
            elif task == "arguana_es":
                curr_tasks = load_ArguAna_Spanish()
            elif task == "climatefever_es":
                curr_tasks = load_ClimateFEVER_Spanish()
            # ----- 原始任务（对应 Code-Switching 版本）-----
            elif task == "tweet_sentiment_og":
                curr_tasks = load_TweetSentiment_Original()
            elif task == "arxiv_clustering_og":
                curr_tasks = load_ArxivClustering_Original()
            elif task == "twitter_semeval_og":
                curr_tasks = load_TwitterSemEval_Original()
            elif task == "stsbenchmark_og":
                curr_tasks = load_STSBenchmark_Original()
            elif task == "askubuntu_og":
                curr_tasks = load_AskUbuntu_Original()
            elif task == "arguana_og":
                curr_tasks = load_ArguAna_Original()
            elif task == "climatefever_og":
                curr_tasks = load_ClimateFEVER_Original()
            # ----- 大整合 -----
            elif task == "all_new_og":
                curr_tasks = load_all_new_tasks_Original()
            elif task == "all_new_ch":
                curr_tasks = load_all_new_tasks_Chinese()
            elif task == "all_new_jp":
                curr_tasks = load_all_new_tasks_Japanese()
            elif task == "all_new_fr":
                curr_tasks = load_all_new_tasks_French()
            elif task == "all_new_de":
                curr_tasks = load_all_new_tasks_German()
            else:
                print(f"ERROR: Task {task} is not supported!")
                continue
            All_tasks_names.extend(curr_tasks)
    else:
        for task in args.tasks:
            All_tasks_names.append(task)

    # 根据模型路径选择加载方式
    if "aligned" in args.model_path.lower():
        print(f"[INFO] 检测到 aligned 模型，使用 load_model_ST 加载（支持 prompt）")
        model = load_model_ST(args.model_path)
    else:
        print(f"[INFO] 使用 load_model 加载（MTEB 内置配置）")
        model = load_model(args.model_path)

    evaluation = MTEB(tasks=All_tasks_names)

    model_name = os.path.basename(args.model_path)
    results = evaluation.run(
        model,
        encode_kwargs={"batch_size": args.batch_size, "show_progress_bar": True},
        output_folder=output_dir,
        model_name=model_name,
    )
    print(results)
