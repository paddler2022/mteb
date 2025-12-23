import mteb
import os
import json
from mteb import MTEB
from mteb.types import PromptType
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper
from mteb.models.instruct_wrapper import InstructSentenceTransformerModel
from sentence_transformers import SentenceTransformer
from mteb.tasks.retrieval.eng import TRECCOVIDCodeSwitching, SCIDOCSCodeSwitchingRetrieval, ArguAnaCodeSwitching, ClimateFEVERHardNegativesCodeSwitching, CQADupstackGamingRetrievalCodeSwitching, FiQA2018CodeSwitching, TRECCOVID, SCIDOCS, HotpotQAHardNegativesCodeSwitching, Touche2020CodeSwitching, Touche2020, Touche2020v3Retrieval, Touche2020v3RetrievalCodeSwitching
from mteb.tasks.instruction_reranking.eng import Core17InstructionRetrievalCodeSwitching, News21InstructionRetrievalCodeSwitching, Robust04InstructionRetrievalCodeSwitching, Core17InstructionRetrieval, News21InstructionRetrieval, Robust04InstructionRetrieval
from mteb.tasks.classification.multilingual import MassiveIntentClassificationCodeSwitching, AmazonCounterfactualClassificationCodeSwitching
from mteb.tasks.reranking.eng import AskUbuntuDupQuestionsCodeSwitching
from mteb.tasks.sts.eng import SickrCodeSwitchingSTS
from mteb.tasks.sts.multilingual import STS22CodeSwitching
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
    "TNews": "Classify the fine-grained category of the given news title",
    "IFlyTek": "Given an App description text, find the appropriate fine-grained category",
    "MultilingualSentiment": "Classify sentiment of the customer review into positive, neutral, or negative",
    "JDReview": "Classify the customer review for iPhone on e-commerce platform into positive or negative",
    "OnlineShopping": "Classify the customer review for online shopping into positive or negative",
    "Waimai": "Classify the customer review from a food takeaway platform into positive or negative",
    # Clustering tasks
    "ArxivClusteringP2P": "Identify the main and secondary category of Arxiv papers based on the titles and abstracts",
    "ArxivClusteringS2S": "Identify the main and secondary category of Arxiv papers based on the titles",
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
    "BIOSSES": "Retrieve semantically similar text",
    "SICK-R": "Retrieve semantically similar text",
    "SickrCodeSwitchingSTS": "Retrieve semantically similar text",
    "SummEval": "Retrieve semantically similar text",
    # Pair Classification tasks
    "PawsX": "Retrieve semantically similar text",
    "XNLI": "Retrieve semantically similar text",
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
    if "qwen" in model_path.lower():
        kwargs["attn_implementation"] = "flash_attention_2"
        print("[INFO] 使用 Flash Attention 2")
    else:
        print("[INFO] 使用默认 Attention 实现")
    return kwargs


def load_model(model_path):
    """加载 HuggingFace 模型（使用 MTEB 内置配置）"""
    # if "e5" in model_path.lower():
    #     model = mteb.get_model(model_path)
    # else:
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
    model_kwargs = get_model_kwargs()

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

# def load_model(model_path):
#     """加载 HuggingFace 模型（使用 MTEB 内置配置）"""
#     if "e5" in model_path.lower():
#         model = mteb.get_model(model_path)
#     else:
#         model = mteb.get_model(
#             model_path,
#             model_kwargs={
#                 "attn_implementation": "flash_attention_2",
#                 "torch_dtype": torch.bfloat16
#             },
#             trust_remote_code = True,
#         )
#     return model
#
#
# def load_model_ST(model_path):
#     """
#     加载本地模型并支持 prompt
#
#     根据模型类型自动选择正确的 wrapper 和 prompt 配置:
#     - E5 系列: 使用 query:/passage: 前缀
#     - Qwen Embedding 系列: 使用 Instruct 模板
#     - 其他模型: 暂未使用，先用默认prompt
#     """
#     model_path_lower = model_path.lower()
#
#     # E5 系列模型
#     if "e5" in model_path_lower:
#         print(f"[INFO] 检测到 E5 模型，使用 query:/passage: prompt")
#         model = SentenceTransformerEncoderWrapper(
#             model=model_path,
#             revision=None,
#             model_prompts=E5_PROMPTS,
#             trust_remote_code=True
#         )
#         return model
#
#     # Qwen3 Embedding 系列模型
#     elif "qwen" in model_path_lower and "embedding" in model_path_lower:
#         print(f"[INFO] 检测到 Qwen Embedding 模型，使用 Instruct 模板")
#         model = InstructSentenceTransformerModel(
#             model_name=model_path,
#             revision=None,
#             instruction_template=qwen3_instruction_template,
#             apply_instruction_to_passages=False,
#             prompts_dict=QWEN3_PROMPTS,
#             model_kwargs={
#                 "attn_implementation": "flash_attention_2",
#                 "torch_dtype": torch.bfloat16,
#                 "device_map": "cuda"
#             },
#             trust_remote_code=True,
#         )
#         # Qwen3 在 flash_attention_2 模式下需要 left padding
#         if hasattr(model, 'model') and hasattr(model.model, 'tokenizer'):
#             model.model.tokenizer.padding_side = 'left'
#         return model
#
#     # 其他模型（无特殊 prompt）
#     else:
#         print(f"[INFO] 未检测到特殊 prompt 需求，使用默认配置")
#         model = SentenceTransformer(
#             model_path,
#             model_kwargs={
#                 "attn_implementation": "flash_attention_2",
#                 "torch_dtype": torch.bfloat16,
#                 "device_map": "cuda"
#             },
#             device="cuda"
#         )
#         model.tokenizer.padding_side = 'left'
#         return model


# ========== 任务加载函数 ==========

def load_classification_code_switching_task():
    task_amazon_counterfactual_cs = AmazonCounterfactualClassificationCodeSwitching(
        validation_file="/root/autodl-tmp/workdir/mteb/Code_Switching_dataset/Classification/mteb_amazon_counterfactual_en_validation_gpt-5-mini.jsonl",
        test_file="/root/autodl-tmp/workdir/mteb/Code_Switching_dataset/Classification/mteb_amazon_counterfactual_en_test_gpt-5-mini.jsonl",
    )
    task_massive_intent_cs = MassiveIntentClassificationCodeSwitching(
        validation_file="/root/autodl-tmp/workdir/mteb/Code_Switching_dataset/Classification/mteb_amazon_massive_intent_en_validation_gpt-5-mini.jsonl",
        test_file="/root/autodl-tmp/workdir/mteb/Code_Switching_dataset/Classification/mteb_amazon_massive_intent_en_test_gpt-5-mini.jsonl",
    )
    classification_cs_tasks = [task_amazon_counterfactual_cs, task_massive_intent_cs]
    return classification_cs_tasks


def load_reranking_code_switching_task():
    task_Ubuntu_dup_cs = AskUbuntuDupQuestionsCodeSwitching(query_file="/root/autodl-tmp/workdir/mteb/Code_Switching_dataset/Reranking/mteb_AskUbuntuDupQuestions_queries_test_gpt-5-mini.jsonl")
    return [task_Ubuntu_dup_cs]


def load_retrieval_code_switching_task():
    task_trec_covid_cs = TRECCOVIDCodeSwitching(query_file="/root/autodl-tmp/workdir/mteb/Code_Switching_dataset/Retrieval/mteb_trec-covid_queries_queries_gpt-5-mini.jsonl")
    task_ArguAna_cs = ArguAnaCodeSwitching(query_file="/root/autodl-tmp/workdir/mteb/Code_Switching_dataset/Retrieval/mteb_arguana_queries_queries_gpt-5-mini.jsonl")
    task_ClimateFEVERHardNegatives_cs = ClimateFEVERHardNegativesCodeSwitching(query_file="/root/autodl-tmp/workdir/mteb/Code_Switching_dataset/Retrieval/mteb_climate-fever_queries_queries_gpt-5-mini.jsonl")
    task_scidoc_cs = SCIDOCSCodeSwitchingRetrieval(query_file="/root/autodl-tmp/workdir/mteb/Code_Switching_dataset/Retrieval/mteb_scidocs_queries_queries_gpt-5-mini.jsonl")
    task_cqa_gaming_cs = CQADupstackGamingRetrievalCodeSwitching(query_file="/root/autodl-tmp/workdir/mteb/Code_Switching_dataset/Retrieval/mteb_cqadupstack-gaming_queries_queries_gpt-5-mini.jsonl")
    task_fiqa_cs = FiQA2018CodeSwitching(query_file="/root/autodl-tmp/workdir/mteb/Code_Switching_dataset/Retrieval/mteb_fiqa_queries_queries_gpt-5-mini.jsonl")
    task_hotpotqa_cs = HotpotQAHardNegativesCodeSwitching(query_file="/root/autodl-tmp/workdir/mteb/Code_Switching_dataset/Retrieval/mteb_hotpotqa_queries_merged.jsonl")
    Retrieval_tasks_unchanged = [task_hotpotqa_cs, task_ArguAna_cs, task_ClimateFEVERHardNegatives_cs, task_cqa_gaming_cs, task_fiqa_cs, task_scidoc_cs, task_trec_covid_cs]
    return Retrieval_tasks_unchanged


def load_retrieval_original_task():
    task_trec_covid = TRECCOVID()
    task_scidoc = SCIDOCS()
    Retrieval_tasks_original = [task_trec_covid, task_scidoc]
    return Retrieval_tasks_original


def load_IR_code_switching_task():
    task_core17_cs = Core17InstructionRetrievalCodeSwitching(
        query_file="/root/autodl-tmp/workdir/mteb/Code_Switching_dataset/IR/core17/jhu-clsp_core17-instructions-mteb_required_format_queries_gpt-5-mini.jsonl",
        instruction_file="/root/autodl-tmp/workdir/mteb/Code_Switching_dataset/IR/core17/jhu-clsp_core17-instructions-mteb_required_format_instructions_gpt-5-mini.jsonl"
    )
    task_news21_cs = News21InstructionRetrievalCodeSwitching(
        query_file="/root/autodl-tmp/workdir/mteb/Code_Switching_dataset/IR/news21/jhu-clsp_news21-instructions-mteb_required_format_queries_gpt-5-mini.jsonl",
        instruction_file="/root/autodl-tmp/workdir/mteb/Code_Switching_dataset/IR/news21/jhu-clsp_news21-instructions-mteb_required_format_instructions_gpt-5-mini.jsonl"
    )
    task_robust04_cs = Robust04InstructionRetrievalCodeSwitching(
        query_file="/root/autodl-tmp/workdir/mteb/Code_Switching_dataset/IR/robust04/jhu-clsp_robust04-instructions-mteb_required_format_queries_gpt-5-mini.jsonl",
        instruction_file="/root/autodl-tmp/workdir/mteb/Code_Switching_dataset/IR/robust04/jhu-clsp_robust04-instructions-mteb_required_format_instructions_gpt-5-mini.jsonl"
    )
    IR_tasks_cs = [task_core17_cs, task_news21_cs, task_robust04_cs]
    return IR_tasks_cs


def load_IR_original_task():
    task_core17 = Core17InstructionRetrieval()
    task_news21 = News21InstructionRetrieval()
    task_robust04 = Robust04InstructionRetrieval()
    IR_task_orginal = [task_core17, task_news21, task_robust04]
    return IR_task_orginal


def load_STS_code_switching_task():
    task_sick_cs = SickrCodeSwitchingSTS(data_file="/root/autodl-tmp/workdir/mteb/Code_Switching_dataset/STS/mteb_sickr-sts_None_test_gpt-5-mini.jsonl")
    task_sts22_cs = STS22CodeSwitching(data_file="/root/autodl-tmp/workdir/mteb/Code_Switching_dataset/STS/mteb_sts22-crosslingual-sts_zh-en_test_gpt-5-mini.jsonl")
    STS_tasks_cs = [task_sick_cs, task_sts22_cs]
    return STS_tasks_cs


def load_Retrieval_And_IR_fixed_task_Chinese():
    task_HumanEval_ch = HumanEvalRetrievalCodeSwitching(
        query_file="/root/autodl-tmp/workdir/mteb/CodeSwitching_Dataset_fixed/RetrievaLChinese_fixed/Chinese_embedding-benchmark_HumanEval_queries_queries_gpt-5-mini.jsonl")
    task_core17_ch = Core17InstructionRetrievalCodeSwitching(
        query_file="/root/autodl-tmp/workdir/mteb/CodeSwitching_Dataset_fixed/IR_Chinese_fixed/core17/fixed_core17_queries_cn.jsonl",
        instruction_file="/root/autodl-tmp/workdir/mteb/CodeSwitching_Dataset_fixed/IR_Chinese_fixed/core17/fixed_core17_instructions_cn.jsonl"
    )
    task_news21_ch = News21InstructionRetrievalCodeSwitching(
        query_file="/root/autodl-tmp/workdir/mteb/CodeSwitching_Dataset_fixed/IR_Chinese_fixed/news21/fixed_news21_queries_cn.jsonl",
        instruction_file="/root/autodl-tmp/workdir/mteb/CodeSwitching_Dataset_fixed/IR_Chinese_fixed/news21/fixed_news21_instructions_cn.jsonl"
    )
    task_robust04_ch = Robust04InstructionRetrievalCodeSwitching(
        query_file="/root/autodl-tmp/workdir/mteb/CodeSwitching_Dataset_fixed/IR_Chinese_fixed/robust04/fixed_robust04_queries_cn.jsonl",
        instruction_file="/root/autodl-tmp/workdir/mteb/CodeSwitching_Dataset_fixed/IR_Chinese_fixed/robust04/fixed_robust04_instructions_cn.jsonl"
    )
    tasks = [task_HumanEval_ch, task_core17_ch, task_news21_ch, task_robust04_ch]
    return tasks


def load_Retrieval_Chinese():
    task_touchev3_ch = Touche2020v3RetrievalCodeSwitching(
        query_file="/root/autodl-tmp/workdir/mteb/CodeSwitching_Dataset_fixed/RetrievaLChinese_fixed/mteb_webis-touche2020-v3_queries_chinese.jsonl")
    task_trec_covid_ch = TRECCOVIDCodeSwitching(
        query_file="/root/autodl-tmp/workdir/mteb/CodeSwitching_Dataset_fixed/RetrievaLChinese_fixed/mteb_trec-covid_queries_queries_gpt-5-mini.jsonl")
    tasks = [task_trec_covid_ch, task_touchev3_ch]
    return tasks


def load_Retrieval_And_IR_fixed_task_Japanese():
    task_HumanEval_jp = HumanEvalRetrievalCodeSwitching(
        query_file="/root/autodl-tmp/workdir/mteb/CodeSwitching_Dataset_fixed/RetrievaLJapanese_fixed/Japanese_embedding-benchmark_HumanEval_queries_queries_gpt-5-mini.jsonl")
    task_core17_jp = Core17InstructionRetrievalCodeSwitching(
        query_file="/root/autodl-tmp/workdir/mteb/CodeSwitching_Dataset_fixed/IR_Japanese_fixed/core17/core17_queries_jp.jsonl",
        instruction_file="/root/autodl-tmp/workdir/mteb/CodeSwitching_Dataset_fixed/IR_Japanese_fixed/core17/core17_instructions_jp.jsonl"
    )
    task_news21_jp = News21InstructionRetrievalCodeSwitching(
        query_file="/root/autodl-tmp/workdir/mteb/CodeSwitching_Dataset_fixed/IR_Japanese_fixed/news21/news21_queries_jp.jsonl",
        instruction_file="/root/autodl-tmp/workdir/mteb/CodeSwitching_Dataset_fixed/IR_Japanese_fixed/news21/news21_instructions_jp.jsonl"
    )
    task_robust04_jp = Robust04InstructionRetrievalCodeSwitching(
        query_file="/root/autodl-tmp/workdir/mteb/CodeSwitching_Dataset_fixed/IR_Japanese_fixed/robust04/robust04_queries_jp.jsonl",
        instruction_file="/root/autodl-tmp/workdir/mteb/CodeSwitching_Dataset_fixed/IR_Japanese_fixed/robust04/robust04_instructions_jp.jsonl"
    )
    tasks = [task_HumanEval_jp, task_core17_jp, task_news21_jp, task_robust04_jp]
    return tasks


def load_Retrieval_Japanese():
    task_trec_covid_jp = TRECCOVIDCodeSwitching(
        query_file="/root/autodl-tmp/workdir/mteb/CodeSwitching_Dataset_fixed/RetrievaLJapanese_fixed/Japanese_mteb_trec-covid_queries_queries_gpt-5-mini.jsonl")
    task_touchev3_jp = Touche2020v3RetrievalCodeSwitching(
        query_file="/root/autodl-tmp/workdir/mteb/CodeSwitching_Dataset_fixed/RetrievaLJapanese_fixed/mteb_webis-touche2020-v3_queries_japanese.jsonl")
    tasks = [task_trec_covid_jp, task_touchev3_jp]
    return tasks


def load_HumanEval_Touche2020():
    task_humaneval = HumanEvalRetrieval()
    task_touche = Touche2020()
    return [task_humaneval, task_touche]


def load_Touche2020RetrievalV3_ch():
    touchev3_ch = Touche2020v3RetrievalCodeSwitching(
        query_file="/root/autodl-tmp/workdir/mteb/CodeSwitching_Dataset_fixed/RetrievaLChinese_fixed/mteb_webis-touche2020-v3_queries_chinese.jsonl")
    touchev3 = Touche2020v3Retrieval()
    return [touchev3, touchev3_ch]


def load_Touche2020RetrievalV3_jp():
    touchev3_jp = Touche2020v3RetrievalCodeSwitching(
        query_file="/root/autodl-tmp/workdir/mteb/CodeSwitching_Dataset_fixed/RetrievaLJapanese_fixed/mteb_webis-touche2020-v3_queries_japanese.jsonl")
    return [touchev3_jp]


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
        output_dir = os.path.join(args.evaluation_output_dir, "20251222", args.output_subfolder_name)
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
            elif task == "Retrieval_cs":
                curr_tasks = load_retrieval_code_switching_task()
            elif task == "Classification_cs":
                curr_tasks = load_classification_code_switching_task()
            elif task == "Reranking_cs":
                curr_tasks = load_reranking_code_switching_task()
            elif task == "STS_cs":
                curr_tasks = load_STS_code_switching_task()
            elif task == "missing_leaderboard":
                curr_tasks = load_HumanEval_Touche2020()
            elif task == "touchev3_ch":
                curr_tasks = load_Touche2020RetrievalV3_ch()
            elif task == "touchev3_jp":
                curr_tasks = load_Touche2020RetrievalV3_jp()
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
