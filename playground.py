import mteb
import os
import json
from mteb import MTEB
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

def load_model(model_path):
    if "e5" in model_path:
        model = mteb.get_model(model_path)
    # elif "gemma-300m" in model_path:
    #     model = SentenceTransformer(model_path, device="cuda")
    else:
        model = mteb.get_model(
            model_path,
            model_kwargs={
                "attn_implementation": "flash_attention_2",
                "torch_dtype": torch.bfloat16
            }

        )
    return model

def load_model_ST(model_path):
    if "e5" in model_path:
        model = SentenceTransformer(model_path, device="cuda")
    else:
        model = SentenceTransformer(
            model_path,
            model_kwargs={
                  "attn_implementation": "flash_attention_2",
                  "torch_dtype": torch.bfloat16,
                "device_map": "cuda"
            },
            device="cuda"
        )
        model.tokenizer.padding_side = 'left'
    return model

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

# code-switching queries file path input
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
    # IR_tasks_cs = [task_robust04_cs]
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


    task_HumanEval_jp = HumanEvalRetrievalCodeSwitching(query_file="/root/autodl-tmp/workdir/mteb/CodeSwitching_Dataset_fixed/RetrievaLJapanese_fixed/Japanese_embedding-benchmark_HumanEval_queries_queries_gpt-5-mini.jsonl")
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
    touchev3_ch = Touche2020v3RetrievalCodeSwitching(query_file="/root/autodl-tmp/workdir/mteb/CodeSwitching_Dataset_fixed/RetrievaLChinese_fixed/mteb_webis-touche2020-v3_queries_chinese.jsonl")
    touchev3 = Touche2020v3Retrieval()
    return [touchev3, touchev3_ch]

def load_Touche2020RetrievalV3_jp():
    touchev3_jp = Touche2020v3RetrievalCodeSwitching(query_file="/root/autodl-tmp/workdir/mteb/CodeSwitching_Dataset_fixed/RetrievaLJapanese_fixed/mteb_webis-touche2020-v3_queries_japanese.jsonl")
    return [touchev3_jp]

def load_original_tasks():
    task_core17 = Core17InstructionRetrieval()
    task_news21 = News21InstructionRetrieval()
    task_robust04 = Robust04InstructionRetrieval()
    task_touche2020 = Touche2020v3Retrieval()
    task_humaneval = HumanEvalRetrieval()
    task_treccovid = TRECCOVID()
    tasks = [task_humaneval, task_core17, task_news21, task_robust04]
    #tasks = [task_core17]
    return tasks

def load_og_Retrieval_tasks():
    task_touche2020 = Touche2020v3Retrieval()
    task_treccovid = TRECCOVID()
    return [task_treccovid, task_touche2020]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="MTEB configuration selection"
    )
    parser.add_argument("--mode", choices=["batch", "single"], default="batch")
    parser.add_argument("--tasks", nargs='+', help="list of the tasks in types you need to run", type=str, required=True)
    parser.add_argument("--model_path", required=True, help="Either hf path or local path", type=str)
    parser.add_argument("--batch_size", required=True, help="batch_size", type=int, default=4)
    parser.add_argument("--output_subfolder_name", help="number of epochs", type=str, default=None)
    parser.add_argument(
        "--evaluation_output_dir",
        required=True,
        help="Evaluation output directory",
        type=str,
    )

    args = parser.parse_args()

    current_time = datetime.now()
    date = current_time.strftime("%Y%m%d")
    # precise_time = current_time.strftime("%H%M%S")
    if args.output_subfolder_name:
        output_dir = os.path.join(args.evaluation_output_dir, date, args.output_subfolder_name)
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
    if "aligned" in args.model_path:
        model = load_model_ST(args.model_path)
    else:
        model = load_model(args.model_path)
    evaluation = MTEB(tasks=All_tasks_names)

    model_name = os.path.basename(args.model_path)
    results = evaluation.run(
        model,
        encode_kwargs={"batch_size":args.batch_size, "show_progress_bar": True},
        output_folder=output_dir,
        model_name=model_name,
    )
    print(results)