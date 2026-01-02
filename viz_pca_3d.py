"""
Query Embedding 3D Visualization Script
对比原版 (HuggingFace) 和 CodeSwitching (本地) 版本的 query embeddings
PCA 降维到 3D 并可视化

直接从 HuggingFace datasets 加载 queries，不加载 corpus
"""

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os
import json
import yaml
from sentence_transformers import SentenceTransformer
import torch
import mteb
from datasets import load_dataset

# 导入 MTEB 相关
from mteb.types import PromptType
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper
from mteb.models.instruct_wrapper import InstructSentenceTransformerModel


# ========== E5 Prompt 配置 ==========
E5_PROMPTS = {
    PromptType.query.value: "query: ",
    PromptType.document.value: "passage: ",
}

# ========== Qwen3 Prompt 配置 ==========
QWEN3_PROMPTS = {
    "Core17InstructionRetrieval": "Retrieval the relevant passage for the given query",
    "News21InstructionRetrieval": "Retrieval the relevant passage for the given query",
    "Robust04InstructionRetrieval": "Retrieval the relevant passage for the given query",
    "Core17InstructionRetrievalCodeSwitching": "Retrieval the relevant passage for the given query",
    "News21InstructionRetrievalCodeSwitching": "Retrieval the relevant passage for the given query",
    "Robust04InstructionRetrievalCodeSwitching": "Retrieval the relevant passage for the given query",
    "Touche2020Retrieval.v3": "Given a question, retrieve detailed and persuasive arguments that answer the question",
    "Touche2020v3RetrievalCodeSwitching": "Given a question, retrieve detailed and persuasive arguments that answer the question",
    "TRECCOVID": "Given a query on COVID-19, retrieve documents that answer the query",
    "TRECCOVIDCodeSwitching": "Given a query on COVID-19, retrieve documents that answer the query",
    "HumanEvalRetrieval": "Given a question about code problem, retrieval code that can solve user's problem",
    "HumanEvalRetrievalCodeSwitching": "Given a question about code problem, retrieval code that can solve user's problem",
}


def qwen3_instruction_template(instruction: str, prompt_type: PromptType | None = None) -> str:
    """Qwen3 Embedding 模型的 instruction 模板"""
    if not instruction or prompt_type == PromptType.document:
        return ""
    if isinstance(instruction, dict):
        if prompt_type is None:
            instruction = list(instruction.values())[0]
        else:
            instruction = instruction[prompt_type]
    return f"Instruct: {instruction}\nQuery:"


# ========== HuggingFace 数据集配置 ==========
HF_DATASETS = {
    "Core17": {
        "dataset": "jhu-clsp/core17-instructions-mteb",
        "query_subset": "queries",
        "instruction_subset": "instruction",
        "query_key": "text",
        "has_instruction": True,
    },
    "News21": {
        "dataset": "jhu-clsp/news21-instructions-mteb",
        "query_subset": "queries",
        "instruction_subset": "instruction",
        "query_key": "text",
        "has_instruction": True,
    },
    "Robust04": {
        "dataset": "jhu-clsp/robust04-instructions-mteb",
        "query_subset": "queries",
        "instruction_subset": "instruction",
        "query_key": "text",
        "has_instruction": True,
    },
    "Touche2020v3": {
        "dataset": "mteb/webis-touche2020-v3",
        "query_subset": "queries",
        "query_key": "text",
        "has_instruction": False,
    },
    "TRECCOVID": {
        "dataset": "mteb/trec-covid",
        "query_subset": "queries",
        "query_key": "text",
        "has_instruction": False,
    },
    "HumanEval": {
        "dataset": "embedding-benchmark/HumanEval",
        "query_subset": "queries",
        "query_key": "text",
        "has_instruction": False,
    },
}


def load_config(config_path):
    """加载 YAML 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# ========== 模型加载函数 ==========

def load_model(model_path):
    """加载 HuggingFace 模型（使用 MTEB 内置配置）"""
    if "e5" in model_path.lower():
        model = mteb.get_model(model_path)
    else:
        model = mteb.get_model(
            model_path,
            model_kwargs={
                "attn_implementation": "flash_attention_2",
                "torch_dtype": torch.bfloat16
            }
        )
    return model, "hf"


def load_model_ST(model_path):
    """
    加载本地模型并支持 prompt
    """
    model_path_lower = model_path.lower()

    if "e5" in model_path_lower:
        print(f"[INFO] 检测到 E5 模型，使用 query:/passage: prompt")
        model = SentenceTransformerEncoderWrapper(
            model=model_path,
            revision=None,
            model_prompts=E5_PROMPTS,
        )
        return model, "e5"

    elif "qwen" in model_path_lower and "embedding" in model_path_lower:
        print(f"[INFO] 检测到 Qwen Embedding 模型，使用 Instruct 模板")
        model = InstructSentenceTransformerModel(
            model_name=model_path,
            revision=None,
            instruction_template=qwen3_instruction_template,
            apply_instruction_to_passages=False,
            prompts_dict=QWEN3_PROMPTS,
            model_kwargs={
                "attn_implementation": "flash_attention_2",
                "torch_dtype": torch.bfloat16,
                "device_map": "cuda"
            },
        )
        if hasattr(model, 'model') and hasattr(model.model, 'tokenizer'):
            model.model.tokenizer.padding_side = 'left'
        return model, "qwen3"

    else:
        print(f"[INFO] 未检测到特殊 prompt 需求，使用默认配置")
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
        return model, "default"


# ========== 数据加载函数 ==========

def load_queries_from_hf(task_name):
    """
    从 HuggingFace 直接加载 queries (不加载 corpus)
    对于 InstructionRetrieval 任务，同时加载 instruction 并组合
    组合方式: query + " " + instruction (与 mteb 框架一致)
    """
    if task_name not in HF_DATASETS:
        raise ValueError(f"Unknown task: {task_name}")

    config = HF_DATASETS[task_name]
    dataset_name = config["dataset"]
    query_subset = config["query_subset"]
    query_key = config["query_key"]
    has_instruction = config.get("has_instruction", False)

    print(f"  Loading queries from HuggingFace: {dataset_name} ({query_subset})")
    ds_queries = load_dataset(dataset_name, query_subset, trust_remote_code=True)

    # 获取 queries (按 id 索引)
    if "queries" in ds_queries:
        split_data = ds_queries["queries"]
    elif "test" in ds_queries:
        split_data = ds_queries["test"]
    else:
        first_split = list(ds_queries.keys())[0]
        split_data = ds_queries[first_split]

    queries_dict = {}
    for item in split_data:
        qid = str(item.get("_id", item.get("id", "")))
        queries_dict[qid] = item[query_key]

    # 如果有 instruction，加载并组合
    if has_instruction:
        instruction_subset = config.get("instruction_subset", "instruction")
        print(f"  Loading instructions from HuggingFace: {dataset_name} ({instruction_subset})")
        ds_inst = load_dataset(dataset_name, instruction_subset, trust_remote_code=True)

        # 获取 instruction (按 query-id 索引)
        if "instruction" in ds_inst:
            inst_split = ds_inst["instruction"]
        elif "test" in ds_inst:
            inst_split = ds_inst["test"]
        else:
            first_split = list(ds_inst.keys())[0]
            inst_split = ds_inst[first_split]

        inst_dict = {}
        for item in inst_split:
            qid = str(item.get("query-id", item.get("_id", "")))
            inst_dict[qid] = item.get("instruction", "")

        # 组合: query + " " + instruction (与 mteb 框架一致)
        combined = []
        for qid, query in queries_dict.items():
            instruction = inst_dict.get(qid, "")
            if instruction:
                combined.append(query + " " + instruction)
            else:
                combined.append(query)
        print(f"  Combined {len(combined)} queries with instructions")
        return combined
    else:
        return list(queries_dict.values())


def load_queries_from_jsonl(query_file, instruction_file=None):
    """
    从本地 JSONL 文件加载 queries
    如果提供 instruction_file，则组合 query + " " + instruction
    """
    print(f"  Loading queries from local file: {query_file}")

    # 加载 queries
    queries_dict = {}
    with open(query_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                qid = str(data.get("_id", data.get("id", "")))
                text = data.get("text", data.get("query", ""))
                queries_dict[qid] = text

    # 如果有 instruction 文件，加载并组合
    if instruction_file:
        print(f"  Loading instructions from local file: {instruction_file}")
        inst_dict = {}
        with open(instruction_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    qid = str(data.get("query-id", data.get("_id", "")))
                    instruction = data.get("instruction", "")
                    inst_dict[qid] = instruction

        # 组合: query + " " + instruction (与 mteb 框架一致)
        combined = []
        for qid, query in queries_dict.items():
            instruction = inst_dict.get(qid, "")
            if instruction:
                combined.append(query + " " + instruction)
            else:
                combined.append(query)
        print(f"  Combined {len(combined)} queries with instructions")
        return combined
    else:
        return list(queries_dict.values())


# ========== 编码函数 ==========

def encode_queries(model, model_type, queries, task_name, batch_size=32):
    """编码 queries"""
    if model_type == "hf":
        if hasattr(model, 'model'):
            embeddings = model.model.encode(queries, batch_size=batch_size,
                                            show_progress_bar=True, convert_to_numpy=True)
        else:
            embeddings = model.encode(queries, batch_size=batch_size,
                                      show_progress_bar=True, convert_to_numpy=True)

    elif model_type == "e5":
        prompt = E5_PROMPTS.get(PromptType.query.value, "")
        queries_formatted = [prompt + q for q in queries]
        if hasattr(model, 'model'):
            embeddings = model.model.encode(queries_formatted, batch_size=batch_size,
                                            show_progress_bar=True, convert_to_numpy=True)
        else:
            embeddings = model.encode(queries_formatted, batch_size=batch_size,
                                      show_progress_bar=True, convert_to_numpy=True)

    elif model_type == "qwen3":
        instruction = QWEN3_PROMPTS.get(task_name, "")
        if instruction:
            queries_formatted = [f"Instruct: {instruction}\nQuery:{q}" for q in queries]
        else:
            queries_formatted = queries
        if hasattr(model, 'model'):
            embeddings = model.model.encode(queries_formatted, batch_size=batch_size,
                                            show_progress_bar=True, convert_to_numpy=True)
        else:
            embeddings = model.encode(queries_formatted, batch_size=batch_size,
                                      show_progress_bar=True, convert_to_numpy=True)

    else:
        if hasattr(model, 'model'):
            embeddings = model.model.encode(queries, batch_size=batch_size,
                                            show_progress_bar=True, convert_to_numpy=True)
        else:
            embeddings = model.encode(queries, batch_size=batch_size,
                                      show_progress_bar=True, convert_to_numpy=True)

    return embeddings


# ========== 3D 可视化函数 ==========



def visualize_pca_3d(emb_original, emb_cs, task_name, output_dir, model_name,
                     elev=30, azim=45, interactive=False):
    """PCA 3D 可视化"""
    embeddings = np.vstack([emb_original, emb_cs])
    labels = np.array(['Original'] * len(emb_original) + ['CodeSwitching'] * len(emb_cs))

    pca = PCA(n_components=3)
    reduced = pca.fit_transform(embeddings)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    colors = {'Original': 'blue', 'CodeSwitching': 'red'}

    for label in ['Original', 'CodeSwitching']:
        mask = labels == label
        ax.scatter(reduced[mask, 0], reduced[mask, 1], reduced[mask, 2],
                   label=f'{label}', alpha=0.6, c=colors[label], s=50)

    # ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    # ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    # ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
    # ax.legend(fontsize=15, loc='upper right')
    ax.set_title(f'{task_name}\nModel:{model_name}', fontsize=32)
    # ax.tick_params(axis='both', which='major', labelsize=32)



    #########################################################
    # 设置视角
    TICK_SIZE = 32

    ax.tick_params(axis='x', labelsize=TICK_SIZE)
    ax.tick_params(axis='y', labelsize=TICK_SIZE)
    ax.tick_params(axis='z', labelsize=TICK_SIZE, pad=12)

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')

    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.12)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    ax.legend(fontsize=28, frameon=False, loc='upper right', bbox_to_anchor=(1.07, 1.03), handletextpad=0.1)

    ax.view_init(elev=elev, azim=azim)

    output_path = os.path.join(output_dir, f'pca_3d_{task_name}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    output_path = os.path.join(output_dir, f'pca_3d_{task_name}.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"Saved: {output_path}")

    # 保存多个视角
    angles = [(30, 45), (30, 135), (30, 225), (30, 315), (60, 45), (0, 0)]
    for elev, azim in angles:
        ax.view_init(elev=elev, azim=azim)
        angle_path = os.path.join(output_dir, f'pca_3d_{task_name}_elev{elev}_azim{azim}.png')
        plt.savefig(angle_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {angle_path}")
        angle_path = os.path.join(output_dir, f'pca_3d_{task_name}_elev{elev}_azim{azim}.pdf')
        plt.savefig(angle_path, format='pdf', bbox_inches='tight')
        print(f"Saved: {angle_path}")

    if interactive:
        plt.show()
    else:
        plt.close()

    # 保存 embeddings 和降维结果 (3D 版本)
    np.save(os.path.join(output_dir, f'{task_name}_3d_original.npy'), emb_original)
    np.save(os.path.join(output_dir, f'{task_name}_3d_codeswitching.npy'), emb_cs)
    np.save(os.path.join(output_dir, f'{task_name}_3d_pca.npy'), reduced)
    print(f"Saved: {task_name}_3d_original.npy, {task_name}_3d_codeswitching.npy, {task_name}_3d_pca.npy")

    # 打印方差解释比例
    total_var = sum(pca.explained_variance_ratio_[:3])
    print(f"\nPCA Variance Explained:")
    print(f"  PC1: {pca.explained_variance_ratio_[0]:.2%}")
    print(f"  PC2: {pca.explained_variance_ratio_[1]:.2%}")
    print(f"  PC3: {pca.explained_variance_ratio_[2]:.2%}")
    print(f"  Total (3 PCs): {total_var:.2%}")


# ========== 主函数 ==========

def main():
    parser = argparse.ArgumentParser(description="Query Embedding 3D Visualization (Single Task)")
    parser.add_argument("--config", required=True, help="Path to task config YAML file")
    parser.add_argument("--interactive", action="store_true", help="Show interactive 3D plot")
    parser.add_argument("--elev", type=float, default=30, help="Elevation angle for 3D view")
    parser.add_argument("--azim", type=float, default=45, help="Azimuth angle for 3D view")
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    task_name = config["task_name"]
    model_path = config["model_path"]
    output_dir = config.get("output_dir", "./visualization_output")
    batch_size = config.get("batch_size", 32)
    load_local = config.get("load_local", False)
    cs_query_file = config["cs_query_file"]
    cs_instruction_file = config.get("cs_instruction_file", None)

    os.makedirs(output_dir, exist_ok=True)
    model_name = os.path.basename(model_path)

    print("=" * 60)
    print(f"Task: {task_name}")
    print(f"Model: {model_path}")
    print(f"Load Local Model: {load_local}")
    print(f"Output: {output_dir}")
    if cs_instruction_file:
        print(f"Has Instruction: Yes")
    print("=" * 60)

    # 检查任务是否支持
    if task_name not in HF_DATASETS:
        print(f"Error: Unknown task '{task_name}'")
        print(f"Supported tasks: {list(HF_DATASETS.keys())}")
        return

    # 加载模型
    print("\n[1] Loading model...")
    if load_local:
        model, model_type = load_model_ST(model_path)
    else:
        model, model_type = load_model(model_path)
    print(f"Model type: {model_type}")

    # 加载 queries
    print("\n[2] Loading queries...")

    print("Loading original queries from HuggingFace...")
    queries_original = load_queries_from_hf(task_name)
    print(f"Original queries: {len(queries_original)}")

    print("Loading CodeSwitching queries from local file...")
    queries_cs = load_queries_from_jsonl(cs_query_file, cs_instruction_file)
    print(f"CodeSwitching queries: {len(queries_cs)}")

    if not queries_original or not queries_cs:
        print("Error: No queries found")
        return

    # 编码
    print("\n[3] Encoding queries...")
    print("Encoding original queries...")
    emb_original = encode_queries(model, model_type, queries_original, task_name, batch_size)

    print("Encoding CodeSwitching queries...")
    emb_cs = encode_queries(model, model_type, queries_cs, task_name + "CodeSwitching", batch_size)

    # 3D 可视化
    print("\n[4] Creating 3D PCA visualization...")
    visualize_pca_3d(emb_original, emb_cs, task_name, output_dir, model_name,
                     elev=args.elev, azim=args.azim, interactive=args.interactive)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
