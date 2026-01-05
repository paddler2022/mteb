"""
Query Embedding 3D Visualization Script
对比原版 (HuggingFace) 和 CodeSwitching (本地) 版本的 query embeddings
PCA 降维到 3D 并可视化

直接从 HuggingFace datasets 加载 queries，不加载 corpus
增加 Dispersion 计算和显示功能
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
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


# ========== 工具函数 ==========

def to_python_type(obj):
    """
    递归转换 numpy 类型为 Python 原生类型，用于 JSON 序列化
    """
    if isinstance(obj, dict):
        return {k: to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_python_type(v) for v in obj]
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ========== Dispersion 计算函数 ==========

def compute_dispersion(embeddings, method="avg_cosine"):
    """
    计算 embedding 的 dispersion（分散度）

    Args:
        embeddings: numpy array, shape (n_samples, embedding_dim)
        method: 计算方法
            - "avg_cosine": 1 - 平均余弦相似度（越大越分散）
            - "avg_cosine_raw": 平均余弦相似度（越小越分散）
            - "center_distance": 平均到中心的距离
            - "isotropy": 基于特征值的各向同性度量

    Returns:
        tuple: (dispersion_value, auxiliary_info_dict)
    """
    n_samples = len(embeddings)

    if method == "avg_cosine" or method == "avg_cosine_raw":
        # 计算所有向量对的余弦相似度
        sim_matrix = cosine_similarity(embeddings)
        # 排除对角线（自身相似度=1）
        avg_sim = (sim_matrix.sum() - n_samples) / (n_samples * n_samples - n_samples)

        if method == "avg_cosine":
            # 返回 1 - avg_sim，值越大表示越分散
            dispersion = 1 - avg_sim
        else:
            dispersion = avg_sim

        return float(dispersion), {"avg_cosine_similarity": float(avg_sim)}

    elif method == "center_distance":
        # 计算向量到中心的平均欧氏距离
        center = embeddings.mean(axis=0)
        distances = np.linalg.norm(embeddings - center, axis=1)
        avg_dist = distances.mean()
        std_dist = distances.std()

        return float(avg_dist), {"avg_distance": float(avg_dist), "std_distance": float(std_dist)}

    elif method == "isotropy":
        # 基于 PCA 特征值计算各向同性
        # 参考: https://arxiv.org/abs/1909.00512
        centered = embeddings - embeddings.mean(axis=0)
        cov = np.cov(centered.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]  # 降序

        # 计算有效维度 (participation ratio)
        eigenvalues_normalized = eigenvalues / eigenvalues.sum()
        participation_ratio = 1.0 / (eigenvalues_normalized ** 2).sum()

        # 归一化到 [0, 1]
        isotropy = participation_ratio / len(eigenvalues)

        return float(isotropy), {
            "participation_ratio": float(participation_ratio),
            "top_eigenvalue_ratio": float(eigenvalues[0] / eigenvalues.sum())
        }

    else:
        raise ValueError(f"Unknown method: {method}")


def compute_all_dispersion_metrics(embeddings, name=""):
    """
    计算所有 dispersion 指标

    Args:
        embeddings: numpy array
        name: 数据集名称（用于打印）

    Returns:
        dict: 包含所有指标的字典（所有值都是 Python 原生类型）
    """
    results = {}

    # 1. 平均余弦相似度
    disp_cosine, info_cosine = compute_dispersion(embeddings, method="avg_cosine")
    results["dispersion_cosine"] = float(disp_cosine)
    results["avg_cosine_similarity"] = float(info_cosine["avg_cosine_similarity"])

    # 2. 中心距离
    disp_center, info_center = compute_dispersion(embeddings, method="center_distance")
    results["avg_center_distance"] = float(disp_center)
    results["std_center_distance"] = float(info_center["std_distance"])

    # 3. 各向同性
    disp_isotropy, info_isotropy = compute_dispersion(embeddings, method="isotropy")
    results["isotropy"] = float(disp_isotropy)
    results["participation_ratio"] = float(info_isotropy["participation_ratio"])
    results["top_eigenvalue_ratio"] = float(info_isotropy["top_eigenvalue_ratio"])

    if name:
        print(f"\n  [{name}] Dispersion Metrics:")
        print(f"    Avg Cosine Similarity: {results['avg_cosine_similarity']:.4f}")
        print(f"    Dispersion (1-cos):    {results['dispersion_cosine']:.4f}")
        print(f"    Avg Center Distance:   {results['avg_center_distance']:.4f}")
        print(f"    Isotropy:              {results['isotropy']:.4f}")
        print(f"    Top Eigenvalue Ratio:  {results['top_eigenvalue_ratio']:.4f}")

    return results


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
    """PCA 3D 可视化（含 dispersion 计算和显示）"""

    # ===== 计算 Dispersion =====
    print("\n[Dispersion Analysis]")
    metrics_original = compute_all_dispersion_metrics(emb_original, name="Original")
    metrics_cs = compute_all_dispersion_metrics(emb_cs, name="CodeSwitching")

    disp_orig = metrics_original["dispersion_cosine"]
    disp_cs = metrics_cs["dispersion_cosine"]
    avg_sim_orig = metrics_original["avg_cosine_similarity"]
    avg_sim_cs = metrics_cs["avg_cosine_similarity"]

    # ===== PCA 降维 =====
    n_original = len(emb_original)
    n_cs = len(emb_cs)

    embeddings = np.vstack([emb_original, emb_cs])
    labels = np.array(['Original'] * n_original + ['CodeSwitching'] * n_cs)

    pca = PCA(n_components=3)
    reduced = pca.fit_transform(embeddings)

    # 拆分降维后的结果
    reduced_original = reduced[:n_original]
    reduced_cs = reduced[n_original:]

    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_subplot(111, projection='3d')

    colors = {'Original': 'blue', 'CodeSwitching': 'red'}

    for label in ['Original', 'CodeSwitching']:
        mask = labels == label
        ax.scatter(reduced[mask, 0], reduced[mask, 1], reduced[mask, 2],
                   label=f'{label}', alpha=0.6, c=colors[label], s=50)

    # ===== 标题（含 dispersion）=====
    ax.set_title(f'{task_name}\nModel: {model_name}', fontsize=24)

    # ===== 设置视角和样式 =====
    TICK_SIZE = 24

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

    # ===== Legend（含 dispersion）=====
    legend_labels = [
        f'Original (disp={disp_orig:.3f})',
        f'CodeSwitching (disp={disp_cs:.3f})'
    ]
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=colors['Original'], markersize=10),
               plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=colors['CodeSwitching'], markersize=10)]

    ax.legend(handles, legend_labels, fontsize=18, frameon=False,
              loc='upper right', bbox_to_anchor=(1.05, 1.0), handletextpad=0.1)

    # ===== 添加文本框显示详细指标 =====
    textstr = (f'Dispersion Metrics:\n'
               f'Original:  cos_sim={avg_sim_orig:.3f}, disp={disp_orig:.3f}\n'
               f'CodeSwitch: cos_sim={avg_sim_cs:.3f}, disp={disp_cs:.3f}')

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    fig.text(0.02, 0.02, textstr, fontsize=12, verticalalignment='bottom',
             bbox=props, family='monospace')

    ax.view_init(elev=elev, azim=azim)

    # ===== 保存图片 =====
    output_path = os.path.join(output_dir, f'pca_3d_{task_name}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")

    output_path_pdf = os.path.join(output_dir, f'pca_3d_{task_name}.pdf')
    plt.savefig(output_path_pdf, format='pdf', bbox_inches='tight')
    print(f"Saved: {output_path_pdf}")

    # 保存多个视角
    angles = [(30, 45), (30, 135), (30, 225), (30, 315), (60, 45), (0, 0)]
    for elev_angle, azim_angle in angles:
        ax.view_init(elev=elev_angle, azim=azim_angle)
        angle_path = os.path.join(output_dir, f'pca_3d_{task_name}_elev{elev_angle}_azim{azim_angle}.png')
        plt.savefig(angle_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {angle_path}")
        angle_path_pdf = os.path.join(output_dir, f'pca_3d_{task_name}_elev{elev_angle}_azim{azim_angle}.pdf')
        plt.savefig(angle_path_pdf, format='pdf', bbox_inches='tight')
        print(f"Saved: {angle_path_pdf}")

    if interactive:
        plt.show()
    else:
        plt.close()

    # ===== 保存 embeddings 和降维结果 =====
    print("\n[Saving Embeddings]")

    # 1. 保存原始高维 embedding
    np.save(os.path.join(output_dir, f'{task_name}_original_highdim.npy'), emb_original)
    np.save(os.path.join(output_dir, f'{task_name}_codeswitching_highdim.npy'), emb_cs)
    print(f"  Saved high-dimensional embeddings:")
    print(f"    - {task_name}_original_highdim.npy (shape: {emb_original.shape})")
    print(f"    - {task_name}_codeswitching_highdim.npy (shape: {emb_cs.shape})")

    # 2. 保存合并后的 PCA 降维结果
    np.save(os.path.join(output_dir, f'{task_name}_pca3d_combined.npy'), reduced)
    print(f"  Saved combined PCA result:")
    print(f"    - {task_name}_pca3d_combined.npy (shape: {reduced.shape})")

    # 3. 保存分别降维到 3D 后的 embedding
    np.save(os.path.join(output_dir, f'{task_name}_original_pca3d.npy'), reduced_original)
    np.save(os.path.join(output_dir, f'{task_name}_codeswitching_pca3d.npy'), reduced_cs)
    print(f"  Saved separate 3D PCA embeddings:")
    print(f"    - {task_name}_original_pca3d.npy (shape: {reduced_original.shape})")
    print(f"    - {task_name}_codeswitching_pca3d.npy (shape: {reduced_cs.shape})")

    # ===== 保存 dispersion 结果到 JSON =====
    dispersion_results = {
        "task": task_name,
        "model": model_name,
        "original": {
            "n_samples": int(n_original),
            "embedding_dim": int(emb_original.shape[1]),
            **metrics_original
        },
        "codeswitching": {
            "n_samples": int(n_cs),
            "embedding_dim": int(emb_cs.shape[1]),
            **metrics_cs
        },
        "pca_variance_explained": {
            "PC1": float(pca.explained_variance_ratio_[0]),
            "PC2": float(pca.explained_variance_ratio_[1]),
            "PC3": float(pca.explained_variance_ratio_[2]),
            "total_3PCs": float(sum(pca.explained_variance_ratio_[:3]))
        },
        "saved_files": {
            "original_highdim": f"{task_name}_original_highdim.npy",
            "codeswitching_highdim": f"{task_name}_codeswitching_highdim.npy",
            "pca3d_combined": f"{task_name}_pca3d_combined.npy",
            "original_pca3d": f"{task_name}_original_pca3d.npy",
            "codeswitching_pca3d": f"{task_name}_codeswitching_pca3d.npy"
        }
    }

    # 确保所有值都是 Python 原生类型
    dispersion_results = to_python_type(dispersion_results)

    disp_path = os.path.join(output_dir, f'{task_name}_dispersion.json')
    with open(disp_path, 'w', encoding='utf-8') as f:
        json.dump(dispersion_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {disp_path}")

    # ===== 打印 PCA 方差解释比例 =====
    total_var = sum(pca.explained_variance_ratio_[:3])
    print(f"\nPCA Variance Explained:")
    print(f"  PC1: {pca.explained_variance_ratio_[0]:.2%}")
    print(f"  PC2: {pca.explained_variance_ratio_[1]:.2%}")
    print(f"  PC3: {pca.explained_variance_ratio_[2]:.2%}")
    print(f"  Total (3 PCs): {total_var:.2%}")

    return dispersion_results


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

    # 3D 可视化（含 dispersion 计算）
    print("\n[4] Creating 3D PCA visualization with dispersion analysis...")
    dispersion_results = visualize_pca_3d(
        emb_original, emb_cs, task_name, output_dir, model_name,
        elev=args.elev, azim=args.azim, interactive=args.interactive
    )

    # 打印最终摘要
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Original Dispersion:      {dispersion_results['original']['dispersion_cosine']:.4f}")
    print(f"  CodeSwitching Dispersion: {dispersion_results['codeswitching']['dispersion_cosine']:.4f}")
    print(f"  Difference (CS - Orig):   {dispersion_results['codeswitching']['dispersion_cosine'] - dispersion_results['original']['dispersion_cosine']:.4f}")
    print("=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()