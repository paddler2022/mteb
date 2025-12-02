import mteb
import json
from mteb import MTEB
from sentence_transformers import SentenceTransformer
from mteb.tasks.retrieval.eng import TRECCOVIDCodeSwitching, AILAStatutesCodeSwitching, HagridRetrievalCodeSwitching, SCIDOCSCodeSwitchingRetrieval
import torch

model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-0.6B",
    model_kwargs={
          "attn_implementation": "flash_attention_2",
          "torch_dtype": torch.float16,
        "device_map": "cuda"
    },
    device="cuda"
)
model.tokenizer.padding_side = 'left'
# code-switching queries file path input
#task = TRECCOVIDCodeSwitching(query_file="/root/autodl-tmp/workdir/mteb/Code_Switching_data/mteb_trec-covid_queries_queries_eval_gpt-5.jsonl")
#task = AILAStatutesCodeSwitching(query_file="/root/autodl-tmp/workdir/mteb/Code_Switching_data/mteb_AILA_statutes_queries_queries_eval_gpt-5.jsonl")
#task = HagridRetrievalCodeSwitching(query_file="/root/autodl-tmp/workdir/mteb/Code_Switching_data/mteb_HagridRetrieval_queries_dev_eval_gpt-5.jsonl")
task = SCIDOCSCodeSwitchingRetrieval(query_file="/root/autodl-tmp/workdir/mteb/Code_Switching_data/mteb_scidocs_queries_queries_eval_gpt-5.jsonl")

evaluation = MTEB(tasks=[task])
results = evaluation.run(
    model,
    encode_kwargs={"batch_size": 4, "show_progress_bar": True},
    output_folder="./results"  # 在这里指定输出目录
)
print(results)
results_to_save = [r.to_dict() if hasattr(r, 'to_dict') else r for r in results]
with open('results.json', 'w', encoding='utf-8') as f:
    json.dump(results_to_save, f, ensure_ascii=False, indent=2)

print("Results saved to results.json")