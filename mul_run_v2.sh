python playground_trial.py --model_path  Qwen/Qwen3-Embedding-4B --evaluation_output_dir ./results_fixed_dataset --mode batch --batch_size 8 --tasks Original --output_subfolder_name OG
python playground_trial.py --model_path  Qwen/Qwen3-Embedding-4B --evaluation_output_dir ./results_fixed_dataset  --mode batch --batch_size 8 --tasks Fixed_Japanese --output_subfolder_name Chinese
python playground_trial.py --model_path  Qwen/Qwen3-Embedding-4B --evaluation_output_dir ./results_fixed_dataset --mode batch --batch_size 8 --tasks Fixed_Chinese --output_subfolder_name Japanese
python playground_trial.py --model_path  Qwen/Qwen3-Embedding-4B --evaluation_output_dir ./results_fixed_dataset --mode batch --batch_size 1 --tasks Fixed_R_Chinese --output_subfolder_name Chinese
python playground_trial.py --model_path  Qwen/Qwen3-Embedding-4B --evaluation_output_dir ./results_fixed_dataset  --mode batch --batch_size 1 --tasks Fixed_R_Japanese --output_subfolder_name Japanese
python playground_trial.py --model_path  Qwen/Qwen3-Embedding-4B --evaluation_output_dir ./results_fixed_dataset --mode batch --batch_size 1 --tasks OG_Retrieval --output_subfolder_name OG

python playground_trial.py --model_path  Qwen/Qwen3-Embedding-8B --evaluation_output_dir ./results_fixed_dataset --mode batch --batch_size 8 --tasks Original --output_subfolder_name OG
python playground_trial.py --model_path  Qwen/Qwen3-Embedding-8B --evaluation_output_dir ./results_fixed_dataset  --mode batch --batch_size 8 --tasks Fixed_Japanese --output_subfolder_name Chinese
python playground_trial.py --model_path  Qwen/Qwen3-Embedding-8B --evaluation_output_dir ./results_fixed_dataset --mode batch --batch_size 8 --tasks Fixed_Chinese --output_subfolder_name Japanese
python playground_trial.py --model_path  Qwen/Qwen3-Embedding-8B --evaluation_output_dir ./results_fixed_dataset --mode batch --batch_size 1 --tasks Fixed_R_Chinese --output_subfolder_name Chinese
python playground_trial.py --model_path  Qwen/Qwen3-Embedding-8B --evaluation_output_dir ./results_fixed_dataset  --mode batch --batch_size 1 --tasks Fixed_R_Japanese --output_subfolder_name Japanese
python playground_trial.py --model_path  Qwen/Qwen3-Embedding-8B --evaluation_output_dir ./results_fixed_dataset --mode batch --batch_size 1 --tasks OG_Retrieval --output_subfolder_name OG