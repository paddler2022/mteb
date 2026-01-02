python playground.py --model_path  Qwen/Qwen3-Embedding-4B --evaluation_output_dir ./results_fixed_dataset  --mode batch --batch_size 4 --tasks touchev3_jp --output_subfolder_name Japanese
python playground.py --model_path  Qwen/Qwen3-Embedding-4B --evaluation_output_dir ./results_fixed_dataset --mode batch --batch_size 4 --tasks touchev3 --output_subfolder_name OG

python playground.py --model_path  Salesforce/SFR-Embedding-2_R --evaluation_output_dir ./results_fixed_dataset --mode batch --batch_size 32 --tasks treccovid --output_subfolder_name OG
python playground.py --model_path  Salesforce/SFR-Embedding-2_R --evaluation_output_dir ./results_fixed_dataset  --mode batch --batch_size 32 --tasks treccovid_jp --output_subfolder_name Japanese
python playground.py --model_path  Salesforce/SFR-Embedding-2_R --evaluation_output_dir ./results_fixed_dataset --mode batch --batch_size 32 --tasks treccovid_ch --output_subfolder_name Chinese

python playgroundpy --model_path  nvidia/llama-embed-nemotron-8b --evaluation_output_dir ./results_fixed_dataset  --mode batch --batch_size 32 --tasks treccovid_jp --output_subfolder_name Japanese
python playground.py --model_path  nvidia/llama-embed-nemotron-8b --evaluation_output_dir ./results_fixed_dataset --mode batch --batch_size 32 --tasks treccovid_ch --output_subfolder_name Chinese

python playground.py --model_path  Salesforce/SFR-Embedding-2_R --evaluation_output_dir ./results_fixed_dataset --mode batch --batch_size 4 --tasks touchev3 --output_subfolder_name OG
python playground.py --model_path  Salesforce/SFR-Embedding-2_R --evaluation_output_dir ./results_fixed_dataset  --mode batch --batch_size 4 --tasks touchev3_jp --output_subfolder_name Japanese
python playground.py --model_path  Salesforce/SFR-Embedding-2_R --evaluation_output_dir ./results_fixed_dataset --mode batch --batch_size 4 --tasks touchev3_ch --output_subfolder_name Chinese

python playground.py --model_path  nvidia/llama-embed-nemotron-8b --evaluation_output_dir ./results_fixed_dataset --mode batch --batch_size 4 --tasks touchev3 --output_subfolder_name OG
python playground.py --model_path  nvidia/llama-embed-nemotron-8b --evaluation_output_dir ./results_fixed_dataset  --mode batch --batch_size 4 --tasks touchev3_jp --output_subfolder_name Japanese
python playground.py --model_path  nvidia/llama-embed-nemotron-8b --evaluation_output_dir ./results_fixed_dataset --mode batch --batch_size 4 --tasks touchev3_ch --output_subfolder_name Chinese
