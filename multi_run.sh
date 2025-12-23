python playground_trial.py --model_path  /root/autodl-tmp/workdir/Model_Embedding_Align/aligned_greenplm_e5_large_v2 --evaluation_output_dir ./results_fixed_dataset --mode batch --batch_size 32 --tasks Fixed_Chinese --output_subfolder_name Chinese
python playground_trial.py --model_path  /root/autodl-tmp/workdir/Model_Embedding_Align/aligned_greenplm_e5_large_v2 --evaluation_output_dir ./results_fixed_dataset --mode batch --batch_size 32 --tasks Original --output_subfolder_name OG

python playground_trial.py --model_path  /root/autodl-tmp/workdir/Model_Embedding_Align/aligned_greenplm_e5_large_v2 --evaluation_output_dir ./results_fixed_dataset --mode batch --batch_size 4 --tasks Fixed_R_Chinese --output_subfolder_name Chinese
python playground_trial.py --model_path  /root/autodl-tmp/workdir/Model_Embedding_Align/aligned_greenplm_e5_large_v2 --evaluation_output_dir ./results_fixed_dataset --mode batch --batch_size 4 --tasks OG_Retrieval --output_subfolder_name OG
