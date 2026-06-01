# [KDD2026(Feb. Cycle)]From GPS Points to Travel Patterns: Flexible and Semantic Trajectory Generation with LLMs

Arxiv[https://arxiv.org/abs/2605.30014].The implementation of HTP. The dataset is coming soon.

## File Structure
```
HTP
├── config                         # configuration for training
│   ├── chengdu
│   │   ├── stage1_config.yaml
│   │   └── stage2_config.yaml
│   ├── ds_config.json             # deepspeed configuration
│   └── porto
│       ├── stage1_config.yaml
│       └── stage2_config.yaml
├── dataloader_stage1.py           # dataloader for RQVAE
├── dataloader_stage2.py           # convert traj. into travel patterns
├── model                          # RQVAE model
│   ├── rq_quant.py
│   ├── rqvae.py
│   └── unet.py
├── preprocess_stage1.py           # preprocess for RQVAE
├── preprocess_stage2.py           # preprocess for LLM
├── preprocess_stage3_data4llm.py  # preprocess for LLM
├── readme.md
├── stage1_main.py                 # training for RQVAE
├── stage1_trainer.py              # trainer for RQVAE
├── stage2_main.py                 # training for LLM
├── stage2_merge_model.py          # merge LLM with Lora
├── stage3_pattern_generation.py   # generate travel patterns by LLM
├── stage3_traj_generation.py      # generate GPS traj.
└── stage4_evaluation.py           # evaluate performance
```

## Run Code
```
# preprocess for RQVAE
python preprocess_stage1.py --city {YOUR_CITY_NAME} --data_dir {YOUR_DATA_DIR} --seed 42 --node2vec_dim 128 --node2vec_epochs 20

# training for RQVAE
accelerate launch --multi_gpu --num_processes 4 --num_machines 1 --mixed_precision bf16 stage1_main.py --seed 42 --exp_name {YOUR_EXP_NAME} --city {YOUR_CITY_NAME}

# preprocess for LLM
python preprocess_stage2.py --exp_name {YOUR_EXP_NAME} --city {YOUR_CITY_NAME} --seed 42 --device {YOUR_GPU_DEVICE}
python preprocess_stage3_data4llm.py --exp_name {YOUR_EXP_NAME} --city {YOUR_CITY_NAME} --seed 42

# training for LLM
accelerate launch --num_processes 4 --num_machines 1 --mixed_precision bf16 stage2_main.py --exp_name {YOUR_EXP_NAME} --city {YOUR_CITY_NAME} --model_name {YOUR_LLM_MODEL_DIR}

# Merge LLM with Lora
python stage2_merge_model.py --exp_name {YOUR_EXP_NAME} --city {YOUR_CITY_NAME} --model_name {YOUR_LLM_MODEL_DIR}

# generation travel patterns by LLM
accelerate launch --multi_gpu --num_machines 1 --num_processes 4  --mixed_precision bf16 stage3_pattern_generation.py --seed 42--city {YOUR_CITY_NAME} --exp_name {YOUR_EXP_NAME} --ddp True

# generation GPS traj.
python stage3_traj_generation.py --seed 42 --exp_name {YOUR_EXP_NAME} --city {YOUR_CITY_NAME} --device {YOUR_GPU_DEVICE}

# evaulation performance
python stage4_evaluation.py --exp_name {YOUR_EXP_NAME} --city {YOUR_CITY_NAME}
```
