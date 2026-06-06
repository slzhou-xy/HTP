# [KDD2026(Feb. Cycle)] From GPS Points to Travel Patterns: Flexible and Semantic Trajectory Generation with LLMs

This repository provides the official implementation of **HTP**, introduced in our paper [*From GPS Points to Travel Patterns: Flexible and Semantic Trajectory Generation with LLMs*](https://arxiv.org/abs/2605.30014).

HTP uses **Qwen3-1.7B-Thinking** as its base language model. Please download the model from Hugging Face before running the pipeline.

## Data
We use the Porto and Chengdu datasets. You can download them from [Google Drive](https://drive.google.com/file/d/1X7veClkdG8Z1pkTNnR0j9cNBAbrLTseD/view?usp=sharing).
After downloading the data, place each city directory under `traj_data`. Each city directory contains the following files:

- `rn/`: Road network data.
- `road_emb_128d.pt`: A 128-dimensional road embedding generated using Node2Vec.
- `train_index.npy` and `test_index.npy`: Indices defining the training and test splits.
- `traj_rel_info.parquet`: Relative trajectory information, including the relative position `percent` of each GPS point along its matched road segment and the corresponding `dx` and `dy` offsets.
- `zcore.json`: Statistics used to standardize the `dx` and `dy` offsets.
- `traj.parquet`: The original trajectory data. In the Chengdu dataset, `flag` indicates whether the taxi is occupied by a passenger. In the Porto dataset, `call_type` indicates the trip's call type. In both datasets, `cpath` is the continuous road-segment sequence obtained after map matching, while `opath` is the sequence of road segments individually aligned with the original GPS points.


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

Run all commands from the `HTP` directory and replace the values in braces with your own settings.

### 1. Preprocess Data

Preprocess trajectories and generate Node2Vec road embeddings.

```bash
python preprocess_stage1.py --city {CITY} --data_dir {DATA_DIR} --seed 42 --node2vec_dim 128 --node2vec_epochs 20
```

### 2. Train RQ-VAE

Train the RQ-VAE to learn discrete travel-pattern representations.

```bash
accelerate launch --multi_gpu --num_processes 4 --num_machines 1 --mixed_precision bf16 stage1_main.py --city {CITY} --exp_name {EXP_NAME} --seed 42
```

### 3. Prepare LLM Data

Encode trajectories into discrete travel-pattern tokens.

```bash
python preprocess_stage2.py --city {CITY} --exp_name {EXP_NAME} --seed 42 --device {DEVICE}
```

Convert the tokens and trajectory information into LLM training data.

```bash
python preprocess_stage3_data4llm.py --city {CITY} --exp_name {EXP_NAME} --seed 42
```

### 4. Train and Merge the LLM

Fine-tune the base LLM using LoRA.

```bash
accelerate launch --num_processes 4 --num_machines 1 --mixed_precision bf16 stage2_main.py --city {CITY} --exp_name {EXP_NAME} --model_name {LLM_MODEL_DIR} --seed 42
```

Merge the trained LoRA adapter into the base LLM.

```bash
python stage2_merge_model.py --city {CITY} --exp_name {EXP_NAME} --model_name {LLM_MODEL_DIR} --seed 42
```

### 5. Generate Trajectories

Generate travel patterns and road sequences using the fine-tuned LLM.

```bash
accelerate launch --multi_gpu --num_processes 4 --num_machines 1 --mixed_precision bf16 stage3_pattern_generation.py --city {CITY} --exp_name {EXP_NAME} --seed 42 --ddp true
```

Decode the generated travel patterns into GPS trajectories.

```bash
python stage3_traj_generation.py --city {CITY} --exp_name {EXP_NAME} --seed 42 --device {DEVICE}
```

### 6. Evaluate

Evaluate the generated trajectories.

```bash
python stage4_evaluation.py --city {CITY} --exp_name {EXP_NAME} --seed 42
```
