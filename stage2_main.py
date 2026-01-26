from dataclasses import dataclass, field, asdict
from peft import (
    get_peft_model,
    TaskType,
    LoraConfig
)
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    set_seed,
)
import torch
import os
from pathlib import Path
import yaml
import warnings
import pandas as pd

warnings.filterwarnings("ignore")


@dataclass
class ParserArguments:
    seed: int = field(default=42)
    city: str = field(default='chengdu')
    exp_name: str = field(
        default='global_percent_ep100_bs512_code_256_8421_64d_en_64d_1224',
        metadata={"help": "Experiment name"}
    )
    model_name: str = field(
        default='/home/zhousilin/.cache/huggingface/hub/Qwen3-1.7B',
        metadata={"help": "Model name"}
    )
    lora_r: int = field(default=16, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "LoRA target modules"}
    )


def get_special_tokens(config):
    special_tokens = []
    special_tokens.append('<|t_begin|>')
    special_tokens.append('<|t_end|>')
    special_tokens.append('<|p_begin|>')
    special_tokens.append('<|p_end|>')

    vocab_multi = config['RQ_quant']['vocab_multi']
    n_codebooks = config['RQ_quant']['n_codebooks']

    ch_multi = config['Unet']['ch_multi']
    for i in range(2 ** (len(ch_multi) - 1)):
        special_tokens.append(f'<t_{i}>')

    # max_range = 256
    # max_range = [32, 64, 128, 256]
    # max_range = [128, 128, 128, 128]
    base_vocab_size = config['RQ_quant']['base_vocab_size']
    prefix_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g'][:n_codebooks]

    for k, prefix in enumerate(prefix_list):
        for i in range(base_vocab_size // vocab_multi[k]):
            special_tokens.append(f'<{prefix}_{i}>')
    num_roads = config['num_roads']
    for i in range(num_roads):
        special_tokens.append(f'<road_{i}>')

    return special_tokens


def prepare_chat_dataset(data_path, local_rank=0):
    if local_rank == 0:
        print(f"Loading parquet file: {data_path}")
    data_pq = pd.read_parquet(data_path)
    if local_rank == 0:
        print(f"Data shape: {data_pq.shape}")
        print(f"Columns: {list(data_pq.columns)}")

    texts = []
    system_message = "You are a professional spatial-temporal expert who needs to generate the possible sampling type and travel pattern for users based on their trip information."
    for _, row in data_pq.iterrows():
        assistant_content = f"<think>\n\n</think>\n\n{row['answer']}"
        formatted_text = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{row['desc']}<|im_end|>
<|im_start|>assistant
{assistant_content}<|im_end|>"""

        texts.append(formatted_text)

    if local_rank == 0:
        print(f"Total texts: {len(texts)}")

        print("\\nFirst 3 text examples:")
        for i, text in enumerate(texts[:3]):
            print(f"  [{i}] Length: {len(text)} chars")
            print(f"  [{i}] Text: {text}")
            print()

    dataset_dict = {
        'text': texts
    }
    return Dataset.from_dict(dataset_dict)


class CustomDataCollator:
    def __init__(self, tokenizer, mlm=False):
        self.tokenizer = tokenizer
        self.mlm = mlm

    def __call__(self, features):
        input_ids = [feature["input_ids"] for feature in features]
        attention_mask = [feature["attention_mask"] for feature in features]

        max_length = max(len(ids) for ids in input_ids)

        padded_input_ids = []
        padded_attention_mask = []
        labels = []

        for i, (ids, mask) in enumerate(zip(input_ids, attention_mask)):
            padding_length = max_length - len(ids)
            padded_ids = ids + [self.tokenizer.pad_token_id] * padding_length
            padded_mask = mask + [0] * padding_length

            label = padded_ids.copy()

            assistant_end_pos = len(ids)
            assistant_start_tokens = self.tokenizer.encode("<think>\n\n</think>\n\n", add_special_tokens=False)

            for j in range(len(ids) - len(assistant_start_tokens) + 1):
                if ids[j:j+len(assistant_start_tokens)] == assistant_start_tokens:
                    for k in range(j + len(assistant_start_tokens)):
                        label[k] = -100
                    break

            for k in range(assistant_end_pos, len(label)):
                label[k] = -100
            padded_input_ids.append(padded_ids)
            padded_attention_mask.append(padded_mask)
            labels.append(label)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def tokenize_function(examples, tokenizer):
    tokenized = tokenizer(
        examples['text'],
        # padding='longest',
        padding=False,
        truncation=False,
        add_special_tokens=True,
        return_attention_mask=True,
    )
    return tokenized


def main(args):
    city = args.city
    config = yaml.load(open(f'config/{city}/stage2_config.yaml', 'r'), Loader=yaml.FullLoader)
    training_args = config['training_args']
    training_args['output_dir'] = Path('logs', city, args.exp_name, training_args['output_dir'])
    training_args['label_names'] = ["labels"]
    training_args = TrainingArguments(**training_args)
    model_name = args.model_name
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    stage1_config_path = Path('logs', city, args.exp_name, 'stage1_config.yaml')
    stage1_config = yaml.load(open(stage1_config_path, 'r'), Loader=yaml.FullLoader)

    if local_rank == 0:
        args_dict = asdict(args)
        config['args_settings'] = args_dict
        with open(os.path.join('logs', config['city'], args.exp_name, 'stage2_config.yaml'), 'w') as f:
            yaml.dump(config, f)

    train_data_path = Path('logs', config['city'], args.exp_name, 'data', 'llm_train_traj.parquet')

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    special_tokens = get_special_tokens(stage1_config)
    if local_rank == 0:
        print(f"Total special tokens: {len(special_tokens)}")

    num_added = tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    tokenized_special_tokens = tokenizer.convert_tokens_to_ids(special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    valid_special_token_ids = []
    valid_special_tokens = []
    for i, token_id in enumerate(tokenized_special_tokens):
        if token_id != tokenizer.unk_token_id:
            valid_special_token_ids.append(token_id)
            valid_special_tokens.append(special_tokens[i])
    assert len(valid_special_token_ids) == len(valid_special_tokens) == num_added
    if local_rank == 0:
        print(f"Valid special tokens: {len(valid_special_token_ids)}")
        print(f"First 10 valid special tokens: {valid_special_tokens[:10]}")
        print(f"Training token IDs range: {min(valid_special_token_ids)} to {max(valid_special_token_ids)}")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=(args.lora_target_modules).split(","),
        task_type=TaskType.CAUSAL_LM,
        trainable_token_indices={
            'embed_tokens': valid_special_token_ids,
        }
    )

    # model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = prepare_chat_dataset(train_data_path, local_rank=local_rank)
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing training data"
    )

    data_collator = CustomDataCollator(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(Path(training_args.output_dir, 'saved_lora_model'))
    tokenizer.save_pretrained(Path(training_args.output_dir, 'saved_lora_model'))


if __name__ == '__main__':
    parser = HfArgumentParser(ParserArguments)
    args, = parser.parse_args_into_dataclasses()
    set_seed(args.seed)
    main(args)
