from utils import pdump
from dataclasses import dataclass, field
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
import torch
from pathlib import Path
import warnings
from torch.utils.data import DataLoader, Dataset

import pandas as pd
from tqdm import tqdm
import re
from accelerate import Accelerator
from accelerate.utils import gather_object

warnings.filterwarnings("ignore")


@dataclass
class ParserArguments:
    seed: int = field(default=42)
    city: str = field(default='porto')
    exp_name: str = field(
        default='global_percent_ep100_bs512_code_256_8421_64d_en_64d_1224',
        metadata={"help": "Experiment name"}
    )
    ddp: bool = field(
        default=False,
        metadata={"help": "Whether to use DDP"}
    )


def format_chat_prompt(user_content):
    """Format input as chat format prompt"""
    system_message = "You are a professional spatial-temporal generation expert who needs to generate the possible travel pattern and sampling type for users based on their trip information."
    think_block = "<think>\n\n</think>\n\n"
    chat_prompt = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
{think_block}"""
    return chat_prompt


class TestDataset(Dataset):
    """Dataset for loading test data from parquet files"""

    def __init__(self, parquet_file):
        self.df = pd.read_parquet(parquet_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            'traj_id': row['traj_id'],
            'input_ids': row['desc'],
            # 'labels': row['answer'],
            'user_id': row.get('user_id', f'user_{idx}')
        }


class TestCollator:
    """Collator for test data"""

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = "left"

    def __call__(self, batch):
        batch_prompts = []
        # targets = [d["labels"] for d in batch]
        user_ids = [d["user_id"] for d in batch]
        traj_ids = [d["traj_id"] for d in batch]

        for d in batch:
            message = d["input_ids"]
            prompt_text = format_chat_prompt(message)
            batch_prompts.append(prompt_text)

        return {
            "inputs": batch_prompts,
            # "targets": targets,
            "user_ids": user_ids,
            "traj_ids": traj_ids
        }


def main(args):
    accelerator = Accelerator() if args.ddp else None
    is_main_process = accelerator.is_local_main_process if accelerator is not None else True

    city = args.city
    test_data_path = Path('logs', city, args.exp_name, 'data', 'llm_test_traj.parquet')

    model_path = f'logs/{city}/{args.exp_name}/stage2_no_oov_ckpt_5e-4/saved_sft_model'

    if accelerator is not None:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16
        ).to('cuda:0')

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    test_dataset = TestDataset(test_data_path)
    collator = TestCollator(args, tokenizer)

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        collate_fn=collator,
        shuffle=False,
        num_workers=0,  # Use 0 for compatibility
        pin_memory=True,
        drop_last=False,
    )

    if accelerator is not None:
        model, test_loader = accelerator.prepare(model, test_loader)
        model = accelerator.unwrap_model(model)

    generate_results = []

    with torch.no_grad():
        model.eval()
        failed_count = 0
        # disable=not is_main_process
        for batch_data in tqdm(test_loader, ncols=100, desc=f"rank {accelerator.process_index} Gen"):
            inputs_texts = batch_data["inputs"]

            enc = tokenizer(
                inputs_texts,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            enc = {k: v.to(model.device) for k, v in enc.items()}

            generate_kwargs = {
                "input_ids": enc["input_ids"],
                "attention_mask": enc.get("attention_mask", None),
                "max_new_tokens": 256,
                "output_scores": False,
                "return_dict_in_generate": True,
                "temperature": 0.7,
                "output_logits": False,
                "output_attentions": False,
                # "top_k": 10,
                # "top_p": 0.9,
                # "repetition_penalty": 1.1
            }
            output = model.generate(**generate_kwargs)
            output_ids = output["sequences"]
            # scores = output.get("scores", None)
            decoded_text = tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            for i, text in enumerate(decoded_text):
                pattern_list, t_tags, road_seq = extract_response(text)
                if pattern_list is None:
                    failed_count += 1
                    continue

                generate_results.append({
                    'pattern_list': pattern_list,
                    'sampling_type': t_tags,
                    'traj_id': batch_data["traj_ids"][i],
                    'generated_road_seq': road_seq
                })
            accelerator.wait_for_everyone()

    if accelerator is not None:
        all_generate_results = gather_object(generate_results)
        all_failed_counts = gather_object([failed_count])
        total_failed = sum(all_failed_counts)
    else:
        all_generate_results = generate_results
        total_failed = failed_count

    if is_main_process:
        traj_id_set = []
        new_all_generate_results = []
        for result in all_generate_results:
            if result['traj_id'] not in traj_id_set:
                traj_id_set.append(result['traj_id'])
                new_all_generate_results.append(result)

        all_generate_results = new_all_generate_results
        print("Generation complete:")
        print(f"Successful: {len(all_generate_results)}\n")
        print(f"Failed: {total_failed}\n")
        print(f"Total: {len(all_generate_results) + total_failed}")
        save_path = Path('logs', city, args.exp_name, 'data', 'oov_generated_patterns_5e-4.pkl')
        pdump(all_generate_results, save_path)


def extract_response(text):
    p_blocks = re.findall(r"<\|p_begin\|>(.*?)<\|p_end\|>", text)

    pattern_list = [
        list(map(int, re.findall(r"<a_(\d+)>.*?<b_(\d+)>.*?<c_(\d+)>.*?<d_(\d+)>", block)[0]))
        for block in p_blocks
    ]

    t_tags = re.findall(r"<t_\d+>", text)[0]

    road_seq = [int(r) for r in re.findall(r"<road_(\d+)>", text)]

    LENS_TYPE_TOKEN_MAPPING = {
        f'<t_{i}>': str([int(b) for b in format(i, '03b')])
        for i in range(8)
    }

    t_tags = LENS_TYPE_TOKEN_MAPPING[t_tags]

    return pattern_list, t_tags, road_seq


if __name__ == '__main__':
    parser = HfArgumentParser(ParserArguments)
    args, = parser.parse_args_into_dataclasses()
    set_seed(args.seed)
    main(args)
