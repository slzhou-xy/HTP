from dataclasses import dataclass, field
from peft import (
    PeftModel
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
import torch
import warnings
from pathlib import Path
import yaml
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
        default='/huggingface/hub/Qwen3-1.7B',
        metadata={"help": "Model name"}
    )


def get_special_tokens(config):
    special_tokens = []
    special_tokens.append('<|t_begin|>')
    special_tokens.append('<|t_end|>')
    special_tokens.append('<|p_begin|>')
    special_tokens.append('<|p_end|>')

    vocab_multi = config['RQ_quant']['vocab_multi']
    n_codebooks = config['RQ_quant']['n_codebooks']

    for i in range(2 ** (n_codebooks - 1)):
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


def main(args):
    city = args.city
    stage1_config_path = Path('logs', city, args.exp_name, 'stage1_config.yaml')
    stage1_config = yaml.load(open(stage1_config_path, 'r'), Loader=yaml.FullLoader)

    model_name = args.model_name
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    special_tokens = get_special_tokens(stage1_config)
    # num_added = tokenizer.add_tokens(special_tokens)
    num_added = tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(
        model, f'logs/{city}/{args.exp_name}/stage2_no_oov_ckpt_5e-4/saved_lora_model')
    model.eval()
    model = model.merge_and_unload()
    model.save_pretrained(Path(f'logs/{city}/{args.exp_name}/stage2_ckpt', 'saved_sft_model'))
    tokenizer.save_pretrained(Path(f'logs/{city}/{args.exp_name}/stage2_ckpt', 'saved_sft_model'))


if __name__ == '__main__':
    parser = HfArgumentParser(ParserArguments)
    args, = parser.parse_args_into_dataclasses()
    set_seed(args.seed)
    main(args)
