from utils import Config
from pathlib import Path
from loguru import logger
from model.rqvae import RQVAE
from dataloader_stage1 import get_dataloader
import os
import yaml
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed, DistributedDataParallelKwargs
from transformers import HfArgumentParser
from dataclasses import dataclass, field, asdict
from stage1_trainer import Trainer
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class TrainingArguments:
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for initialization"}
    )
    exp_name: str = field(
        default='global_percent_ep100_bs512_code_256_1111_64d_en_64d_1224',
        metadata={"help": "Experiment name"}
    )
    city: str = field(
        default='chengdu',
        metadata={"help": "City name"}
    )


def main_epoch(args, accelerator):
    city = args.city
    config_path = Path('config', city, 'stage1_config.yaml')
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    road_emb = torch.load(Path(config['data_dir'], config['city'], 'road_emb_128d.pt'))

    train_loader = get_dataloader(
        config['data_dir'],
        config['city'],
        config['batch_size'],
        config['mbr'],
    )

    model = RQVAE(config=config, road_emb=road_emb)

    log_dir = Path(config['save_dir'], args.exp_name)
    model_dir = Path(config['save_dir'], args.exp_name, 'rqvae')

    if accelerator.is_main_process:
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        # save model config
        args_dict = asdict(args)
        config['args_settings'] = args_dict
        config['num_roads'] = road_emb.shape[0]
        with open(Path(log_dir, 'stage1_config.yaml'), 'w') as f:
            yaml.dump(config, f)

        logger.add(
            sink=f'{log_dir}/train_log.log',
            mode='w',
            format="[{time:YYYY-M-D HH:mm:ss}] [{module}:{line}] -> {message}"
        )

        logger.info(Config(config))
        logger.info(f'Model size: {sum(p.numel() for p in model.parameters())}')
        logger.info(model)

    trainer = Trainer(
        args=args,
        config=config,
        model=model,
        train_loader=train_loader,
        log_dir=log_dir,
        model_dir=model_dir,
        accelerator=accelerator,
    )

    trainer.train()


if __name__ == '__main__':
    parser = HfArgumentParser(TrainingArguments)
    args, = parser.parse_args_into_dataclasses()
    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    set_seed(args.seed)
    main_epoch(args, accelerator)
