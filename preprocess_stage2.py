from utils import pdump, set_seed
from pathlib import Path
import yaml
import numpy as np
import os
from transformers import HfArgumentParser
from dataclasses import dataclass, field
from model.rqvae import RQVAE
from tqdm import tqdm
import torch
from loguru import logger
import pandas as pd
from dataloader_stage2 import get_dataloader
import warnings

warnings.filterwarnings("ignore")


@dataclass
class DataArguments:
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for initialization"}
    )
    exp_name: str = field(
        default='global_percent_ep100_bs512_code_256_8421_64d_en_64d_1224',
        metadata={"help": "Experiment name"}
    )
    city: str = field(
        default='chengdu',
        metadata={"help": "City name"}
    )
    device: str = field(
        default='cuda:2',
        metadata={"help": "Device for training"}
    )


def merge_info(info_list):
    if isinstance(info_list[0], dict):
        res = pd.DataFrame(info_list).mean().to_dict()
    else:
        res = []
        for i in range(len(info_list[0])):
            dict_i = [info[i] for info in info_list]
            res.append(pd.DataFrame(dict_i).mean().to_dict())
    return res


def prepare_model(config, road_emb):
    model = RQVAE(
        config=config,
        road_emb=road_emb,
    )
    model_path = Path(config['save_dir'], args.exp_name, 'rqvae', 'rqvae.pt')
    model.load_state_dict(torch.load(model_path, weights_only=True))
    return model, Path(config['save_dir'], args.exp_name, 'data')


@torch.no_grad()
def get_latent_code(model, loader, device):
    loss_list = []
    recon_loss_list = []
    rq_quant_loss_list = []
    rq_info_list = []
    traj_latent_info = []

    model = model.to(device)
    model.eval()
    model.reset_codebook_hit()

    for batch_idx, batch_data in enumerate(tqdm(loader, ncols=100, desc='Get latent code')):
        x = batch_data['x'].to(device)
        traj_id = batch_data['traj_id']
        traj_len = batch_data['traj_len'].to(device)
        road_seq = batch_data['road_seq'].to(device)
        road_gps = batch_data['road_gps'].to(device)
        road_gps_len = batch_data['road_gps_len'].to(device)
        dxy = batch_data['dxy'].to(device)
        road_percent = batch_data['road_percent'].to(device)

        rtn, pattern_lens = model.encode(x, traj_len, road_seq, road_gps, road_gps_len, dxy, road_percent)

        recon_loss, quant_loss, code, info = rtn['recon_loss'], rtn['quant_loss'], rtn['code'], rtn['info']
        loss = recon_loss + quant_loss
        loss = loss.item()
        recon_loss = recon_loss.item()
        rq_quant_loss = quant_loss.item()

        loss_list.append(loss)
        recon_loss_list.append(recon_loss)
        rq_quant_loss_list.append(rq_quant_loss)
        rq_info_list.append(info)

        pattern_lens = pattern_lens.cpu().tolist()
        road_seq_lens = (road_seq != 0).sum(dim=-1).cpu().tolist()

        for i, plen in enumerate(pattern_lens):
            traj_latent_info.append({
                'traj_id': int(traj_id[i].item()),
                'len_type': rtn['lens_type'][i].cpu().long().tolist(),
                'latent_code': code[i, :plen].cpu().numpy(),
                'road_seq': (road_seq[i, :road_seq_lens[i]] - 1).cpu().long().tolist(),
            })

    avg_loss = np.mean(loss_list)
    avg_recon_loss = np.mean(recon_loss_list)
    avg_rq_quant_loss = np.mean(rq_quant_loss_list)

    avg_rq_info = merge_info(rq_info_list)
    avg_rq_info_str_list = [', '.join(f"{k}: {v:.8f}" for k, v in info.items()) for info in avg_rq_info]

    logger.info(f"Eval  Loss: {avg_loss:.8f}, Recon: {avg_recon_loss:.8f}, RQ_quant: {avg_rq_quant_loss:.8f}")
    for i, avg_rq_info_str in enumerate(avg_rq_info_str_list):
        logger.info(f"Eval  |RQ_Info_{i + 1}: {avg_rq_info_str}")

    traj_latent_code = sorted(traj_latent_info, key=lambda x: x['traj_id'])
    return traj_latent_code


def make_token(traj_latent_code, base_rq_vocab_size, vocab_multi, ch_multi):
    LENS_TYPE_TOKEN_MAPPING = {
        str([int(b) for b in format(i, '03b')]): f'<t_{i}>'
        for i in range(2 ** (len(ch_multi) - 1))
    }

    rq_vocab_sizes = [base_rq_vocab_size // vm for vm in vocab_multi]
    token_prefix = [chr(i) for i in range(ord('a'), ord('a') + len(rq_vocab_sizes))]

    LATENT_TOKEN_MAPPING = []
    for i, size in enumerate(rq_vocab_sizes):
        LATENT_TOKEN_MAPPING.extend([f'<{token_prefix[i]}_{n}>' for n in range(size)])

    traj_token_list = []

    for item in tqdm(traj_latent_code, ncols=100, desc='Make extend vocab'):
        len_type_token = LENS_TYPE_TOKEN_MAPPING[str(item['len_type'])]
        latent_token = [[f'<{token_prefix[i]}_{code}>' for i, code in enumerate(row)] for row in item['latent_code']]
        traj_token_list.append({
            'traj_id': item['traj_id'],
            'len_type_token': len_type_token,
            'latent_token': latent_token,
            'road_seq': item['road_seq']
        })

    return traj_token_list


def main(args):
    config = yaml.load(open(Path(f'logs/{args.city}', f'{args.exp_name}', 'stage1_config.yaml'), 'r'), Loader=yaml.FullLoader)
    device = args.device
    road_emb = torch.load(Path(config['data_dir'], config['city'], 'road_emb_128d.pt'))
    train_loader = get_dataloader(
        config['data_dir'],
        config['city'],
        2048,
        config['mbr'],
    )

    model, save_dir = prepare_model(config, road_emb)
    traj_latent_code = get_latent_code(model, train_loader, device)
    traj_latent_token = make_token(
        traj_latent_code,
        config['RQ_quant']['base_vocab_size'],
        config['RQ_quant']['vocab_multi'],
        config['Unet']['ch_multi']
    )

    os.makedirs(Path(f'logs/{args.city}', f'{args.exp_name}', 'data'), exist_ok=True)
    pdump(traj_latent_token, Path(save_dir, 'train_latent_token.pkl'))

    test_loader = get_dataloader(
        config['data_dir'],
        config['city'],
        2048,
        config['mbr'],
        is_eval=True
    )
    traj_latent_code = get_latent_code(model, test_loader, device)
    traj_latent_token = make_token(
        traj_latent_code,
        config['RQ_quant']['base_vocab_size'],
        config['RQ_quant']['vocab_multi'],
        config['Unet']['ch_multi']
    )
    pdump(traj_latent_token, Path(save_dir, 'test_latent_token.pkl'))


if __name__ == "__main__":
    parser = HfArgumentParser(DataArguments)
    args, = parser.parse_args_into_dataclasses()
    set_seed(42)
    main(args)
