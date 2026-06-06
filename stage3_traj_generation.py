from dataclasses import dataclass, field
import json
import torch
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from model.rqvae import RQVAE
from utils import pload, pdump
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import shapely
from shapely.geometry import LineString
from transformers import (
    HfArgumentParser,
    set_seed,
)
import warnings
warnings.filterwarnings("ignore")


@dataclass
class ParserArguments:
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for initialization"}
    )
    exp_name: str = field(
        default='YOUR_NAME',
        metadata={"help": "Experiment name"}
    )
    city: str = field(
        default='porto',
        metadata={"help": "City name"}
    )
    device: str = field(
        default='cpu'
    )


class TestDataset(Dataset):
    def __init__(self, generated_data, real_traj):
        self.data = generated_data
        self.real_traj = real_traj

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_i = self.data[idx]
        road_seq = data_i['generated_road_seq']
        generated_code = data_i['pattern_list']
        sampling_tyep = data_i['sampling_type']
        traj_id = data_i['traj_id']
        real_road_seq = self.real_traj.loc[self.real_traj['traj_id'] == traj_id, 'opath'].values[0]

        return generated_code, road_seq, sampling_tyep, traj_id, real_road_seq


def batch_loader(batch_data, road_geometry, road_geometry_lens):
    bs = len(batch_data)
    generated_code, road_seq, sampling_type, traj_id, real_road_seq = zip(*batch_data)

    codes = []
    code_lens = []
    road_seqs = []
    road_gps_list = []
    road_gps_len = []
    sampling_type_list = []
    traj_id_list = []
    real_road_seqs = []

    for i in range(bs):
        codes.extend(generated_code[i])
        code_lens.append(len(generated_code[i]))
        path_i = torch.tensor(road_seq[i], dtype=torch.long)
        road_gps_list.append(road_geometry[path_i])
        road_gps_len.append(road_geometry_lens[path_i])
        road_seqs.append(path_i + 1)
        traj_id_list.append(traj_id[i])
        sampling_type_list.append(eval(sampling_type[i]))
        real_road_seqs.append(real_road_seq[i])

    road_gps_len = torch.cat(road_gps_len)
    road_gps = torch.cat(road_gps_list, dim=0)[:, :max(road_gps_len)]

    return {
        'codes': torch.LongTensor(codes),
        'code_lens':  torch.LongTensor(code_lens),
        'gen_road_seq': pad_sequence(road_seqs, batch_first=True, padding_value=0),
        'sampling_type': torch.LongTensor(sampling_type_list),
        'road_gps': road_gps,
        'road_gps_len': torch.LongTensor(road_gps_len),
        'traj_id': traj_id_list,
        'real_road_seq': real_road_seqs
    }


def get_dataloader(real_traj, generate_data, data_dir: str, city: str, batch_size: int, mbr: dict):
    road_info = pd.read_csv(Path(data_dir, city, 'rn/edge_info.csv'))

    road_geometry = road_info['geometry'].apply(lambda x: np.array(list(shapely.from_wkt(x).coords))).tolist()
    road_geometry_list = []
    road_geometry_lens = []

    for geometry in road_geometry:
        geometry[:, 0] = (geometry[:, 0] - mbr['min_lon']) / (mbr['max_lon'] - mbr['min_lon'])
        geometry[:, 1] = (geometry[:, 1] - mbr['min_lat']) / (mbr['max_lat'] - mbr['min_lat'])
        road_geometry_lens.append(geometry.shape[0])
        road_geometry_list.append(torch.tensor(geometry, dtype=torch.float32))

    road_geometry_lens = torch.LongTensor(road_geometry_lens)
    road_geometry = pad_sequence(road_geometry_list, batch_first=True, padding_value=0.0)

    dataset = TestDataset(generate_data, real_traj)
    dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=batch_size,
                            collate_fn=lambda x: batch_loader(x, road_geometry, road_geometry_lens))
    return dataloader


def main(args):
    exp_name = args.exp_name
    city = args.city
    device = args.device

    config_path = Path('logs', city, exp_name, 'stage1_config.yaml')
    model_path = Path('logs', city, exp_name, 'rqvae', 'rqvae.pt')
    generated_data_path = Path('logs', city, exp_name, 'data', 'generated_patterns.pkl')
    traj = pd.read_parquet(f'../traj_dataset/{city}/traj.parquet')
    index = np.load(f'../traj_dataset/{city}/test_index.npy')
    real_traj = traj.iloc[index].reset_index(drop=True)

    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    mbr = config['mbr']
    args = config['args_settings']
    data_dir = config['data_dir']

    generated_data = pload(generated_data_path)
    road_emb = torch.load(Path(config['data_dir'], config['city'], 'road_emb_128d.pt'))

    dataloader = get_dataloader(
        real_traj,
        generated_data,
        data_dir,
        city,
        batch_size=2048,
        mbr=mbr,
    )
    model = RQVAE(config=config, road_emb=road_emb)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model = model.to(device)
    model.eval()

    save_data = []
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc='Generate Traj on Road'):
            codes = batch_data['codes'].to(device)
            code_lens = batch_data['code_lens'].to(device)
            gen_road_seq = batch_data['gen_road_seq'].to(device)
            road_gps = batch_data['road_gps'].to(device)
            road_gps_len = batch_data['road_gps_len'].to(device)
            all_next_lens_type = batch_data['sampling_type'].to(device)
            real_road_seq = batch_data['real_road_seq']
            final_preds_percent, final_preds_dxy, traj_lens = model.decode(
                codes, code_lens, all_next_lens_type, gen_road_seq, road_gps, road_gps_len)

            final_preds_percent = final_preds_percent.cpu().numpy()
            final_preds_dxy = final_preds_dxy.cpu().numpy()
            traj_lens = traj_lens.cpu().numpy()
            traj_ids = batch_data['traj_id']

            gen_road_lens = (gen_road_seq != 0).sum(dim=-1).long()
            gen_road_seq = gen_road_seq.cpu().numpy()

            for i in range(len(traj_lens)):
                traj_id = traj_ids[i]
                pred_percent_i = final_preds_percent[i][:traj_lens[i]]
                pred_dxy_i = final_preds_dxy[i][:traj_lens[i]]

                save_data.append({
                    'traj_id': traj_id,
                    'pred_percent': pred_percent_i,
                    'pred_dxy': pred_dxy_i,
                    'gen_road_seq': gen_road_seq[i][:gen_road_lens[i]] - 1,
                    'real_road_seq': real_road_seq[i]
                })

    z_score_path = Path(data_dir, city, 'z_score.json')
    z_score = json.load(open(z_score_path, 'r'))
    rn_data = pd.read_csv(Path(config['data_dir'], config['city'], 'rn/edge_info.csv'))

    generated_gps_traj = []
    for data in tqdm(save_data, desc='Reconstruct GPS Traj'):
        traj_id = data['traj_id']
        pred_percent = data['pred_percent']
        pred_dxy = data['pred_dxy']
        gen_road_seq = data['gen_road_seq']
        real_road_seq = data['real_road_seq']

        road_geo = []
        road_lens = []
        for road in gen_road_seq:
            geo = shapely.from_wkt(rn_data.iloc[road]['geometry'])
            road_lens.append(geo.length)
            geo = list(geo.coords)
            if len(road_geo) == 0:
                road_geo.extend(geo)
            else:
                road_geo.extend(geo[1:])
        road_geo = LineString(road_geo)
        total_len = sum(road_lens)
        cumsum_road_lens = np.cumsum(road_lens)
        cumsum_road_lens_percent = cumsum_road_lens / total_len

        new_traj = []
        pred_percent = pred_percent * z_score['std_percent_dist'] + z_score['mean_percent_dist']
        pred_percent = pred_percent.cumsum()
        pred_percent = np.abs(pred_percent)
        pred_percent = np.clip(pred_percent, 0.0, 0.9999999)
        new_road_traj = []
        for i in range(len(pred_percent)):
            percent = pred_percent[i]
            dxy = pred_dxy[i]
            proj_point = road_geo.interpolate(percent * road_geo.length)
            lon, lat = proj_point.x, proj_point.y
            dx = (dxy[0] * z_score['std_dx']) + z_score['mean_dx']
            dy = (dxy[1] * z_score['std_dy']) + z_score['mean_dy']
            new_traj.append((lon + dx, lat + dy))
        new_road_traj_idx = np.searchsorted(cumsum_road_lens_percent, pred_percent, side='right')
        new_road_traj = gen_road_seq[new_road_traj_idx]

        new_traj = np.array(new_traj)
        generated_gps_traj.append(
            {
                'traj_id': traj_id,
                'gps_traj': new_traj,
                'road_traj': new_road_traj,
            }
        )
    pdump(generated_gps_traj, Path('logs', city, exp_name, 'data', 'generated_trajs.pkl'))


if __name__ == '__main__':
    parser = HfArgumentParser(ParserArguments)
    args, = parser.parse_args_into_dataclasses()
    set_seed(args.seed)
    main(args)
