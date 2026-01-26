from utils import pload
from typing import List
from pathlib import Path
import pandas as pd
import numpy as np
import shapely
import yaml
import json
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


class Stage1Dataset(Dataset):
    def __init__(self, traj_data: List):
        self.data = traj_data
        self.traj_lens = [len(traj['gps_seq']) for traj in traj_data]
        self.path_lens = [len(traj['path']) for traj in traj_data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.traj_lens[idx], self.path_lens[idx]


def batch_loader(batch_data, mbr, z_score, road_geo, road_geo_lens):
    bs = len(batch_data)
    batch_data, lengths, path_lens = zip(*batch_data)
    x = torch.zeros(bs, max(lengths), 2)
    path = torch.zeros(bs, max(path_lens), dtype=torch.long)
    road_gps_list = []
    road_gps_len = []

    traj_id = []
    traj_len = []

    dxy = []
    road_percent = []

    for i, data_i in enumerate(batch_data):
        length_i = lengths[i]
        plength_i = path_lens[i]
        traj_seq = data_i['gps_seq']
        traj_seq = torch.as_tensor(traj_seq, dtype=torch.float32)
        path_i = data_i['path']

        path_i = torch.tensor(path_i, dtype=torch.long)
        road_gps_list.append(road_geo[path_i])
        road_gps_len.append(road_geo_lens[path_i])

        path[i, :plength_i] = path_i + 1
        traj_seq[:, 0] = (traj_seq[:, 0] - mbr['min_lon']) / (mbr['max_lon'] - mbr['min_lon'])
        traj_seq[:, 1] = (traj_seq[:, 1] - mbr['min_lat']) / (mbr['max_lat'] - mbr['min_lat'])

        # 使用as tensor拷贝数据
        dxy_i = torch.as_tensor(data_i['dxy'], dtype=torch.float32)
        dxy_i[:, 0] = (dxy_i[:, 0] - z_score['mean_dx']) / z_score['std_dx']
        dxy_i[:, 1] = (dxy_i[:, 1] - z_score['mean_dy']) / z_score['std_dy']
        road_percent_i = torch.as_tensor(data_i['percent_dist'], dtype=torch.float32)
        road_percent_i = (road_percent_i - z_score['mean_percent_dist']) / z_score['std_percent_dist']

        x[i, :length_i] = traj_seq
        traj_id.append(data_i['traj_id'])
        traj_len.append(length_i)
        dxy.append(dxy_i)
        road_percent.append(road_percent_i)

    road_gps_len = torch.cat(road_gps_len)
    road_gps = torch.cat(road_gps_list, dim=0)[:, :max(road_gps_len)]
    dxy = pad_sequence(dxy, batch_first=True, padding_value=0.0)
    road_percent = pad_sequence(road_percent, batch_first=True, padding_value=0.0)

    return {
        'traj_id': torch.LongTensor(traj_id),
        'x': x,
        'traj_len': torch.LongTensor(traj_len),
        'road_seq': path,
        'road_gps': road_gps,
        'road_gps_len': road_gps_len,
        'dxy': dxy,
        'road_percent': road_percent
    }


def get_dataloader(data_dir: str, city: str, batch_size: int, mbr: dict, do_eval=False):
    if not do_eval:
        traj_data = pload(Path(data_dir, city, 'processed_train_traj.pkl'))
    else:
        traj_data = pload(Path(data_dir, city, 'processed_test_traj.pkl'))
    road_info = pd.read_csv(Path(data_dir, city, 'rn/edge_info.csv'))
    z_score = json.load(open(Path(data_dir, city, 'z_score.json'), 'r'))

    road_geo = road_info['geometry'].apply(lambda x: np.array(list(shapely.from_wkt(x).coords))).tolist()
    road_geo_list = []
    road_geo_lens = []

    for geo in road_geo:
        geo[:, 0] = (geo[:, 0] - mbr['min_lon']) / (mbr['max_lon'] - mbr['min_lon'])
        geo[:, 1] = (geo[:, 1] - mbr['min_lat']) / (mbr['max_lat'] - mbr['min_lat'])
        road_geo_lens.append(geo.shape[0])
        road_geo_list.append(torch.tensor(geo, dtype=torch.float32))

    road_geo_lens = torch.LongTensor(road_geo_lens)
    road_geo = pad_sequence(road_geo_list, batch_first=True, padding_value=0.0)

    dataset = Stage1Dataset(traj_data)
    dataloader = DataLoader(dataset=dataset, shuffle=True if not do_eval else False, batch_size=batch_size,
                            collate_fn=lambda batch_data: batch_loader(batch_data, mbr, z_score, road_geo, road_geo_lens))
    return dataloader


if __name__ == '__main__':
    from tqdm import tqdm

    config_path = 'config/chengdu/stage1_config.yaml'
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    loader = get_dataloader(config['data_dir'], config['city'], config['batch_size'], config['mbr'], True)
    for i, data in enumerate(tqdm(loader)):
        print('xx')
