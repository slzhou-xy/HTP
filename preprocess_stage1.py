from utils import pdump, set_seed
from pathlib import Path
import os
import numpy as np
import pandas as pd
from transformers import HfArgumentParser
from dataclasses import dataclass, field
from tqdm import tqdm
from loguru import logger
import json
import torch
from torch_geometric.nn import Node2Vec


@dataclass
class DataArguments:
    city: str = field(
        default='porto',
        metadata={"help": "The city name for which the data is being preprocessed."}
    )
    data_dir: str = field(
        default='../traj_dataset',
        metadata={"help": "The input data dir. Should contain the.csv files for the task."}
    )
    seed: int = field(
        default=42,
        metadata={"help": "The random seed for reproducibility."}
    )
    node2vec_dim: int = field(
        default=128,
        metadata={"help": "The dimension of the node2vec embedding."}
    )
    node2vec_epochs: int = field(
        default=20,
        metadata={"help": "The number of epochs for node2vec training."}
    )


def traj_csv_to_pkl(city, traj_data):
    traj_pkl = []
    all_dx = []
    all_dy = []
    all_global_percent_dist = []

    for i in tqdm(range(len(traj_data)), desc='Prepare traj', ascii=' >=', ncols=100):
        global_percent = traj_data.iloc[i]['path_percent']
        global_percent = np.array([0] + global_percent.tolist())
        global_percent_dist = global_percent[1:] - global_percent[:-1]

        dx = traj_data.iloc[i]['dx']
        dy = traj_data.iloc[i]['dy']

        traj_id = traj_data.iloc[i]['traj_id']
        user_id = traj_data.iloc[i]['user_id']
        if city == 'chengdu':
            flag = traj_data.iloc[i]['flag']
        elif city == 'porto':
            flag = traj_data.iloc[i]['call_type']
        time_seq = traj_data.iloc[i]['time']

        gps_seq = np.vstack(traj_data.iloc[i]['geometry'])
        cpath = traj_data.iloc[i]['cpath']

        all_dx.extend(dx.tolist())
        all_dy.extend(dy.tolist())
        all_global_percent_dist.extend(global_percent_dist.tolist())

        traj_dict = {
            'traj_id': traj_id,
            'user_id': user_id,
            'flag': flag,
            'path': cpath,
            'time_seq': time_seq,
            'gps_seq': gps_seq,
            'dxy': np.array([dx, dy]).T,
            'percent_dist': global_percent_dist
        }
        traj_pkl.append(traj_dict)

    z_score = {
        'mean_dx': np.mean(all_dx),
        'mean_dy': np.mean(all_dy),
        'std_dx': np.std(all_dx),
        'std_dy': np.std(all_dy),
        'mean_percent_dist': np.mean(all_global_percent_dist),
        'std_percent_dist': np.std(all_global_percent_dist),
    }

    return traj_pkl, z_score


def train_node2vec(edge_index: torch.Tensor, emb_dim: int, epochs: int, device: str, save_path: str):
    edge_index = edge_index.to(device)
    model = Node2Vec(edge_index, embedding_dim=emb_dim,
                     walk_length=50, context_size=10, walks_per_node=10,
                     num_negative_samples=10, p=1, q=1, sparse=True).to(device)
    if os.path.exists(save_path + f'/node2vec_model_{emb_dim}d.pt'):
        model.load_state_dict(torch.load(save_path + f'/node2vec_model_{emb_dim}d.pt', weights_only=True), strict=True)
        model = model.to(device)
        model.eval()
        node_emb = model()
        return node_emb

    loader = model.loader(batch_size=32, shuffle=True)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.001)

    model.train()
    epoch_train_loss_best = 1e9
    epoch_best = 0
    epoch_patience = 5
    epoch_worse_count = 0
    for ep in range(epochs):
        total_loss = 0
        for pos_rw, neg_rw in tqdm(loader, desc='Train node2vec', ncols=100, ascii=' >='):
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        mean_loss = total_loss / len(loader)
        logger.info(f"[node2vec] epoch={ep}, loss={mean_loss:.8f}")
        if total_loss < epoch_train_loss_best:
            epoch_train_loss_best = total_loss
            epoch_best = ep
            epoch_worse_count = 0
            torch.save(model.state_dict(), save_path + f'/node2vec_model_{emb_dim}d.pt')
        else:
            epoch_worse_count += 1
            if epoch_worse_count >= epoch_patience:
                logger.info(f"[node2vec] early stopping at epoch={ep}")
                break
    logger.info(f"[node2vec] best epoch={epoch_best}, loading model")
    model.load_state_dict(torch.load(save_path + f'/node2vec_model_{emb_dim}d.pt', weights_only=True), strict=True)
    model = model.to(device)
    model.eval()
    node_emb = model()
    return node_emb.cpu().detach()


def preprocess_data(args: DataArguments):
    train_traj_pkl_path = Path(args.data_dir, args.city, 'processed_train_traj.pkl')
    test_traj_pkl_path = Path(args.data_dir, args.city, 'processed_test_traj.pkl')
    z_score_path = Path(args.data_dir, args.city, 'z_score.json')
    road_emb_path = Path(args.data_dir, args.city, f'road_emb_{args.node2vec_dim}d.pt')

    if not os.path.exists(road_emb_path):
        logger.info('Preprocessing road info.')

        edge_graph = pd.read_csv(os.path.join(args.data_dir, args.city, 'rn/edge_graph.csv'))
        start_road_id = edge_graph['from_edge_id']
        end_road_id = edge_graph['to_edge_id']
        edge_index = torch.tensor([start_road_id, end_road_id], dtype=torch.long)

        road_emb = train_node2vec(
            edge_index,
            args.node2vec_dim,
            args.node2vec_epochs,
            args.device,
            Path(args.data_dir, args.city)
        )

        torch.save(road_emb, road_emb_path)

    if not os.path.exists(train_traj_pkl_path):
        logger.info('Preprocessing traj data.')
        traj_data = pd.read_parquet(Path(args.data_dir, args.city, 'traj.parquet'))
        traj_rel_info = pd.read_parquet(Path(args.data_dir, args.city, 'traj_rel_info.parquet'))
        traj_data = pd.concat([traj_data, traj_rel_info[['path_percent', 'dx', 'dy']]], axis=1)
        lens = traj_data.shape[0]

        if os.path.exists(Path(args.data_dir, args.city, 'train_index.npy')):
            train_index = np.load(Path(args.data_dir, args.city, 'train_index.npy'))
            test_index = np.load(Path(args.data_dir, args.city, 'test_index.npy'))
        else:
            index = np.random.permutation(lens)
            train_index = index[:int(lens*0.9)]
            test_index = index[int(lens*0.9):]

            np.save(Path(args.data_dir, args.city, 'train_index'), train_index)
            np.save(Path(args.data_dir, args.city, 'test_index'), test_index)

        train_data = traj_data.iloc[train_index].reset_index(drop=True)
        test_data = traj_data.iloc[test_index].reset_index(drop=True)

        train_traj_pkl, z_score = traj_csv_to_pkl(args.city, train_data)
        pdump(train_traj_pkl, train_traj_pkl_path)
        json.dump(z_score, open(z_score_path, 'w'))

        test_traj_pkl, _ = traj_csv_to_pkl(args.city, test_data)
        pdump(test_traj_pkl, test_traj_pkl_path)


if __name__ == "__main__":
    parser = HfArgumentParser(DataArguments)
    args, = parser.parse_args_into_dataclasses()
    set_seed(args.seed)
    preprocess_data(args)
