from utils import pload, set_seed
from pathlib import Path
from transformers import HfArgumentParser
from dataclasses import dataclass, field
import pandas as pd
from tqdm import tqdm
import numpy as np
from datetime import datetime


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


def get_parquet(token_data, rn_data, traj_data, city):
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    days = ['midnight', 'morning', 'afternoon', 'evening']

    road_names = rn_data['road_name'].to_numpy()
    road_lengths = rn_data['length'].to_numpy()

    desc_list = []
    traj_id_list = []
    user_id_list = []
    answer_list = []

    for i, _ in enumerate(tqdm(token_data)):
        traj_id = token_data[i]['traj_id']
        road_seq = token_data[i]['road_seq']
        latent_token = token_data[i]['latent_token']
        len_type_token = token_data[i]['len_type_token']

        traj_info = traj_data[traj_data['traj_id'] == traj_id]
        user_id = traj_info['user_id'].tolist()[0]
        # flag = raw_traj['flag']
        time_seq = traj_info['time'].tolist()[0]
        road_seq2 = traj_info['cpath'].tolist()[0]

        if not np.all(np.array(road_seq) == road_seq2):
            raise ValueError('road_seq is not equal to road_seq2')

        start_date = datetime.fromtimestamp(time_seq[0])
        weekday = weekdays[start_date.weekday()]
        hour = start_date.hour
        minite = start_date.minute
        day_t = days[int(hour / 6)]

        end_date = datetime.fromtimestamp(time_seq[-1])
        duration = end_date - start_date
        minutes, seconds = divmod(duration.total_seconds(), 60)
        travel_time = f"{int(minutes)} minutes and {int(seconds)} seconds"
        start_time = f"{weekday} {day_t} at {hour:02d}:{minite:02d}"

        travel_length = round(sum([road_lengths[r] for r in road_seq]) / 1000, 2)
        travel_length = f"{travel_length} kilometers"

        road_seq_name = []

        for r in road_seq:
            r = road_names[r]
            if r == 'Unnamed road':
                continue
            if not isinstance(r, str):
                continue
            if r == '未知道路':
                continue
            if r[0] == '[':
                r = eval(r)[0]
            road_seq_name.append(r)

        road_seq_name = [road_seq_name[i] for i in range(len(road_seq_name)) if i == 0 or road_seq_name[i] != road_seq_name[i-1]]
        road_seq_name = ', '.join(road_seq_name)

        road_seq = [f'<road_{r}>' for r in road_seq]
        road_seq = ''.join(road_seq)
        latent_token = ['<|p_begin|>' + ''.join(token) + '<|p_end|>' for token in latent_token]
        latent_token = '<|t_begin|>' + ', '.join(latent_token) + '<|t_end|>'

        answer = (
            f'{len_type_token}, {latent_token}'
        )

        if city == 'chengdu':
            interval = 10
        else:
            interval = 15

        desc = (
            f"Here is a summary of the user's travel information: "
            f"The road sequence is {road_seq}, mainly passing through {road_seq_name}. "
            f"The trip started at {start_time}, with an estimated travel time of {travel_time}, "
            f"and a total travel distance of approximately {travel_length}. "
            f"The GPS sampling interval is about one point every {interval} seconds. "
            f"Please generate a sampling type and a travel pattern sequence of this trip."
        )

        traj_id_list.append(traj_id)
        desc_list.append(desc)
        user_id_list.append(user_id)
        answer_list.append(answer)

    return pd.DataFrame(
        {
            'traj_id': traj_id_list,
            'desc': desc_list,
            'user_id': user_id_list,
            'answer': answer_list
        }
    )


def main(args):
    train_data = pload(Path('logs', args.city, args.exp_name, 'data', 'train_latent_token.pkl'))
    rn_data = pd.read_csv(f'../traj_dataset/{args.city}/rn/edge_info.csv')
    raw_traj = pd.read_parquet(f'../traj_dataset/{args.city}/traj.parquet')

    train_data = get_parquet(train_data, rn_data, raw_traj, args.city)
    save_path = Path('logs', args.city, args.exp_name, 'data')
    train_data.to_parquet(Path(save_path, 'llm_train_traj.parquet'))
    print(train_data.head(5))

    test_data = pload(Path('logs', args.city, args.exp_name, 'data', 'test_latent_token.pkl'))

    train_data = get_parquet(test_data, rn_data, raw_traj, args.city)
    save_path = Path('logs', args.city, args.exp_name, 'data')
    train_data.to_parquet(Path(save_path, 'llm_test_traj.parquet'))
    print(train_data.head(5))


if __name__ == "__main__":
    parser = HfArgumentParser(DataArguments)
    args, = parser.parse_args_into_dataclasses()
    set_seed(42)
    main(args)
