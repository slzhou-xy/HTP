import pickle
import random
import numpy as np
import torch
import torch.distributed as dist
from timm.scheduler import CosineLRScheduler, StepLRScheduler


def pload(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def pdump(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


class Config(object):
    def __init__(self, dic):
        for key in dic:
            setattr(self, key, dic[key])

    def __str__(self):
        dic = self.__dict__.copy()
        lst = list(filter(
            lambda p: (not p[0].startswith('__')) and not isinstance(p[1], classmethod),
            dic.items()
        ))
        return '\n'.join(['\n' + str(k) + ' = ' + str(v) if i == 0 else str(k) + ' = ' + str(v) for i, (k, v) in enumerate(lst)])


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_everything(seed: int = 42):
    rank = 0
    if dist.is_initialized():
        rank = dist.get_rank()
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置种子
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_optimizer(parameters, args):
    if 'opt' not in args.keys():
        return torch.optim.Adam(params=parameters, lr=args['lr'], **args.optim_args)
    elif args['opt'] == 'AdamW':
        return torch.optim.AdamW(params=parameters, lr=args['lr'])
    elif args['opt'] == 'Adam':
        return torch.optim.Adam(params=parameters, lr=args['lr'])
    else:
        raise NotImplementedError('No Optimizer!')


def get_scheduler(optimizer, args):
    if args['sched'] == 'cos':
        return CosineLRScheduler(optimizer=optimizer,
                                 t_initial=args['n_epochs'],
                                 warmup_t=args['warmup_epoch'],
                                 warmup_lr_init=args['warmup_lr_init'],
                                 lr_min=args['lr_min'])
    elif args['sched'] == 'step':
        return StepLRScheduler(optimizer=optimizer,
                               decay_rate=args['decay_rate'],
                               decay_t=args['decay_t'],
                               warmup_t=args['warmup_epoch'],
                               warmup_lr_init=args['warmup_lr_init'])
    # elif args['sched'] == 'cos_ann':
    #     return CosineAnnealingLR(optimizer=optimizer,
    #                              T_max=args['n_epochs'],
    #                              eta_min=args['lr_min'])
    else:
        raise NotImplementedError('No Scheduler!')


class TemperatureScheduler:
    def __init__(
        self,
        t0: float,
        min_t: float,
        anneal_rate: float,
        step_size: int,
    ) -> None:
        self.t0 = t0
        self.min_t = min_t
        self.anneal_rate = anneal_rate
        self.step_size = step_size
        self.t = t0

    def update_t(self, iter):
        if iter % self.step_size == self.step_size-1:
            self.t = np.maximum(self.t*np.exp(-self.anneal_rate*iter), self.min_t)

    def get_t(self, iter):
        self.update_t(iter)
        return self.t
