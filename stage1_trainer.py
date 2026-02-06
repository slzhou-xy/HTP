from utils import get_optimizer, get_scheduler
from loguru import logger
import os
from tqdm import tqdm
import numpy as np
import torch
import pandas as pd
from torch.utils.tensorboard import SummaryWriter


def setup_tensorboard(log_dir, accelerator, reset=True):
    if not accelerator.is_main_process:
        return None

    tb_dir = os.path.join(log_dir, 'rqvae/tensorboard')

    if reset and os.path.exists(tb_dir):
        import shutil
        shutil.rmtree(tb_dir)
    writer = SummaryWriter(tb_dir)
    return writer


def log_str(accelerator, str):
    if accelerator.is_main_process:
        logger.info(str)


def write_scalar(accelerator, writer, tag, scalar_value, global_step):
    if accelerator.is_main_process:
        writer.add_scalar(tag, scalar_value, global_step)


def merge_info(info_list):
    if isinstance(info_list[0], dict):
        res = pd.DataFrame(info_list).mean().to_dict()
    else:
        res = []
        for i in range(len(info_list[0])):
            dict_i = [info[i] for info in info_list]
            res.append(pd.DataFrame(dict_i).mean().to_dict())
    return res


class Trainer:
    def __init__(self,
                 args,
                 config,
                 model,
                 train_loader,
                 log_dir,
                 model_dir,
                 accelerator,
                 ):
        self.log_dir = log_dir
        self.model_dir = model_dir

        self.writer = setup_tensorboard(log_dir, accelerator)

        self.args = args

        self.config = config
        self.n_epochs = config['optimizer']['n_epochs']
        self.n_steps = len(train_loader) * self.n_epochs

        self.model = model
        self.train_loader = train_loader

        self.optimizer = get_optimizer(self.model.parameters(), self.config['optimizer'])
        self.scheduler = get_scheduler(self.optimizer, self.config['optimizer'])
                     
        self.accelerator = accelerator
        self.model, self.optimizer, self.scheduler, self.train_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler, self.train_loader
        )

        self.is_main_process = self.accelerator.is_main_process

    def train(self):
        self.global_step = 0
        self.eval_global_step = 0

        for epoch in range(self.n_epochs):
            self.train_epoch(epoch)
            self.accelerator.unwrap_model(self.model).reset_codebook_hit()
        if self.is_main_process:
            self.writer.close()

    def train_epoch(self, epoch):
        loss_list = []
        recon_loss_list = []
        quant_loss_list = []
        percent_loss_list = []
        dxy_loss_list = []

        rq_info_list = []

        pbar = tqdm(self.train_loader, ncols=110, disable=not self.accelerator.is_main_process)

        self.model.train()
        self.scheduler.step(epoch)

        for batch_idx, batch_data in enumerate(pbar):
            loss, recon_loss, quant_loss, percent_loss, dxy_loss, rq_info = self.train_step(batch_data)

            loss_list.append(loss)
            recon_loss_list.append(recon_loss)
            quant_loss_list.append(quant_loss)
            percent_loss_list.append(percent_loss)
            dxy_loss_list.append(dxy_loss)

            rq_info_list.append(rq_info)

            if self.writer is not None:
                write_scalar(self.accelerator, self.writer, 'train/loss', loss, self.global_step)
                write_scalar(self.accelerator, self.writer, 'train/recon_loss', recon_loss, self.global_step)
                write_scalar(self.accelerator, self.writer, 'train/quant_loss', quant_loss, self.global_step)
                write_scalar(self.accelerator, self.writer, 'train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
                write_scalar(self.accelerator, self.writer, 'train/percent_loss', percent_loss, self.global_step)
                write_scalar(self.accelerator, self.writer, 'train/dxy_loss', dxy_loss, self.global_step)

            pbar.set_description(
                f"[Train {epoch+1}/{self.n_epochs}|L:{loss:.3f}|R:{recon_loss:.3f}|Q:{quant_loss:.3f}||P:{percent_loss:.3f}|D:{dxy_loss:.3f}]")

            self.global_step += 1

        avg_loss = np.mean(loss_list)
        avg_quant_loss = np.mean(quant_loss_list)
        avg_recon_loss = np.mean(recon_loss_list)
        avg_percent_loss = np.mean(percent_loss_list)
        avg_dxy_loss = np.mean(dxy_loss_list)

        avg_rq_info = merge_info(rq_info_list)
        avg_rq_info_str_list = [', '.join(f"{k}: {v:.8f}" for k, v in info.items()) for info in avg_rq_info]

        log_str(self.accelerator,
                f"Train Ep {epoch+1}/{self.n_epochs} | loss: {avg_loss:.8f}, recon: {avg_recon_loss:.8f}, quant: {avg_quant_loss:.8f}, percent: {avg_percent_loss:.8f}, dxy: {avg_dxy_loss:.8f}")

        for i, avg_rq_info_str in enumerate(avg_rq_info_str_list):
            log_str(self.accelerator, f"Train Ep {epoch+1}/{self.n_epochs}|RQ_Info_{i + 1}: {avg_rq_info_str}")

        if self.accelerator.is_main_process:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            torch.save(unwrapped_model.state_dict(), os.path.join(self.model_dir, "rqvae.pt"))

    def train_step(self, batch_data):
        self.optimizer.zero_grad()
        x = batch_data['x'].to(self.accelerator.device)
        road_seq = batch_data['road_seq'].to(self.accelerator.device)
        road_gps = batch_data['road_gps'].to(self.accelerator.device)
        road_gps_len = batch_data['road_gps_len'].to(self.accelerator.device)
        traj_len = batch_data['traj_len'].to(self.accelerator.device)
        dxy = batch_data['dxy'].to(self.accelerator.device)
        road_percent = batch_data['road_percent'].to(self.accelerator.device)

        rtn = self.model(x, traj_len, road_seq, road_gps, road_gps_len, dxy, road_percent)
        recon_loss, quant_loss, _, info = rtn['recon_loss'], rtn['quant_loss'], rtn['code'], rtn['info']
        loss = recon_loss + quant_loss

        self.accelerator.backward(loss)
        self.optimizer.step()

        return (
            loss.item(),
            recon_loss.item(),
            quant_loss.item(),
            rtn['percent_loss'].item(),
            rtn['dxy_loss'].item(),
            info
        )
