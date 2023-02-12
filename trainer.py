# -*- coding: UTF-8 -*-
"""
@Project ：RES 
@File ：trainer.py
@Author ：AnthonyZ
@Date ：2023/1/6 00:07
"""

import os
from functools import partial

from model.network import UNet
from model.diffusion import GaussianDiffusion, extract
from data import ClevrData, PhraseCut
from utils import init_logger, iou_mask


import torch
import torch.optim as optim
import torch.nn as nn

from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm


class Trainer:
    def __init__(self, config):
        self.device = config.device
        self.diffusion = GaussianDiffusion(config.num_train_timesteps)
        self.network = UNet(
            in_channel=config.channel,
            out_channel=config.mask_channel,
            inner_channel=config.inner_channel,
            norm_groups=config.norm_groups,
            channel_mults=config.channel_mults,
            attn_res=config.attn_res,
            res_blocks=config.res_blocks,
            dropout=config.dropout,
            with_noise_level_emb=config.with_noise_level_emb,
            image_size=config.image_size
            ).to(self.device)

        self.gate = config.gate
        self.epochs = config.epochs
        self.num_train_timesteps = config.num_train_timesteps
        self.num_eval_timesteps = config.num_eval_timesteps

        train_data = PhraseCut(config)
        test_data = PhraseCut(config, False)

        self.train_batch_size = config.train_batch_size
        self.eval_batch_size = config.eval_batch_size
        self.dataloader_train = DataLoader(dataset=train_data, num_workers=config.num_workers, batch_size=self.train_batch_size, shuffle=config.shuffle)
        self.dataloader_validation = DataLoader(dataset=test_data, num_workers=config.num_workers, batch_size=self.eval_batch_size, shuffle=False)

        self.learning_rate = config.learning_rate
        if config.loss == 'L1':
            self.loss = nn.L1Loss()
        if config.loss == 'L2':
            self.loss = nn.MSELoss()
        else:
            print('Loss not implemented, setting the loss to L2 (default one)')

        # log保存部分
        if not os.path.exists(config.logdir):
            os.mkdir(config.logdir)
        self.logger, self.log_path = init_logger(config)

    def save_model(self, name, EMA=False):
        if not EMA:
            torch.save(self.network.state_dict(), name)
        # else:
        #     torch.save(self.ema_model.state_dict(), name)

    def train(self, epoch):
        optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)

        print('Starting Training')

        train_tq = tqdm(self.dataloader_train, desc="Training Epoch " + str(epoch))

        for index, (img, truth, phrase) in enumerate(train_tq):
            self.network.train()
            optimizer.zero_grad()
            phrase = phrase.to(self.device)
            t = torch.randint(0, self.num_train_timesteps, (truth.shape[0],)).long()
            noisy_image, noise_ref = self.diffusion.noisy_image(t, truth)
            noise_pred = self.diffusion.noise_prediction(
                denoise_fn=self.network,
                y_noisy=noisy_image.to(self.device),
                image=img.to(self.device),
                promote=phrase,
                t=t.to(self.device)
            )
            loss = self.loss(noise_ref.to(self.device), noise_pred)
            loss.backward()
            optimizer.step()
            train_tq.set_postfix({"loss": "%.3g" % loss.item()})

    def val(self, epoch):
        to_torch = partial(torch.tensor, dtype=torch.float32)
        tq_val = tqdm(self.dataloader_validation, desc="Testing Epoch" + str(epoch))
        IoU = 0.0
        total = 0
        with torch.no_grad():
            self.network.eval()
            for index, (img, truth, phrase) in enumerate(tq_val):
                phrase = phrase.to(self.device)
                T = self.num_eval_timesteps
                # alphas = np.linspace(1e-4, 0.09, T)
                # gammas = np.cumprod(alphas, axis=0)
                betas = np.linspace(1e-6, 0.01, T)
                alphas = 1. - betas
                gammas = np.cumprod(alphas, axis=0)
                y = torch.randn_like(truth.float())
                for t in range(T):
                    if t == 0:
                        z = torch.randn_like(truth.float())
                    else:
                        z = torch.zeros_like(truth.float())
                    time = (torch.ones((truth.shape[0],)) * t).long()
                    y = extract(
                        to_torch(np.sqrt(1 / alphas)), time, y.shape) \
                        * (y - (extract(to_torch((1 - alphas) / np.sqrt(1 - gammas)), time, y.shape)) * self.network(img.to(self.device), y.to(self.device), phrase, time.to(self.device)).detach().cpu())\
                        + extract(to_torch(np.sqrt(1 - alphas)), time, z.shape) * z
                y = (torch.sigmoid(y) > self.gate)
                m = 0.0
                for i in range(img.shape[0]):
                    m += iou_mask(y[i], truth[i])
                IoU += m
                total += img.shape[0]
                tq_val.set_postfix({'mIoU': (IoU / total)})
            mIoU = IoU / total
            self.logger.info("***** Eval results *****")
            self.logger.info('\nEpoch:{}, Validation set: mIoU: {:.5f}\n'.format(epoch, mIoU))

    def main_step(self):
        self.logger.info("  Num Epochs = %d", self.epochs)
        self.logger.info("  Training Batch size = %d", self.train_batch_size)
        self.logger.info("  Testing Batch size = %d", self.eval_batch_size)
        for epoch in range(self.epochs):
            self.train(epoch)
            self.val(epoch)
            model_path = os.path.join(self.log_path, "model", "{}.pth".format(epoch))
            self.logger.info("\n  Saving model into {}".format(model_path))
            self.save_model(model_path)
