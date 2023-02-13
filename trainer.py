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
from data import LFWPeople
from utils import init_logger, show_tensor_example


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
            out_channel=config.channel,
            inner_channel=config.inner_channel,
            norm_groups=config.norm_groups,
            channel_mults=config.channel_mults,
            attn_res=config.attn_res,
            res_blocks=config.num_resblocks,
            dropout=config.dropout,
            with_noise_level_emb=config.with_noise_level_emb,
            image_size=config.image_size
            ).to(self.device)

        self.epochs = config.epochs
        self.num_train_timesteps = config.num_train_timesteps
        self.num_eval_timesteps = config.num_eval_timesteps

        train_data = LFWPeople(image_size=config.image_size, path=config.data)

        self.train_batch_size = config.train_batch_size
        self.eval_batch_size = config.eval_batch_size
        self.dataloader = DataLoader(dataset=train_data, num_workers=config.num_workers, batch_size=self.train_batch_size, shuffle=config.shuffle)

        self.channel = config.channel
        self.image_size = config.image_size

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
        train_tq = tqdm(self.dataloader, desc="Training Epoch " + str(epoch))
        for index, batch in enumerate(train_tq):
            self.network.train()
            optimizer.zero_grad()
            truth = batch[0]
            t = torch.randint(0, self.num_train_timesteps, (truth.shape[0],)).long()
            noisy_image, noise_ref = self.diffusion.noisy_image(t, truth)
            noise_pred = self.diffusion.noise_prediction(
                denoise_fn=self.network,
                x=noisy_image.to(self.device),
                t=t.to(self.device)
            )
            loss = self.loss(noise_ref.to(self.device), noise_pred)
            loss.backward()
            optimizer.step()
            train_tq.set_postfix({"loss": "%.3g" % loss.item()})

    def val(self, epoch):
        to_torch = partial(torch.tensor, dtype=torch.float32)
        with torch.no_grad():
            self.network.eval()
            T = self.num_eval_timesteps
            # alphas = np.linspace(1e-4, 0.09, T)
            # gammas = np.cumprod(alphas, axis=0)
            betas = np.linspace(1e-6, 0.01, T)
            alphas = 1. - betas
            gammas = np.cumprod(alphas, axis=0)
            x_t = torch.randn((self.eval_batch_size, self.channel, self.image_size[0], self.image_size[1]))
            tq_val = tqdm(range(T), desc="Generating "+str(epoch))
            for t in tq_val:
                if t == 0:
                    z = torch.randn_like(x_t.float())
                else:
                    z = torch.zeros_like(x_t.float())
                time = (torch.ones((x_t.shape[0],)) * t).long()
                x_t = extract(to_torch(np.sqrt(1 / alphas)), time, x_t.shape) \
                    * (
                            x_t - (extract(to_torch((1 - alphas) / np.sqrt(1 - gammas)), time, x_t.shape)) * self.network(x_t.to(self.device), time.to(self.device)).detach().cpu()
                    ) \
                    + extract(to_torch(np.sqrt(1 - alphas)), time, z.shape) * z
            if not os.path.exists(os.path.join(self.log_path, "example")):
                os.mkdir(os.path.join(self.log_path, "example"))
                example_path = os.path.join(
                    self.log_path,
                    "example",
                    "example_{}_index{}.png".format(self.num_eval_timesteps, epoch)
                )
                print(x_t.shape)
                show_tensor_example(x_t, example_path)

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

