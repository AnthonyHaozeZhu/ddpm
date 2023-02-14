# -*- coding: UTF-8 -*-
"""
@Project ：RES 
@File ：trainer.py
@Author ：AnthonyZ
@Date ：2023/1/6 00:07
"""

import os

from model.network import UNet
from model.diffusion import DenoiseDiffusion
from data import LFWPeople
from utils import init_logger, show_tensor_example


import torch
import torch.optim as optim
import torch.nn as nn

from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
import shutil


class Trainer:
    def __init__(self, config, config_path):
        self.device = config.device

        self.network = UNet(
            ).to(self.device)
        print(UNet)
        self.diffusion = DenoiseDiffusion(eps_model=self.network, n_steps=config.num_train_timesteps, device=self.device)

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
        shutil.copyfile(config_path, os.path.join(self.log_path, "conf.yaml"))

    def save_model(self, name, EMA=False):
        if not EMA:
            torch.save(self.network.state_dict(), name)
        # else:
        #     torch.save(self.ema_model.state_dict(), name)

    def train(self, epoch):
        optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        print('Starting Training')
        train_tq = tqdm(self.dataloader, desc="Training Epoch " + str(epoch))
        self.network.train()
        for index, batch in enumerate(train_tq):
            self.network.train()
            optimizer.zero_grad()
            truth = batch[0]
            loss = self.diffusion.loss(truth.to(self.device))
            loss.backward()
            optimizer.step()
            train_tq.set_postfix({"loss": "%.3g" % loss.item()})

    def val(self, epoch):
        self.network.eval()
        with torch.no_grad():
            self.network.eval()
            sample = self.diffusion.sample(self.eval_batch_size, self.num_eval_timesteps, 3, self.image_size)
            if not os.path.exists(os.path.join(self.log_path, "example")):
                os.mkdir(os.path.join(self.log_path, "example"))
            example_path = os.path.join(
                self.log_path,
                "example",
                "example_{}_index{}.png".format(self.num_eval_timesteps, epoch)
            )
            show_tensor_example(sample, example_path)

    def main_step(self):
        self.logger.info("  Num Epochs = %d", self.epochs)
        self.logger.info("  Training Batch size = %d", self.train_batch_size)
        self.logger.info("  Generating Batch size = %d", self.eval_batch_size)
        self.logger.info("  Training Timesteps = %d", self.num_train_timesteps)
        self.logger.info("  Generating Timesteps = %d", self.num_eval_timesteps)
        for epoch in range(self.epochs):
            self.train(epoch)
            self.val(epoch)
            model_path = os.path.join(self.log_path, "model", "{}.pth".format(epoch))
            self.logger.info("\n  Saving model into {}".format(model_path))
            self.save_model(model_path)

