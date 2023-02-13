# -*- coding: UTF-8 -*-
"""
@Project ：RES 
@File ：main.py
@Author ：AnthonyZ
@Date ：2022/11/19 20:37
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '6'
import argparse
from utils import load_config
from trainer import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./conf.yaml", help="The path of the config file.")
    args = parser.parse_args()
    config = load_config(args.config)
    print('Config loaded')
    trainer = Trainer(config, args.config)
    trainer.main_step()
