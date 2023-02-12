# -*- coding: UTF-8 -*-
"""
@Project ：RES 
@File ：utils.py
@Author ：AnthonyZ
@Date ：2022/11/27 22:38
"""
import logging
import os
import torch
import torchvision
import yaml


def init_logger(config):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S'
    )

    # 使用FileHandler输出到文件
    num_list = []
    for filename in os.listdir(config.logdir):
        if 'log' not in filename:
            continue
        num = int(filename.split('.')[0][3:])
        num_list.append(num)
    num = max(num_list) + 1 if num_list != [] else 1
    log_path = os.path.join(config.logdir, "log{}".format(num))
    path = os.path.join(config.logdir, "log{}".format(num), 'log.txt')
    if not os.path.exists(log_path):
        os.mkdir(log_path)
        os.mkdir(os.path.join(log_path, "model"))
    fh = logging.FileHandler(path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    # 添加两个Handler
    logger.addHandler(fh)
    return logger, log_path


def iou_mask(m1, m2, ioubp=False):
    assert m1.shape == m2.shape
    i = torch.sum(torch.logical_and(m1, m2) > 0)
    u = torch.sum(torch.logical_or(m1, m2) > 0)
    if not ioubp:
        if i == 0:
            return 0
        else:
            return i * 1.0 / u


def show_tensor_example(ground_truth, pred, image, temp, path):
    mask = pred
    pred = pred * image
    ground_truth = ground_truth * image
    batch_size = ground_truth.shape[0]
    image_tensor = torchvision.utils.make_grid(torch.cat((ground_truth, pred, mask.repeat(1, 3, 1, 1), temp.repeat(1, 3, 1, 1)), dim=0), nrow=batch_size, padding=100)
    torchvision.utils.save_image(image_tensor, path)


class Config(dict):
    def __init__(self, config_path):
        super().__init__()
        with open(config_path, 'r') as f:
            self._yaml = f.read()
            self._dict = yaml.safe_load(self._yaml)
            self._dict['PATH'] = os.path.dirname(config_path)

    def __getattr__(self, name):
        if self._dict.get(name) is not None:
            return self._dict[name]
        return None

    def print(self):
        print('Model configurations:')
        print('---------------------------------')
        print(self._yaml)
        print('')
        print('---------------------------------')
        print('')


def load_config(path):
    config_path = path
    config = Config(config_path)
    return config
