# -*- coding: UTF-8 -*-
"""
@Project ：RES 
@File ：data.py
@Author ：AnthonyZ
@Date ：2022/11/20 17:57
"""

import os
import json

import torch
import CLIP
from torch.utils.data.dataset import Dataset
import numpy as np
import cv2
from PhraseCutDataset.utils.refvg_loader import RefVGLoader
from skimage.draw import polygon2mask
from PIL import Image
from torchvision.transforms import transforms
import torch.nn.functional as F


class PhraseCut(Dataset):
    def __init__(self, config, train=True):
        if train:
            self.refvg_loader = RefVGLoader(split="train")
        else:
            self.refvg_loader = RefVGLoader(split="test")
        self.sample_ids = [(i, j)
                           for i in self.refvg_loader.img_ids
                           for j in range(len(self.refvg_loader.get_img_ref_data(i)['phrases']))]
        self.base_path = config.data
        self.image_size = config.image_size
        self.args = config
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def load_sample(self, sample_i, j):
        img_ref_data = self.refvg_loader.get_img_ref_data(sample_i)
        polys_phrase0 = img_ref_data['gt_Polygons'][j]
        phrase = img_ref_data['phrases'][j]
        masks = []
        for polys in polys_phrase0:
            for poly in polys:
                poly = [p[::-1] for p in poly]  # swap x,y
                masks += [polygon2mask((img_ref_data['height'], img_ref_data['width']), poly)]
        seg = np.stack(masks).max(0)
        img = np.array(Image.open(os.path.join(self.base_path, "images", str(img_ref_data['image_id']) + '.jpg')))
        img = torch.from_numpy(img)
        if len(img.shape) == 2:
            img = img.unsqueeze(0).repeat(3, 1, 1).unsqueeze(1).float()
        else:
            img = img.permute(2, 0, 1).unsqueeze(0).float()
        seg = seg.astype("uint8")
        seg = torch.from_numpy(seg).view(1, 1, *seg.shape)
        seg = F.interpolate(seg, self.image_size, mode='nearest')[0, 0]
        img = F.interpolate(img, self.image_size, mode='bilinear', align_corners=True)[0]
        img /= 255.0
        if self.args.mask_channel == 3:
            seg = seg.unsqueeze(0).repeat(3, 1, 1)
        else:
            seg = seg.unsqueeze(0)
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        phrase = CLIP.clip.tokenize(phrase)
        return img, seg, phrase

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, item):
        sample_i, j = self.sample_ids[item]
        return self.load_sample(sample_i, j)


class ClevrData(Dataset):
    def __init__(self, args, is_train=True):
        if is_train:
            self.imag_path = os.path.join(args.data, "images/train")
            self.ref_path = os.path.join(args.data, "refexps/clevr_ref+_train_refexps.json")
            self.scenes_path = os.path.join(args.data, "scenes/clevr_ref+_train_scenes.json")
        else:
            self.imag_path = os.path.join(args.data, "images/test")
            self.ref_path = os.path.join(args.data, "refexps/clevr_ref+_val_refexps.json")
            self.scenes_path = os.path.join(args.data, "scenes/clevr_ref+_val_scenes.json")
        self.scenes = None
        self.exps = None
        self.imgid_scenes = {}
        self.load_scene_refexp()

    def load_scene_refexp(self):
        print('loading scene.json...')
        scenes = json.load(open(self.scenes_path))
        self.scenes = scenes['scenes']
        print('loading refexp.json...')
        self.exps = json.load(open(self.ref_path))['refexps'][:]
        print('loading json done')
        for sce in self.scenes:
            img_id = sce['image_index']
            self.imgid_scenes[img_id] = sce

    def get_mask_from_refexp(self, refexp, height=-1, width=-1):
        sce = self.get_scene_of_refexp(refexp)
        obj_list = self.get_refexp_output_objectlist(refexp)
        heatmap = np.zeros((320, 480))

        def from_imgdensestr_to_imgarray(imgstr):
            img = []
            cur = 0
            for num in imgstr.split(','):
                num = int(num)
                img += [cur]*num
                cur = 1-cur
                img = np.asarray(img).reshape((320,480))
            return img

        for objid in obj_list:
            obj_mask = sce['obj_mask'][str(objid+1)]
            mask_img = from_imgdensestr_to_imgarray(obj_mask)
            heatmap += mask_img
        if height != -1 and width !=-1:
            heatmap = cv2.resize(heatmap, (width, height))
        return heatmap

    def get_scene_of_refexp(self, exp):
        image_index = exp['image_index']
        sce = self.imgid_scenes[image_index]
        return sce

    def get_refexp_output_objectlist(self, exp):
        prog = exp['program']
        # image_filename = exp['image_filename']
        last = prog[-1]
        obj_list = last['_output']
        return obj_list

    def __len__(self):
        return 0

    def __getitem__(self, item):
        return item


