#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.375
        # self.input_size = (416, 416)
        self.input_size = (512,288)
        self.mosaic_scale = (0.5, 1.5)
        self.random_size = (10, 20)
        # self.test_size = (416, 416)
        self.test_size = (512,288)

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.enable_mixup = False
        self.data_dir = "datasets/dataset/tb/COCO"

        self.train_ann = "instances_train2017_v3.json"
        self.val_ann = "instances_val2017_v3.json"
        self.num_classes = 2 #2

