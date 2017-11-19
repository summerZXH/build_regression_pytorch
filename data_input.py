# coding: utf-8
import numpy as np
import cv2
import logging
import PIL.Image as Image
import json
import os
import torch.utils.data as data
import glob

class TrainImageFolder(data.Dataset):
    def __init__(self, root, json_path, transform=None):
        # get all the image name and its label
        self.root = root
        self.json_path = json_path
        self.transform = transform
        self.image_list, self.label_list, self.Length = self.read_txt()

    def __len__(self):
        return self.Length

    def __getitem__(self, index):
        """Returns one data pair (image and label)."""
        img_label = self.label_list[index]
        img_id = self.image_list[index]

        filename = os.path.join(self.root, img_id)
        image = Image.open(filename).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        target = float(img_label)
        return image, target

    def read_txt(self):
        txt_file = self.json_path
        image_list = []
        label_list = []

        with open(txt_file) as f:
            def_lines = f.readlines()
        def_lines.pop(0)

        for def_line in def_lines:
            image_tmp = def_line.split(' ')[0]
            image_list.append(image_tmp)
            label_tmp = float(def_line.split(' ')[1].split('\n')[0])
            label_list.append(label_tmp)

        numbers = len(image_list)

        return image_list, label_list, numbers


class TestImageFolder(data.Dataset):
    def __init__(self, root, json_path, transform=None):

        self.root = root
        self.json_path = json_path
        self.image_list = self.read_txt()
        self.transform = transform

    def __getitem__(self, index):
        filename = self.image_list[index]
        img = Image.open(os.path.join(self.root, filename))
        if self.transform is not None:
            img = self.transform(img)
        return img, filename

    def __len__(self):
        return len(self.image_list)

    def read_txt(self):
        txt_file = self.json_path
        image_list = []

        with open(txt_file) as f:
            def_lines = f.readlines()
        def_lines.pop(0)

        for def_line in def_lines:
            imgage_tmp = def_line.split(' ')[0]
            image_list.append(imgage_tmp)

        return image_list


if __name__=='__main__':

    root = "/home/zhuxiuhong/src/PycharmProjects/insulator/data/insulator_image"
    json_path = '/home/zhuxiuhong/src/PycharmProjects/insulator/data/insulator_train.txt'
    scene = TrainImageFolder(root, json_path)