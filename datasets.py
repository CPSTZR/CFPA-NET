import json
import os
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import torchvision.transforms.functional as F


def BUSI_DATASET(root):
    img_num_class = [cla for cla in os.listdir(root)
                     if os.path.isdir(os.path.join(root, cla))]
    img_num_class.sort()
    class_indices = dict((k, v) for v, k in enumerate(img_num_class))

    image_class1 = class_indices['benign']
    img_dir = os.listdir(os.path.join(root, 'benign', 'images'))
    a = []
    label = []
    for path in img_dir:
        z = (os.path.join(root, 'benign', 'images', path),
             os.path.join(root, 'benign', 'masks', path), image_class1)
        a.append(z)
        label.append(image_class1)

    image_class2 = class_indices['malignant']
    img_dir = os.listdir(os.path.join(root, 'malignant', 'images'))
    b = []
    for path in img_dir:
        z = (os.path.join(root, 'malignant', 'images', path),
             os.path.join(root, 'malignant', 'masks', path), image_class2)
        b.append(z)
        label.append(image_class2)

    image_class3 = class_indices['normal']
    img_dir = os.listdir(os.path.join(root, 'normal', 'images'))
    c = []
    for path in img_dir:
        z = (os.path.join(root, 'normal', 'images', path),
             os.path.join(root, 'normal', 'masks', path), image_class3)
        c.append(z)
        label.append(image_class3)
    out = a + b + c
    return out, label


def COVID_DATASET(root):
    img_num_class = [cla for cla in os.listdir(root)
                     if os.path.isdir(os.path.join(root, cla))]
    img_num_class.sort()
    class_indices = dict((k, v) for v, k in enumerate(img_num_class))

    image_class1 = class_indices['COVID-19']
    img_dir = os.listdir(os.path.join(root, 'COVID-19', 'images'))
    a = []
    label = []
    for path in img_dir:
        z = (os.path.join(root, 'COVID-19', 'images', path),
             os.path.join(root, 'COVID-19', 'infection masks', path), image_class1)
        a.append(z)
        label.append(image_class1)

    image_class2 = class_indices['Non-COVID']
    img_dir = os.listdir(os.path.join(root, 'Non-COVID', 'images'))
    b = []
    for path in img_dir:
        z = (os.path.join(root, 'Non-COVID', 'images', path),
             os.path.join(root, 'Non-COVID', 'infection masks', path), image_class2)
        b.append(z)
        label.append(image_class2)

    image_class3 = class_indices['Normal']
    img_dir = os.listdir(os.path.join(root, 'Normal', 'images'))
    c = []
    for path in img_dir:
        z = (os.path.join(root, 'Normal', 'images', path),
             os.path.join(root, 'Normal', 'infection masks', path), image_class3)
        c.append(z)
        label.append(image_class3)
    out = a + b + c

    return out, label

def KVASIR_DATASET(root):
    img_num_class = [cla for cla in os.listdir(root)
                     if os.path.isdir(os.path.join(root, cla))]
    img_num_class.sort()
    class_indices = dict((k, v) for v, k in enumerate(img_num_class))

    image_class1 = class_indices['polyps']
    img_dir = os.listdir(os.path.join(root, 'polyps', 'images'))
    a = []
    label = []
    for path in img_dir:
        z = (os.path.join(root, 'polyps', 'images', path),
             os.path.join(root, 'polyps', 'masks', path), image_class1)
        a.append(z)
        label.append(image_class1)

    image_class2 = class_indices['normal']
    img_dir = os.listdir(os.path.join(root, 'normal', 'images'))
    b = []
    for path in img_dir:
        z = (os.path.join(root, 'normal', 'images', path),
             os.path.join(root, 'normal', 'masks', path), image_class2)
        b.append(z)
        label.append(image_class2)
    out = a + b
    return out, label

class ImgFolder(data.Dataset):
    def __init__(self, imgs, labels, joint_transform=None, transform=None, target_transform=None):
        self.imgs = imgs
        self.target = labels
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path, gt_path, label = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path).convert('L')
        cla = label
        if self.joint_transform is not None:
            img, target = self.joint_transform(img, target)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, cla

    def collate(self, batch):
        size = [224, 256][np.random.randint(0, 2)]
        image, mask, cla = [list(item) for item in zip(*batch)]
        for i in range(len(batch)):
            image[i] = F.resize(image[i], size, F.InterpolationMode.BILINEAR)
            mask[i] = F.resize(mask[i], size, F.InterpolationMode.BILINEAR)
        image = torch.from_numpy(np.stack(image, axis=0))
        mask = torch.from_numpy(np.stack(mask, axis=0))
        return image, mask, cla

    def __len__(self):
        return len(self.imgs)