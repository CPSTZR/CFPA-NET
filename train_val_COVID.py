import math
import os
import warnings

import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import joint_transforms
from Utils.utils import clip_gradient
from datasets import COVID_DATASET, ImgFolder
from lossfunction import structure_loss
from mics import AvgMeter, Meandice
from network import Network

warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_path = "./datas/Infection Segmentation Data/Train"
val_path = "./datas/Infection Segmentation Data/Val"

# 用于同时增强Image和Mask
double_transforms = joint_transforms.Compose([
    joint_transforms.Resize((256, 256)),
    joint_transforms.RandomHorizontallyFlip(0.6),
    joint_transforms.RandomRotate(30),
    joint_transforms.RandomVerticalFlip(0.6)
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.330, 0.330, 0.330], [0.204, 0.204, 0.204])
])
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.330, 0.330, 0.330], [0.204, 0.204, 0.204])
])
target_transform = transforms.ToTensor()
val_target_transform = transforms.Compose(
    [transforms.Resize((256, 256)), transforms.ToTensor()])  # transforms.Resize((256,256)),


def train(model, train_loader, optimizer):
    model.train()
    loss_record = AvgMeter()
    for pack in tqdm(train_loader):
        optimizer.zero_grad()
        image, gt, label = pack
        image = Variable(image).cuda()
        gts = Variable(gt).cuda()
        bs = image.shape[0]
        preds = model(image)
        loss_init = structure_loss(preds[0], gts) + structure_loss(preds[1], gts) + structure_loss(preds[2], gts)
        loss_final = structure_loss(preds[3], gts)
        loss = loss_init + loss_final
        loss.backward()
        clip_gradient(optimizer, 0.5)
        optimizer.step()
        loss_record.update(loss.item(), bs)
    return loss_record.avg


def validation(model, val_loader):
    model.eval()
    meandice = AvgMeter()
    with torch.no_grad():
        for pack in tqdm(val_loader):
            image, gt, label = pack
            bs = gt.shape[0]
            image = Variable(image).cuda()
            # gt = Variable(gt).cuda()
            preds = model(image)
            dice = Meandice(preds[3], gt)
            meandice.update(dice, bs)

    return meandice.avg


if __name__ == "__main__":
    x_train, y_train = COVID_DATASET(train_path)
    x_val, y_val = COVID_DATASET(val_path)
    train_set = ImgFolder(x_train, y_train, double_transforms, transform, target_transform)
    val_set = ImgFolder(x_val, y_val, None, val_transform, val_target_transform)
    model = Network().cuda()
    params = model.parameters()
    optimizer = optim.Adam(params, lr=0.0001, weight_decay=1e-4)
    num_epochs = 100

    train_loader = DataLoader(train_set, batch_size=8, num_workers=4, shuffle=True)

    val_loader = DataLoader(val_set, batch_size=8, num_workers=4, shuffle=True)

    lf = lambda x: ((1 + math.cos(x * math.pi / num_epochs)) / 2) * (1 - 0.1) + 0.1
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    best = 0
    for epoch in range(num_epochs):
        loss = train(model, train_loader, optimizer)
        if epoch > num_epochs / 10 * 9:
            save_path = 'COVID1/CFAR{}-net.pth'.format(epoch)
            torch.save(model.state_dict(), save_path)
        print(loss)
        meandice = validation(model, val_loader)
        if meandice > best:
            best = meandice
            save_path = 'COVID1/CFAR-best.pth'
            torch.save(model.state_dict(), save_path)
            print('[Saving best:]', save_path, meandice)
