import warnings

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import joint_transforms
from datasets import ImgFolder, BUSI_DATASET
from mics import AvgMeter, cal
from network import Network

warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

val_path = "breast"

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


def validation(model, test_loader):
    model.eval()
    meandice, meaniou, meanmae, meansm = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()

    with torch.no_grad():
        for image, mask, label, path in tqdm(test_loader):
            bs = mask.shape[0]
            image = image.cuda().float()
            mask = mask.float()
            # gt = Variable(gt).cuda()
            preds = model(image)
            out = preds[3]
            # dice = Meandice(out, mask)
            dice, iou, em, sm = cal(out, mask, path)
            meandice.update(dice, bs)
            meaniou.update(iou, bs)
            meanmae.update(em, bs)
            meansm.update(sm, bs)
    return meandice.avg, meaniou.avg, meanmae.avg, meansm.avg


if __name__ == "__main__":
    imgs, label, paths = BUSI_DATASET(val_path)
    x_train, x_test, y_train, y_test = train_test_split(imgs, label, test_size=0.2, random_state=42)
    val_set = ImgFolder(x_test, y_test, None, val_transform, val_target_transform)
    model = Network().cuda()
    model.load_state_dict(torch.load('CFRA-best.pth'))
    val_loader = DataLoader(val_set, batch_size=8, num_workers=0, shuffle=False)

