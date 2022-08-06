import argparse
import os
import random
import numpy as np
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

from pretrained_models_pytorch import pretrainedmodels

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int,
                    help='number of data loading workers', default=2)
parser.add_argument('--epochs', type=int, default=20,
                    help='number of epochs to train for')
parser.add_argument('--cuda', action='store_true', help='enables cuda')

parser.add_argument('--target', type=int, default=859,
                    help='The target class: 859 == toaster')
parser.add_argument('--conf_target', type=float, default=0.9,
                    help='Stop attack on image when target classifier reaches this value for target class')

parser.add_argument('--max_count', type=int, default=1000,
                    help='max number of iterations to find adversarial example')
parser.add_argument('--patch_type', type=str, default='circle',
                    help='patch type: circle or square')
parser.add_argument('--patch_size', type=float, default=0.05,
                    help='patch size. E.g. 0.05 ~= 5% of image ')

parser.add_argument('--train_size', type=int, default=2000,
                    help='Number of training images')
parser.add_argument('--test_size', type=int, default=2000,
                    help='Number of test images')

parser.add_argument('--image_size', type=int, default=299,
                    help='the height / width of the input image to network')

parser.add_argument('--plot_all', type=int, default=1,
                    help='1 == plot all successful adversarial images')

parser.add_argument('--netClassifier', default='inceptionv3',
                    help="The target classifier")

parser.add_argument('--outf', default='./logs',
                    help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, default=1338, help='manual seed')

opt = parser.parse_args()
print(opt)
warnings.simplefilter("ignore")

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
device = torch.device('mps')

# if opt.cuda:
#    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

target = opt.target
conf_target = opt.conf_target
max_count = opt.max_count
patch_type = opt.patch_type
patch_size = opt.patch_size
image_size = opt.image_size
train_size = opt.train_size
test_size = opt.test_size
plot_all = opt.plot_all

assert train_size + \
    test_size <= 50000, "Traing set size + Test set size > Total dataset size"

print("=> creating model ")
netClassifier = pretrainedmodels.__dict__[opt.netClassifier](
    num_classes=1000, pretrained='imagenet')
if opt.cuda:
    netClassifier.to(device)
    # netClassifier.cuda()


print('==> Preparing data..')
normalize = transforms.Normalize(mean=netClassifier.mean,
                                 std=netClassifier.std)
idx = np.arange(50000)
np.random.shuffle(idx)
training_idx = idx[:train_size]
test_idx = idx[train_size:(test_size+train_size)]


test_loader = torch.utils.data.DataLoader(
    dset.ImageFolder('./log_8', transforms.Compose([
        transforms.Resize(round(max(netClassifier.input_size)*1.050)),
        transforms.CenterCrop(max(netClassifier.input_size)),
        transforms.ToTensor(),
        ToSpaceBGR(netClassifier.input_space == 'BGR'),
        ToRange255(max(netClassifier.input_range) == 255),
        normalize,
    ])),
    batch_size=1, shuffle=False, sampler=SubsetRandomSampler(test_idx),
    num_workers=opt.workers, pin_memory=True)

if __name__ == '__main__':
    for batch_idx, (data, labels) in enumerate(test_loader):
        data, labels = Variable(data), Variable(labels)
        prediction = netClassifier(data)
        print(prediction.data.max(1)[1][0])
        if batch_idx == 0:
            break
