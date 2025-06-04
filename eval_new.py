import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import os
import sys
os.chdir('../')
sys.path.append('./')
os.getcwd()
from model import DeepJSCC
import matplotlib.pyplot as plt
import matplotlib
from utils import get_psnr, image_normalization

channel = 'AWGN'
saved = '/home/chun/Deep-JSCC-PyTorch/out/checkpoints/CIFAR10_20_19.0_0.17_AWGN_13h21m38s_on_Jun_02_2024/epoch_999.pkl'

snr = 2000
test_image_dir = './demo/kodim08.png'
times = 10

