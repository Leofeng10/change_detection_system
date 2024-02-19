from __future__ import print_function
import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from tqdm import tqdm
from PIL import Image
from model_cae import CAE
from model_cbam import UC
from models import VanillaAE
from utils import get_dataloader, print_and_write_log, set_random_seed
import matplotlib.pyplot as pp
from skimage.metrics import structural_similarity
from sklearn.metrics import mean_squared_error
import cv2


def update(frame):
    x.append(len(x))
    y.append(frame)
    line.set_data(x, y)
    ax.relim()
    ax.autoscale_view()
    plt.pause(0.001)


x = []
y = []
fig, ax = pp.subplots()
ax.set_ylim(0, 200)
ax.set_xlim(0, 1000)
line, = ax.plot(x, y)

def main():
    parser = argparse.ArgumentParser()
    # basic
    parser.add_argument('--dataset', required=True, help='folderall | filelist | pairfilelist')
    parser.add_argument('--dataroot', default='', help='path to dataset')
    parser.add_argument('--datalist', default='', help='path to dataset file list')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=256, help='dimension of the latent layers')
    parser.add_argument('--nblk', type=int, default=2, help='number of blocks')
    parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.001')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam. default=0.999')
    parser.add_argument('--no_cuda', action='store_true', help='not enable cuda (if use CPU only)')
    parser.add_argument('--netG', default='', help='path to netG (to continue training)')
    parser.add_argument('--expf', default='./experiments',
                        help='folder to save visualized images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual random seed')

    # display and save
    parser.add_argument('--log_iter', type=int, default=50, help='log interval (iterations)')
    parser.add_argument('--visualize_iter', type=int, default=500, help='visualization interval (iterations)')
    parser.add_argument('--ckpt_save_epoch', type=int, default=1, help='checkpoint save interval (epochs)')

    # losses
    parser.add_argument('--mse_w', type=float, default=1.0, help='weight for mse (L2) spatial loss')
    parser.add_argument('--ffl_w', type=float, default=0.0, help='weight for focal frequency loss')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='the scaling factor alpha of the spectrum weight matrix for flexibility')
    parser.add_argument('--patch_factor', type=int, default=1,
                        help='the factor to crop image patches for patch-based focal frequency loss')
    parser.add_argument('--ave_spectrum', action='store_true', help='whether to use minibatch average spectrum')
    parser.add_argument('--log_matrix', action='store_true',
                        help='whether to adjust the spectrum weight matrix by logarithm')
    parser.add_argument('--batch_matrix', action='store_true',
                        help='whether to calculate the spectrum weight matrix using batch-based statistics')
    parser.add_argument('--freq_start_epoch', type=int, default=1, help='the start epoch to add focal frequency loss')

    opt = parser.parse_args()
    opt.is_train = False

    os.makedirs(os.path.join(opt.expf, 'images'), exist_ok=True)
    os.makedirs(os.path.join(opt.expf, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(opt.expf, 'logs'), exist_ok=True)
    train_log_file = os.path.join(opt.expf, 'logs', 'train_log.txt')
    opt.train_log_file = train_log_file

    cudnn.benchmark = True

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print_and_write_log(train_log_file, "Random Seed: %d" % opt.manualSeed)
    set_random_seed(opt.manualSeed)

    if torch.cuda.is_available() and opt.no_cuda:
        print_and_write_log(train_log_file,
                            "WARNING: You have a CUDA device, so you should probably run without --no_cuda")

    dataloader, nc = get_dataloader(opt)
    opt.nc = nc

    print_and_write_log(train_log_file, opt)

    model = UC(opt)

    num_epochs = opt.nepoch
    iters = 0
    scores = []
    losses = []

    for i, data in enumerate(tqdm(dataloader), 0):
        img, img_path = data
        print(img.size())

        real_cpu = img.cpu()
        print(real_cpu.size())
        recon = model.sample(real_cpu)
        visual = torch.cat([real_cpu[:16], recon.detach().cpu()[:16]], 0)
        # vutils.save_image(visual, 'C:/Users/Feng Zhunyi/Desktop/focal-frequency-loss-master/VanillaAE/results/celeba/epoch_020_seed_1112_with_input/%03d.png' % i, normalize=True,
        #                   nrow=16)
        vutils.save_image(visual,
                          'C:/Users/Feng Zhunyi/Desktop/change_detection/VanillaAE/results/imgs/%03d.png' % i,
                          normalize=True,
                          nrow=16)

        recon_img = tensor2im(recon)

        real_img = tensor2im(real_cpu)
        grey_real = cv2.cvtColor(real_img, cv2.COLOR_BGR2GRAY)
        grey_recon = cv2.cvtColor(recon_img, cv2.COLOR_BGR2GRAY)

        score, diff = structural_similarity(grey_real, grey_recon, full=True)
        loss = mean_squared_error(grey_real, grey_recon)
        scores.append(score)
        losses.append(loss)
        if loss > 90:
            print("----------",loss, score, i)

        scaled_loss = np.exp(loss / 20)
        update(scaled_loss)

    # pp.subplot(1, 2, 1)
    # pp.plot(losses, color='red')
    # pp.show()

    ax.set_xlabel("Time")
    ax.set_ylabel("Loss")
    plt.show()








def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Args:
        input_image (torch.tensor): the input tensor array.
        imtype (type): the desired type of the converted numpy image array.
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.detach()
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()
        if image_numpy.shape[0] == 1:
            # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        # post-processing: transpose and scaling
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = input_image
    return image_numpy.astype(imtype)

if __name__ == '__main__':
    main()
