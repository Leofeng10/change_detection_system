import os

import torch
import torch.nn as nn
import torch.optim as optim

from network_trans import Trans
from network_cae import Autoencoder
from focal_frequency_loss import FocalFrequencyLoss as FFL

from networks import MLP
from utils import print_and_write_log, weights_init


class VisionTransformer(nn.Module):
    def __init__(self, opt):
        super(VisionTransformer, self).__init__()
        self.opt = opt
        self.device = torch.device("cuda:0" if not opt.no_cuda else "cpu")
        self.frame = 4
        # generator
        self.netG = Trans(num_frames=self.frame).to(self.device)
        weights_init(self.netG)
        if opt.netG != '':
            self.netG.load_state_dict(torch.load(opt.netG, map_location=self.device))
        print_and_write_log(opt.train_log_file, 'netG:')
        print_and_write_log(opt.train_log_file, str(self.netG))

        # losses
        self.criterion = nn.MSELoss()
        # define focal frequency loss
        self.criterion_freq = FFL(loss_weight=opt.ffl_w,
                                  alpha=opt.alpha,
                                  patch_factor=opt.patch_factor,
                                  ave_spectrum=opt.ave_spectrum,
                                  log_matrix=opt.log_matrix,
                                  batch_matrix=opt.batch_matrix).to(self.device)

        # misc
        self.to(self.device)

        # optimizer
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

    def forward(self):
        pass

    def gen_update(self, data, epoch, matrix=None):

        self.netG.zero_grad()
        real = data.to(self.device)
        if matrix is not None:
            matrix = matrix.to(self.device)
        if self.frame == 0:
            recon = self.netG(real)
        else:
            recon = self.netG(real[:,:self.frame])


        # print("real size-----------", real.size())
        #
        # print("recon size-----------", recon.size())

        # apply pixel-level loss
        errG_pix = self.criterion(recon, real[:, self.frame]) * self.opt.mse_w
        errG_pix = errG_pix.float().cuda()
        # apply focal frequency loss
        if epoch >= self.opt.freq_start_epoch:
            errG_freq = self.criterion_freq(recon, real[:, self.frame], matrix)
        else:
            errG_freq = torch.tensor(0.0).to(self.device)

        errG_pix = errG_freq.float().cuda()

        errG = errG_pix + errG_freq

        errG = errG.float().cuda()

        errG.backward()
        self.optimizerG.step()

        return errG_pix, errG_freq

    def sample(self, x):
        x = x.to(self.device)
        self.netG.eval()
        with torch.no_grad():
            recon = self.netG(x)
        self.netG.train()

        return recon

    def save_checkpoints(self, ckpt_dir, epoch):
        torch.save(self.netG.state_dict(), '%s/netG_epoch_%03d.pth' % (ckpt_dir, epoch))
