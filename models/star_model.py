import torch
import numpy as np
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import losses
from util.metrics import PSNR
import pytorch_msssim
import random
import torch.nn.functional as F

class StarModel(BaseModel):
    def name(self):
        return 'STARModel'

    def label2onehot(self, batch_size, labels):
        """Convert label indices to one-hot vectors."""
        dim = 6
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels] = 1
        return out

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        if self.isTrain:
            if self.train_phase == 'generator':
                self.model_names = ['G']
                self.loss_names = ['G_I_L1', 'G_I_L2', 'SSIM', 'PSNR']
            else:
                self.model_names = ['G', 'D']
                self.loss_names = ['G_GAN_label', 'G_GAN_kind','G_I_L1', 'G_I_L2', 'D_GAN_label', 'D_GAN_kind', 'SSIM', 'PSNR']
                if opt.use_vgg:
                    self.loss_names += ['vgg']
        else:  # during test time, only load Gs
            self.model_names = ['G']
            self.loss_names = ['SSIM', 'PSNR']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals

        self.visual_names = ['real_A', 'fake_B', 'real_B']

        self.netG = networks.define_G(self.opt, opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain,
                                      self.gpu_ids)
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionMSE = torch.nn.MSELoss()
        if opt.use_vgg:
            self.perceptual = losses.PerceptualLoss()
            self.perceptual.initialize(self.criterionMSE)

        self.ssim_loss = pytorch_msssim.SSIM(val_range=1)
        if self.isTrain:
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

        if self.isTrain and self.train_phase == 'together':
            self.no_wgan = opt.no_wgan
            self.no_wgan_gp = opt.no_wgan_gp
            if self.no_wgan_gp == False:
                self.disc_step = opt.disc_step
            else:
                self.disc_step = 1
            self.disc_model = opt.disc_model
            use_sigmoid = opt.no_lsgan
            if opt.disc_model == 'pix2pix':
                self.netD = networks.define_D(opt.input_nc + 1, opt.ndf,
                                              opt.which_model_netD,
                                              opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain,
                                              self.gpu_ids)
            if opt.disc_model == 'traditional':
                self.netD = networks.define_D(self.opt, 1, opt.ndf, opt.which_model_netD,
                                              opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain,
                                              self.gpu_ids)

            self.loss_wgan_gp = opt.loss_wgan_gp
            self.fake_pool = ImagePool(opt.pool_size)
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, use_l1=not opt.no_l1gan).to(self.device)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input, idx):
        self.namelist = ['t1fse', 't2fse', 't1flair', 't2flair', 'pdfse', 'stir']
        self.real_A = input['magic'].to(self.device)
        if self.opt.isTrain:
            self.idx = random.randrange(6)
        else:
            self.idx = idx
        #self.idx = idx
        self.label = self.namelist[self.idx]
        self.label_size = self.real_A.shape[0]
        self.cls = torch.tensor(np.ones(self.label_size)*self.idx, dtype=torch.int64).to(self.device)
        self.label_channel = self.label2onehot(self.label_size, self.idx).to(self.device)
        self.real_B = input[self.label].to(self.device)
        self.image_paths = input['path']

    def forward(self):
        self.fake_B = self.netG(self.real_A, self.label_channel)
        self.loss_PSNR = PSNR(self.real_B, self.fake_B)
        self.loss_SSIM = self.ssim_loss(self.real_B.repeat(1, 3, 1, 1), self.fake_B.repeat(1, 3, 1, 1))

    def backward_D(self):
        if self.disc_model == 'pix2pix':
            fake_AB = self.fake_pool.query(torch.cat((self.real_A, self.fake_B), 1))
            pred_fake = self.netD(fake_AB.detach())
            real_AB = torch.cat((self.real_A, self.real_B), 1)
            pred_real = self.netD(real_AB)

        if self.disc_model == 'traditional':
            fake_AB = self.fake_pool.query(self.fake_B)
            pred_real_label, pred_real_kind = self.netD(self.real_B)
            pred_fake_label, pred_fake_kind = self.netD(fake_AB.detach())
        if self.no_wgan == False:
            self.loss_D_GAN_fake = pred_fake.mean()
            self.loss_D_GAN_real = -pred_real.mean()
        elif self.no_wgan_gp == False:
            self.loss_D_GAN_real = -pred_fake.mean()
            self.loss_D_GAN_fake = pred_fake.mean()
            alpha = torch.rand(self.kreal.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * self.kreal.data + (1 - alpha) * self.kfake.data).requires_grad_(True)
            out_src = self.netD(x_hat)
            self.d_loss_gp = losses.gradient_penalty(out_src, x_hat) * self.loss_wgan_gp
        else:
            self.loss_D_GAN_fake_label = self.criterionGAN(pred_fake_label, False)
            self.loss_D_GAN_real_label = self.criterionGAN(pred_real_label, True)
            self.loss_D_GAN_label = 0.5*(self.loss_D_GAN_fake_label+self.loss_D_GAN_real_label)*self.opt.gamma
            self.loss_D_GAN_kind = F.cross_entropy(pred_real_kind, self.cls)*self.opt.kind
        self.loss_D_GAN = (self.loss_D_GAN_label + self.loss_D_GAN_kind)*self.opt.beta*self.opt.loss_GAN
        if self.no_wgan_gp == False:
            self.loss_D_GAN = self.loss_D_GAN + self.d_loss_gp

        self.loss_D_GAN.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        if self.isTrain and self.train_phase == 'together':
            if self.disc_model == 'pix2pix':
                fake_AB = torch.cat((self.realA, self.fakeB), 1)
                pred_fake = self.netD(fake_AB)
            if self.disc_model == 'traditional':
                pred_fake_label, pred_fake_kind = self.netD(self.fake_B)
            if self.no_wgan == False:
                self.loss_G_GAN = -pred_fake.mean()
            elif self.no_wgan_gp == False:
                self.loss_G_GAN = -pred_fake.mean()
            else:
                self.loss_G_GAN_label = self.criterionGAN(pred_fake_label, True)*self.opt.gamma
                self.loss_G_GAN_kind = F.cross_entropy(pred_fake_kind, self.cls)*self.opt.kind
                self.loss_G_GAN = self.loss_G_GAN_label + self.loss_G_GAN_kind
        else:
            self.loss_G_GAN = 0
        self.loss_G_GAN = self.loss_G_GAN * self.opt.loss_GAN

        # Second, G(A) = B
        self.loss_G_I_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.loss_content_I_l1
        self.loss_G_I_L2 = self.criterionMSE(self.fake_B, self.real_B) * self.opt.loss_content_I_l2
        self.loss_G_CON_I = self.loss_G_I_L1 + self.loss_G_I_L2

        self.loss_G = self.loss_G_CON_I + self.loss_G_GAN - self.loss_SSIM * self.opt.loss_ssim
        if self.opt.use_vgg:
            self.loss_vgg = self.perceptual.get_loss(self.fake_B.repeat(1,3,1,1),self.real_B.repeat(1,3,1,1))*self.opt.loss_vgg
            self.loss_G = self.loss_G + self.loss_vgg
        self.loss_G.backward()

    def optimize_parameters(self):
        if self.isTrain and self.train_phase == 'together':
            self.forward()
            self.set_requires_grad(self.netD, True)
            for iter_d in range(self.disc_step):
                self.optimizer_D.zero_grad()
                self.backward_D()
                self.optimizer_D.step()
            self.set_requires_grad(self.netD, False)
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()
        else:
            self.forward()
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()
