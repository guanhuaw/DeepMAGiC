import torch
import numpy as np
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import losses
from util.metrics import PSNR
import pytorch_msssim


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
                self.loss_names = ['G_I_L1_t1fse', 'G_I_L2_t1fse', 'SSIM_t1fse', 'PSNR_t1fse', \
                                   'G_I_L1_t2fse', 'G_I_L2_t2fse', 'SSIM_t2fse', 'PSNR_t2fse', \
                                   'G_I_L1_t1flair', 'G_I_L2_t1flair', 'SSIM_t1flair', 'PSNR_t1flair', \
                                   'G_I_L1_t2flair', 'G_I_L2_t2flair', 'SSIM_t2flair', 'PSNR_t2flair', \
                                   'G_I_L1_pdfse', 'G_I_L2_pdfse', 'SSIM_pdfse', 'PSNR_pdfse', \
                                   'G_I_L1_stir', 'G_I_L2_stir', 'SSIM_stir', 'PSNR_stir', \
                                   'G_I_L1', 'G_I_L2', 'SSIM', 'PSNR']
            else:
                self.model_names = ['G, D']
                self.loss_names = ['G_GAN', 'G_I_L1', 'G_I_L2', 'D_GAN_fake', 'D_GAN_real', 'SSIM', 'PSNR']
        else:  # during test time, only load Gs
            self.model_names = ['G']
            self.loss_names = ['G_I_L1', 'G_I_L2', 'SSIM', 'PSNR']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals

        self.visual_names = ['real_A', 'fake_t2fse', 'real_t2fse', 'fake_t1fse', 'real_t1fse', \
                             'fake_t1flair', 'real_t1flair', 'fake_t2flair', 'real_t2flair', 'fake_pdfse', 'real_pdfse',
                             'fake_stir',
                             'real_stir']

        self.netG = networks.define_G(self.opt, opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain,
                                      self.gpu_ids)
        self.criterionL1 = torch.nn.L1Loss()
        self.criterionMSE = torch.nn.MSELoss()
        self.ssim_loss = pytorch_msssim.SSIM(val_range=1)
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
                                              opt.n_layers_D_I, opt.norm, use_sigmoid, opt.init_type, opt.init_gain,
                                              self.gpu_ids)
            if opt.disc_model == 'traditional':
                self.netD = networks.define_D(1, opt.ndf, opt.which_model_netD,
                                              opt.n_layers_D_I, opt.norm, use_sigmoid, opt.init_type, opt.init_gain,
                                              self.gpu_ids)

            self.loss_wgan_gp = opt.loss_wgan_gp
            self.fake_pool = ImagePool(opt.pool_size)
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, use_l1=not opt.no_l1gan).to(self.device)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        self.real_A = input['magic'].to(self.device)
        self.label_size = self.real_A.shape[0]
        self.label_t1fse = self.label2onehot(self.label_size, 0).to(self.device)
        self.label_t2fse = self.label2onehot(self.label_size, 1).to(self.device)
        self.label_t1flair = self.label2onehot(self.label_size, 2).to(self.device)
        self.label_t2flair = self.label2onehot(self.label_size, 3).to(self.device)
        self.label_pdfse = self.label2onehot(self.label_size, 4).to(self.device)
        self.label_stir = self.label2onehot(self.label_size, 5).to(self.device)
        self.real_t1fse = input['t1fse'].to(self.device)
        self.real_t2fse = input['t2fse'].to(self.device)
        self.real_t1flair = input['t1flair'].to(self.device)
        self.real_t2flair = input['t2flair'].to(self.device)
        self.real_pdfse = input['pdfse'].to(self.device)
        self.real_stir = input['stir'].to(self.device)
        self.image_paths = input['path']

    def forward(self):
        self.fake_t1fse = self.netG(self.real_A, self.label_t1fse)
        self.fake_t2fse = self.netG(self.real_A, self.label_t2fse)
        self.fake_t1flair = self.netG(self.real_A, self.label_t1flair)
        self.fake_t2flair = self.netG(self.real_A, self.label_t2flair)
        self.fake_pdfse = self.netG(self.real_A, self.label_pdfse)
        self.fake_stir = self.netG(self.real_A, self.label_stir)
        self.loss_PSNR_t1fse = PSNR(self.real_t1fse, self.fake_t1fse)
        self.loss_PSNR_t2fse = PSNR(self.real_t2fse, self.fake_t2fse)
        self.loss_PSNR_t1flair = PSNR(self.real_t1flair, self.fake_t1flair)
        self.loss_PSNR_t2flair = PSNR(self.real_t2flair, self.fake_t2flair)
        self.loss_PSNR_pdfse = PSNR(self.real_pdfse, self.fake_pdfse)
        self.loss_PSNR_stir = PSNR(self.real_stir, self.fake_stir)
        self.loss_PSNR = self.loss_PSNR_t1fse + self.loss_PSNR_t2fse + self.loss_PSNR_t1flair + self.loss_PSNR_t2flair + self.loss_PSNR_pdfse + self.loss_PSNR_stir

    def backward_D(self):
        if self.disc_model == 'pix2pix':
            fake_AB = self.fake_pool.query(torch.cat((self.real_A, self.fake_B), 1))
            pred_fake = self.netD(fake_AB.detach())
            real_AB = torch.cat((self.real_A, self.real_B), 1)
            pred_real = self.netD(real_AB)

        if self.disc_model == 'traditional':
            fake_AB = self.fake_pool.query(self.fake_B)
            pred_real = self.netD(self.real_B)
            pred_fake = self.netD(fake_AB.detach())
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
            self.loss_D_GAN_fake = self.criterionGAN(pred_fake, False)
            self.loss_D_GAN_real = self.criterionGAN(pred_real, True)
        self.loss_D_GAN = 0.5 * (self.loss_D_GAN_fake + self.loss_D_GAN_real)
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
                pred_fake = self.netD(self.fakeB)
            if self.no_wgan == False:
                self.loss_G_GAN = -pred_fake.mean()
            elif self.no_wgan_gp == False:
                self.loss_G_GAN = -pred_fake.mean()
            else:
                self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        else:
            self.loss_G_GAN = 0

        # Second, G(A) = B
        self.loss_G_I_L1_t1fse = self.criterionL1(self.fake_t1fse, self.real_t1fse) * self.opt.loss_content_I_l1
        self.loss_G_I_L2_t1fse = self.criterionMSE(self.fake_t1fse, self.real_t1fse) * self.opt.loss_content_I_l2
        self.loss_G_I_L1_t2fse = self.criterionL1(self.fake_t2fse, self.real_t2fse) * self.opt.loss_content_I_l1
        self.loss_G_I_L2_t2fse = self.criterionMSE(self.fake_t2fse, self.real_t2fse) * self.opt.loss_content_I_l2
        self.loss_G_I_L1_t1flair = self.criterionL1(self.fake_t1flair, self.real_t1flair) * self.opt.loss_content_I_l1
        self.loss_G_I_L2_t1flair = self.criterionMSE(self.fake_t1flair, self.real_t1flair) * self.opt.loss_content_I_l2
        self.loss_G_I_L1_t2flair = self.criterionL1(self.fake_t2flair, self.real_t2flair) * self.opt.loss_content_I_l1
        self.loss_G_I_L2_t2flair = self.criterionMSE(self.fake_t2flair, self.real_t2flair) * self.opt.loss_content_I_l2
        self.loss_G_I_L1_pdfse = self.criterionL1(self.fake_pdfse, self.real_pdfse) * self.opt.loss_content_I_l1
        self.loss_G_I_L2_pdfse = self.criterionMSE(self.fake_pdfse, self.real_pdfse) * self.opt.loss_content_I_l2
        self.loss_G_I_L1_stir = self.criterionL1(self.fake_stir, self.real_stir) * self.opt.loss_content_I_l1
        self.loss_G_I_L2_stir = self.criterionMSE(self.fake_stir, self.real_stir) * self.opt.loss_content_I_l2
        self.loss_G_I_L1 = self.loss_G_I_L1_t1fse + self.loss_G_I_L1_t2fse + self.loss_G_I_L1_t1flair + self.loss_G_I_L1_t2flair + self.loss_G_I_L1_pdfse + self.loss_G_I_L1_stir
        self.loss_G_I_L2 = self.loss_G_I_L2_t1fse + self.loss_G_I_L2_t2fse + self.loss_G_I_L2_t1flair + self.loss_G_I_L2_t2flair + self.loss_G_I_L2_pdfse + self.loss_G_I_L2_stir
        self.loss_G_CON_I = self.loss_G_I_L1 + self.loss_G_I_L2
        self.loss_SSIM_t1fse = self.ssim_loss(self.real_t1fse.repeat(1, 3, 1, 1), self.fake_t1fse.repeat(1, 3, 1, 1))
        self.loss_SSIM_t2fse = self.ssim_loss(self.real_t2fse.repeat(1, 3, 1, 1), self.fake_t2fse.repeat(1, 3, 1, 1))
        self.loss_SSIM_t1flair = self.ssim_loss(self.real_t1flair.repeat(1, 3, 1, 1),
                                                self.fake_t1flair.repeat(1, 3, 1, 1))
        self.loss_SSIM_t2flair = self.ssim_loss(self.real_t2flair.repeat(1, 3, 1, 1),
                                                self.fake_t2flair.repeat(1, 3, 1, 1))
        self.loss_SSIM_pdfse = self.ssim_loss(self.real_pdfse.repeat(1, 3, 1, 1), self.fake_pdfse.repeat(1, 3, 1, 1))
        self.loss_SSIM_stir = self.ssim_loss(self.real_stir.repeat(1, 3, 1, 1), self.fake_stir.repeat(1, 3, 1, 1))
        self.loss_SSIM = self.loss_SSIM_t1fse + self.loss_SSIM_t2fse + self.loss_SSIM_t1flair + self.loss_SSIM_t2flair + self.loss_SSIM_pdfse + self.loss_SSIM_stir

        self.loss_G = self.loss_G_CON_I + self.loss_G_GAN - self.loss_SSIM * self.opt.loss_ssim
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
