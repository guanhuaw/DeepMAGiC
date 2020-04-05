import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import scipy.io as sp
import scipy.ndimage as image
import numpy as np
import torch
from util.util import generate_mask_alpha, generate_mask_beta


class magicdataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        # pay attention to the format of input directory
        self.dir_A = os.path.join(opt.dataroot, opt.phase)
        self.A_paths = make_dataset(self.dir_A)
        self.A_paths = sorted(self.A_paths)
        self.A_size = len(self.A_paths)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        Atemp = np.load(A_path)
        magic = ((Atemp['magic'] / 255) - 0.5) * 2
        t2fse = Atemp['t2fse'] / 255
        t2flair = Atemp['t2flair'] / 255
        t1fse = Atemp['t1fse'] / 255
        t1flair = Atemp['t1flair'] / 255
        pdfse = Atemp['pdfse'] / 255
        stir = Atemp['stir'] / 255

        kkkk = t2flair + t1fse + t2fse + t1flair + stir + pdfse + np.sum(magic[0:3:, :, :], 0)
        mask1 = kkkk > 0.008
        for jj in range(0,24,3):
            magic[jj,:,:] = magic[jj,:,:]/2 + 0.5
        # kkkk = t2flair+t1fse+t2fse+t1flair+stir+pdfse
        # mask = kkkk>0.9

        mask = image.morphology.binary_fill_holes(mask1)
        t1fse = t1fse * mask
        t2fse = t2fse * mask
        t2flair = t2flair * mask
        t1flair = t1flair * mask
        stir = stir * mask
        pdfse = pdfse * mask
        for jj in range(0, 24):
            magic[jj, :, :] = magic[jj, :, :] * mask

        if self.opt.isTrain:

            magic = magic + np.random.normal(loc=0.0, scale=0.008 * np.random.rand(), size=magic.shape)

            if np.random.rand() > 0.5:
                magic = np.flip(magic, 1)
                t2fse = np.flip(t2fse, 0)
                t2flair = np.flip(t2flair, 0)
                t1fse = np.flip(t1fse, 0)
                t1flair = np.flip(t1flair, 0)
                pdfse = np.flip(pdfse, 0)
                stir = np.flip(stir, 0)
            if np.random.rand() > 0.5:
                magic = np.flip(magic, 2)
                t2fse = np.flip(t2fse, 1)
                t2flair = np.flip(t2flair, 1)
                t1fse = np.flip(t1fse, 1)
                t1flair = np.flip(t1flair, 1)
                pdfse = np.flip(pdfse, 1)
                stir = np.flip(stir, 1)
            shiftx = np.random.randint(-150, 500)
            shifty = np.random.randint(-150, 500)
            rot = np.random.randint(0, 180)
            magic = np.roll(magic, shiftx, 2)
            t2fse = np.roll(t2fse, shiftx, 1)
            t2flair = np.roll(t2flair, shiftx, 1)
            t1fse = np.roll(t1fse, shiftx, 1)
            t1flair = np.roll(t1flair, shiftx, 1)
            pdfse = np.roll(pdfse, shiftx, 1)
            stir = np.roll(stir, shiftx, 1)
            magic = np.roll(magic, shifty, 1)
            t2fse = np.roll(t2fse, shifty, 0)
            t2flair = np.roll(t2flair, shifty, 0)
            t1fse = np.roll(t1fse, shifty, 0)
            t1flair = np.roll(t1flair, shifty, 0)
            pdfse = np.roll(pdfse, shifty, 0)
            stir = np.roll(stir, shifty, 0)
            # if np.random.rand() > 0.5:
            #     for ii in range(24):
            #         magic[ii,:,:] = np.rot90(magic[ii,:,:])
            #     t2fse = np.rot90(t2fse)
            #     t2flair = np.rot90(t2flair)
            #     t1fse = np.rot90(t1fse)
            #     t1flair = np.rot90(t1flair)
            #     pdfse = np.rot90(pdfse)
            #     stir = np.rot90(stir)
            # print(t2flair.shape)
            # print()


        #     t2fse = image.interpolation.rotate(t2fse, rot, reshape=False)
        #     t1fse = image.interpolation.rotate(t1fse, rot, reshape=False)
        #     pdfse = image.interpolation.rotate(pdfse, rot, reshape=False)
        #     stir = image.interpolation.rotate(stir, rot, reshape=False)
        #     t1flair = image.interpolation.rotate(t1flair, rot, reshape=False)
        #     t2flair = image.interpolation.rotate(t2flair, rot, reshape=False)
        #     for ii in range(24):
        #         magic[ii,:,:] = image.interpolation.rotate(magic[ii,:,:], rot, reshape=False)
        magic = torch.tensor(magic, dtype=torch.float32)
        t2flair = torch.unsqueeze(torch.tensor(t2flair, dtype=torch.float32), 0)
        t2fse = torch.unsqueeze(torch.tensor(t2fse, dtype=torch.float32), 0)
        t1fse = torch.unsqueeze(torch.tensor(t1fse, dtype=torch.float32), 0)
        t1flair = torch.unsqueeze(torch.tensor(t1flair, dtype=torch.float32), 0)
        pdfse = torch.unsqueeze(torch.tensor(pdfse, dtype=torch.float32), 0)
        stir = torch.unsqueeze(torch.tensor(stir, dtype=torch.float32), 0)
        return {'magic': magic, 't2fse': t2fse, 't2flair': t2flair, 't1fse': t1fse, 't1flair': t1flair, 'pdfse': pdfse,
                'stir': stir, 'path': A_path}

    def __len__(self):
        return self.A_size

    def name(self):
        return 'MagicDataset'

#  generate mask based on alpha
