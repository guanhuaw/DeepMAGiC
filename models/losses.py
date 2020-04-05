import torch
import torch.nn as nn
from torch.nn import init
import functools
import torch.autograd as autograd
import numpy as np
import torchvision.models as models
import util.util as util
from util.image_pool import ImagePool
###############################################################################
# Functions
###############################################################################
class PerceptualLoss():
	def contentFunc(self):
		conv_3_3_layer = 14
		cnn = models.vgg19(pretrained=True).features
		cnn = cnn.cuda()
		model = nn.Sequential()
		model = model.cuda()
		for i,layer in enumerate(list(cnn)):
			model.add_module(str(i),layer)
			if i == conv_3_3_layer:
				break
		return model

	def initialize(self, loss):
		self.criterion = loss
		self.contentFunc = self.contentFunc()

	def get_loss(self, fakeIm, realIm):
		f_fake = self.contentFunc.forward(fakeIm)
		f_real = self.contentFunc.forward(realIm)
		f_real_no_grad = f_real.detach()
		loss = self.criterion(f_fake, f_real_no_grad)
		return loss


def gradient_penalty(self, y, x):
	"""Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
	weight = torch.ones(y.size()).to(self.device)
	dydx = torch.autograd.grad(outputs=y,
							   inputs=x,
							   grad_outputs=weight,
							   retain_graph=True,
							   create_graph=True,
							   only_inputs=True)[0]

	dydx = dydx.view(dydx.size(0), -1)
	dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
	return torch.mean((dydx_l2norm - 1) ** 2)
