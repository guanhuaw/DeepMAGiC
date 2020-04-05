from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os

# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] > 3:
        image_numpy = np.tile(image_numpy[0,:,:], (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = np.transpose(image_numpy, (1, 2, 0))* 255.0
        # print('sdfsdfsdf')
        # print(image_numpy.shape)
    if image_numpy.shape[0] == 3:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    if image_numpy.shape[0] == 2:
        img = np.sqrt(np.square(image_numpy[1,:,:])+np.square(image_numpy[0,:,:]))
        upp = np.percentile(img, 99.9)
        lowp = np.percentile(img, 0.1)
        img = np.clip(img,a_min=lowp,a_max=upp)
        image_numpy = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        image_numpy = image_numpy/(np.amax(image_numpy)+0.0000001)*255.0


    return image_numpy.astype(imtype)

def tensor2imdiff(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))+1)/2 * 255.0
    return image_numpy.astype(imtype)

def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_phase(batch_input):
    temp = batch_input[:,2,:,:]/batch_input[:,1,:,:]
    return temp.unsqueeze(1)

def get_amp(batch_input):
    return torch.sqrt(torch.pow(batch_input[:,2,:,:],2) + torch.pow(batch_input[:,1,:,:],2))

def generate_mask_alpha(size=[320,320], r_factor_designed=3.0, r_alpha=3, axis_undersample=1,
                        acs=3, seed=0, mute=0):
    # init
    mask = np.zeros(size)
    if seed>=0:
        np.random.seed(seed)
    # get samples
    num_phase_encode = size[axis_undersample]
    num_phase_sampled = int(np.floor(num_phase_encode/r_factor_designed))
    # coordinate
    coordinate_normalized = np.array(range(num_phase_encode))
    coordinate_normalized = np.abs(coordinate_normalized-num_phase_encode/2)/(num_phase_encode/2.0)
    prob_sample = coordinate_normalized**r_alpha
    prob_sample = prob_sample/sum(prob_sample)
    # sample
    index_sample = np.random.choice(num_phase_encode, size=num_phase_sampled,
                                    replace=False, p=prob_sample)
    # sample
    if axis_undersample == 0:
        mask[index_sample,:]=1
    else:
        mask[:,index_sample]=1
    mask_temp = np.zeros_like(mask)
    # acs
    if axis_undersample == 0:
        mask[:(acs+1)//2,:]=1
        mask[-acs//2:,:]=1
    else:
        mask[:,:(acs+1)//2]=1
        mask[:,-acs//2:]=1
    # compute reduction
    r_factor = len(mask.flatten())/sum(mask.flatten())
    if not mute:
        print('gen mask size of {1} for R-factor={0:.4f}'.format(r_factor, mask.shape))
        print(num_phase_encode, num_phase_sampled, np.where(mask[0,:]))

    return mask, r_factor

def generate_mask_beta(size=[320,320], r_factor_designed=3.0, axis_undersample=1,
                        acs=8, mute=0):
    # init
    mask = np.zeros(size)
    index_sample = range(0, size[0], int(r_factor_designed))
    # sample
    if axis_undersample == 0:
        mask[index_sample,:]=1
    else:
        mask[:,index_sample]=1
    mask_temp = np.zeros_like(mask)
    # acs
    if axis_undersample == 0:
        mask[:(acs+1)//2,:]=1
        mask[-acs//2:,:]=1
        mask_temp[size[1]//2:,:] = mask[:size[1]//2,:]
        mask_temp[:size[1]//2,:] = mask[size[1]//2:,:]
    else:
        mask[:,:(acs+1)//2]=1
        mask[:,-acs//2:]=1
        mask_temp[:,size[1]//2:] = mask[:,:size[1]//2]
        mask_temp[:,:size[1]//2] = mask[:,size[1]//2:]
    # compute reduction
    r_factor = len(mask.flatten())/sum(mask.flatten())
#    if not mute:
#        print('gen mask size of {1} for R-factor={0:.4f}'.format(r_factor, mask.shape))
#        print(num_phase_encode, num_phase_sampled, np.where(mask[0,:]))

    return mask_temp, r_factor

def complex_matmul(a,b):
    # function to multiply two complex variable in pytorch, the real/imag channel are in the last two channels.
    return a[...,0]*b[...,0]-a[...,1]*b[...,1]