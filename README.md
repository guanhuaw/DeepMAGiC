# Multi-task GAN for Multi-contrast MRI synthesis

## Introduction

This project aims to generate multiple conventional clinical MRI contrasts from multi-echo input (MDME), using a multi-task conditional GAN.

The submission, titled 'Synthesize High-quality Multi-contrast Magnetic Resonance Imaging from Multi-echo Acquisition Using Multi-task Deep Generative Model' has been accepted by IEEE Transactions on Medical Imaging (TMI).

## Dependencies

The code should work for environment with PyTorch 0.4+. A tested conda environment file has been added (not minimum).

## Usage

The code structure inherits [CycleGAN and pix2pix in PyTorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), which is highly modular and use script to pass arguments. You can find more detailed usage guidance and Q&A there.

### Data-loader

In the sample magic_dataset.py, we assume that the data pair is stored in .npz archives. However, it is simple to define your own dataloader.

### Network

You can use or define your network in networks.py. Several common baseline network has been provided. The magic_basic_model provides implementation for the single-contrast GAN or U-Net model. The star_model is the proposed multi-task learning.

### Visualizer

The package use Visdom as the visualizer. You can define a port with command --display_port (default is 8097, as in basic_options.py). Then you could run 'visdom --port xxxx' to start the visdom server (on local or remote server).

### Passing Variables

The code use parser to pass arguments. Definition are in the options folder. Two scripts, for U-Net and Multi-task GAN is provided. Some crucial arguments are listed below:

- --loss_xxxx: control the weighting of loss functions. Beta, Gamma variables are also designed to control different GAN losses.
- --c_dim: the depth of mask vectors, which should be identical to the number of destination contrasts (in this case is 6)
- --input_nc/output_nc: numbers of input and output channels. In this task, the num. of input channels is 24 (3(real, imag, magnitude) * 8 contrasts). The output is single contrast-weighted image, thus single channel.
- --contrast: which contrast (destination modality) to train for single contrast synthesis

## References and Acknowledgments

The framework is built upon Jun-Yan Zhu and Taesung Park's excellent [CycleGAN and pix2pix in PyTorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) toolbox, which provides many useful features.

We used the [pytorch-msssim](https://github.com/jorge-pessoa/pytorch-msssim) to calculate the SSIM index.

The dual-task discriminator adopts the idea from [StarGAN](https://github.com/yunjey/StarGAN).