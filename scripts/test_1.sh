set -ex
python test.py --how_many 10000 --which_epoch latest --contrast stir --dataroot /home/wanggh/magic/magic_paper --name onecontrastunetstir --which_model_netG unet_256 --suffix 1 --model magic_basic --dataset_mode magic --gpu_ids 0
