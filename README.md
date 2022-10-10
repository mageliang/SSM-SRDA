# SSM-SRDA: 结合超分辨率和域适应的遥感图像语义分割方法
Pytorch implementation of our method for adapting semantic segmentation 
from the low-resolution remote sensing dataset (source domain) to the high-resolution remote sensing dataset.


## Paper

Please cite our paper if you find it useful for your research.


## Installation
* Install Pytorch 1.3.0 from http://pytorch.org with python 3.6 and CUDA 10.1


## Dataset
* Download the [Massachusetts Buildings Dataset](https://www.cs.toronto.edu/~vmnih/data/) 
 Training Set as the source domain, and put it `./datasets` folder
 
 * Download the [Inria Aerial Image Labeling Dataset](https://project.inria.fr/aerialimagelabeling/)
 as the target domain, and put it to `./datasets` folder
 
 * Create the Mass-Inria dataset
 
## Acknowledgment
This code is heavily borrowed from [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)



###############################################
## 训练
python train.py --name your_file_name --dataroot ./datasets/mass_inria  --model ssmsrdanet_fa --num_classes 2 --dataset_mode srdanet --A_crop_size 114 --B_crop_size 380 --batch_size 2

## 断点训练
python train.py --name your_file_name --dataroot ./datasets/mass_inria --model ssmsrdanet_fa --num_classes 2 --A_crop_size 114 --B_crop_size 380 --batch_size 10 --dataset_mode srdanet --continue_train  --gpu_ids 0,1

########################################
## 测试

python val_manyclass.py --name your_file_name --dataroot ./datasets/mass_inria --model ssmsrdanet_fa --num_classes 2 --dataset_mode srdanetval --resize_size 380


####################################
## 输出结果图
python output_val_result_images.py --name your_file_name --dataroot ./datasets/mass_inria --model ssmsrdanet_fa --num_classes 2 --dataset_mode srdanetval --resize_size 128 --gpu_ids 0


