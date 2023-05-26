## DuDGAN: Improving Class-Conditional GANs via Dual-Diffusion  - Pytorch Implementation

<img src="https://github.com/taesunyeom/DuDGAN/assets/102474982/7c7055e5-bc7a-4397-b5bb-d77467a67af6"/>


### Requirement
1. one or more CUDA-available GPUs 

2. Python 3.7.x ~ 3.9.x

3. Pytorch 1.8.0+cu111 or version that fit to your enviornment. (We do not sure it works on recent version. We recommend install previous version of Pytorch in https://pytorch.org/get-started/previous-versions/)

Libraries : 
```
  pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3 
```

### Preprocessing dataset

For class-conditional generation, you must need '.json' file which describes the discrete class value for your dataset.
Your dataset configuration must be like,
```
 trainining_dataset_folder
    └── AAA (class name)
           ├── aaa.jpg or .png (image) 
          ...
```
You should make '.json' file by enumerating images and discrete class labels.
```
 trainining_dataset_folder
    └── AAA (class name)
    └── BBB
    └── CCC
    ...
    └── '***.json' --> ["training_dataset_folder/AAA/aaa.jpg", #(class number)], ...
```

### Training Process
For class-conditional training of CIFAR-10,
```
python train.py --outdir='outdir_name' --data='data_path' --gpus=1 --cfg cifar --kimg 10000 --aug no 
--target 0.6 --noise_sd 0.05 --ts_dist priority --cond=true --resume='network_path_for_resuming_training'
```
For rest of the other dataset,
```
python train.py --outdir='outdir_name' --data='data_path' --gpus=1 --cfg paper256 --kimg 10000 --aug no 
--target 0.6 --noise_sd 0.05 --ts_dist priority --cond=true --resume='network_path_for_resuming_training'
```


### Generation
```
python generate.py --outdir='outdir_name' --seeds='select_seeds' --network='pretrained_network_path'
```
```
python interpolation.py --outdir='outdir_name' --network='pretrained_network_path' # Random generation between class 
```


## Calculating Metrics
```
python calc_metrics.py --metrics=kid50k_full,pr50k3_full --data='data_path' --mirror=1 --network='pretrained_network_path'
```



### Citation


### Acknowledgement
The code based on [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch) and [Diffusion-GAN](https://github.com/Zhendong-Wang/Diffusion-GAN). Thanks for their amazing works!
