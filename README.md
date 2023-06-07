## DuDGAN: Improving Class-Conditional GANs via Dual-Diffusion  - Pytorch Implementation

<img src="https://github.com/taesunyeom/DuDGAN/assets/102474982/7c7055e5-bc7a-4397-b5bb-d77467a67af6"/>

### Abstract

Class-conditional image generation using generative adversarial networks (GANs) has been investigated through various techniques; however, it continues to face challenges such as mode collapse, training instability, and low-quality output in cases of datasets with high intra-class variation. Furthermore, most GANs often converge in larger iterations, resulting in poor iteration efficacy in training procedures. While Diffusion-GAN has shown potential in generating realistic samples, it has a critical limitation in generating class-conditional samples. To overcome these limitations, we propose a novel approach for class-conditional image generation using GANs called DuDGAN, which incorporates a dual diffusion-based noise injection process. Our method consists of three unique networks: a discriminator, a generator, and a classifier. During the training process, Gaussian-mixture noises are injected into the two noise-aware networks, the discriminator and the classifier, in distinct ways. This noisy data helps to prevent overfitting by gradually introducing more challenging tasks, leading to improved model performance. As a result, our method outperforms state-of-the-art conditional GAN models for image generation in terms of performance. We evaluated our method using the AFHQ, Food-101, and CIFAR-10 datasets and observed superior results across metrics such as FID, KID, Precision, and Recall score compared with comparison models, highlighting the effectiveness of our approach.

### Requirement
1. one or more CUDA-available GPUs 

2. Python 3.7.x ~ 3.9.x

3. Pytorch 1.8.0+cu111 or version that fit to your enviornment. (We do not sure it works on recent version. We recommend install previous version of Pytorch in https://pytorch.org/get-started/previous-versions/)

4. Required libraries : 
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
Class-conditional generation using pretrained network.
```
python generate.py --outdir='outdir_name' --seeds='select_seeds' --network='pretrained_network_path'
```

Between-class interpolated generation (default setting is 'random_class to another_random_class')
```
python interpolation.py --outdir='outdir_name' --network='pretrained_network_path' # Random generation between class 
```


### Calculating Metrics
```
python calc_metrics.py --metrics=kid50k_full,pr50k3_full --data='data_path' --mirror=1 --network='pretrained_network_path'
```

### Customizing dual-diffusion

work in progress...

### Citation
```
@article{yeom2023dudgan,
  title={DuDGAN: Improving Class-Conditional GANs via Dual-Diffusion},
  author={Yeom, Taesun and Lee, Minhyeok},
  journal={arXiv preprint arXiv:2305.14849},
  year={2023}
}
```

### Acknowledgement
The code based on [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch) and [Diffusion-GAN](https://github.com/Zhendong-Wang/Diffusion-GAN). Thanks for their amazing works!
