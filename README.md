## DuDGAN: Improving Class-Conditional GANs via Dual-Diffusion 

<img src="https://github.com/taesunyeom/DuDGAN/assets/102474982/7c7055e5-bc7a-4397-b5bb-d77467a67af6"/>


### Requirement
---
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
You should make '.jsonl' file by enumerating images and discrete class labels.
```
 trainining_dataset_folder
    └── AAA (class name)
    └── BBB
    └── CCC
    ...
    └── '***.json' --> ["training_dataset_folder/AAA/aaa.jpg", #(class number)], ...
```
    
    


### Training Process
---

```
```
## Metrics
---

```
```

### Image generation using Pretrained Network
---



### Citation
---


### Acknowledgement
The code based on [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch) and [Diffusion-GAN](https://github.com/Zhendong-Wang/Diffusion-GAN). Thanks for their amazing works!
