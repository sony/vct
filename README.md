# [ICML 2025] Implementation of ``VCT: Training Consistency Models with Variational Noise Coupling''
This repository houses the implementation of our work, **VCT: Training Consistency Models with Variational Noise Coupling**, accepted at ICML 2025!
- arXiv: https://arxiv.org/abs/2502.18197

## Requirements
This code uses Weights & Biases, and assumes that you have your wandb key in a file named `wandb_config.py` in a variable named `key=your_wandb_key`.
This code uses the following libraries:
```angular2html
pytorch 
torchvision 
torchaudio 
pytorch-cuda
lightning
torchmetrics[image]
scipy 
scikit-learn 
matplotlib 
wandb
hydra-core
POT
pyspng
```
You can check `requirements.txt` for the exact packases used in our experiments.
## Training
In the following we provide the commands to reproduce our models. To run the baselines, set `model.coupling=ot` for OT, while for independent coupling set `model.coupling=independent`. In either case, set `grad_clip_val=0`. To switch to Flow Matching linear interpolation kernel, set `model.kerne=cot` (originally named as Conditional Optimal Transport, but referred to as LI in the paper).
The batch size is specified as batch per device, so adjust according to the number of GPUs you intend to use.
### iCT-VC Fashion MNIST
```angular2html
python main.py project=vct_fmnist dataset=fmnist dataset.num_workers=16 model=ict network=ddpmpp network.dropout=0.3 dataset.batch_size=128 gradient_clip_val=200 model.class_conditional=False model.kernel=ve model.coupling=vae model.kl_loss_scale=30 network.model_channels=64
```

### iCT-VC CIFAR10
```angular2html
python main.py project=vct_cifar dataset=cifar10 dataset.num_workers=16 model=ict network=ddpmpp network.dropout=0.3 dataset.batch_size=512 gradient_clip_val=200 model.class_conditional=False model.kernel=ve model.coupling=vae model.kl_loss_scale=30
```

### ECM-VC CIFAR10
```angular2html
python main.py project=vct_cifar dataset=cifar10 model=ecm network=ddpmpp network.reload_url='https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl' network.dropout=0.2 dataset.batch_size=128 gradient_clip_val=200 model.class_conditional=False model.kernel=ve model.coupling=vae model.kl_loss_scale=10 deterministic=True
```

### ECM-VC FFHQ 64x64
To prepare the dataset, follow the instructions from [https://github.com/NVlabs/edm](https://github.com/NVlabs/edm), and make sure to specify 'your_data_dir' correctly in the command below.
```angular2html
python main.py project=vct_ffhq dataset=ffhq dataset.data_dir='your_data_dir' model=ecm network=ddpmpp network.channel_mult=[1,2,2,2] network.reload_url='https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-vp.pkl' network.dropout=0.2 dataset.batch_size=128 gradient_clip_val=200 model.class_conditional=False model.kernel=ve model.coupling=vae model.kl_loss_scale=10 deterministic=True
```

### ECM-VC ImageNet 64x64
To prepare the dataset, follow the instructions from [https://github.com/NVlabs/edm](https://github.com/NVlabs/edm) or [https://github.com/locuslab/ect](https://github.com/locuslab/ect), and make sure to specify 'your_data_dir' correctly in the command below.
```angular2html
python main.py project=vct_imagenet reload=False run_path= compute_fid=True save_checkpoints=False log_on_epoch=False log_frequency=1000 dataset=imagenet dataset.data_dir='your_data_dir' dataset.batch_size=128 dataset.num_workers=64 model=ecm model.mid_t=[1.526] model.total_training_steps=200000 model.c=0.06 model.p_mean=-0.8 model.p_std=1.6 model.q=4 model.n_stages=4 model.class_conditional=True model.kernel=cot model.coupling=vae model.kl_loss_scale=90 model.use_lr_decay=True model.learning_rate=0.001 model.ema_rate=0.1 model.ema_type=power model.loss_weighting=karras model.encoder_size=big network=edm2 network.reload_url=https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img64-s-1073741-0.075.pkl network.dropout=0.4 gradient_clip_val=200 deterministic=True
```
## References
Parts of the code were adapted from the following codebases:
- [https://github.com/NVlabs/edm](https://github.com/NVlabs/edm)
- [https://github.com/locuslab/ect](https://github.com/locuslab/ect)
- [https://github.com/Kinyugo/consistency_models](https://github.com/Kinyugo/consistency_models)
- [https://github.com/atong01/conditional-flow-matching](https://github.com/atong01/conditional-flow-matching)
- [https://github.com/NVlabs/edm2](https://github.com/NVlabs/edm2)

## Contact
- Gianluigi Silvestri: gianlu.silvestri@gmail.com
- Chieh-Hsin (Jesse) Lai: Chieh-Hsin.Lai@sony.com
