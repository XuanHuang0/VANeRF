# VANeRF: 3D Visibility-aware Generalizable Neural Radiance Fields for Interacting Hands (AAAI 2024)

This is an official implementation of "[3D Visibility-aware Generalizable Neural Radiance Fields for Interacting Hands](https://arxiv.org/pdf/2401.00979.pdf)".

<p float="left"> 
   <img src="https://github.com/XuanHuang0/VANeRF/blob/main/assets/13_nvs.gif" width="30%" height="30%" /> 
   <img src="https://github.com/XuanHuang0/VANeRF/blob/main/assets/8_nvs.gif" width="30%" height="30%" />
   <img src="https://github.com/XuanHuang0/VANeRF/blob/main/assets/1586_nvs.gif" width="30%" height="30%" />
</p>

## Installation

1. Please install python dependencies specified in requirements.txt:

   ```
   conda create -n vanerf python=3.9
   conda activate vanerf
   pip install -r requirements.txt
   ```
2. Register and download [MANO](https://mano.is.tue.mpg.de/)  data. Put `MANO_LEFT.pkl` and `MANO_RIGHT.pkl` in folder `$ROOT/smplx/models/mano`.

## Data preparation

1. Download [InterHand2.6M](https://mks0601.github.io/InterHand2.6M/) dataset and unzip it. (**Noted**: we used the `v1.0_5fps` version and `H+M` subset) The directory structure of `$ROOT/InterHand2.6M` is expected as follows:

    ```
   InterHand2.6M    
   │   ├── annotations 
   │   │   ├── skeleton.txt
   │   │   ├── subject.txt  
   │   │   ├── test    
   │   │   ├── train  
   │   │   └── val  
   │   └── images   
   │       ├── test 
   │       ├── train
   │       └── val 
   ```
    
2. Process the dataset by :

   ```
   python data_process/dataset_process.py
   ```

## Evaluation

1. Download the [pretrained model](https://drive.google.com/file/d/1lAxA2lR8sOOFw1XwgBberHDgV2C_XQwM/view?usp=sharing) and put it to `$ROOT/EXPERIMENTS/vanerf/ckpts/model.ckpt`.

2. Run evaluation:

   ```
   #small view variation
   python train.py --config ./configs/vanerf.json --run_val --model_ckpt ./EXPERIMENTS/vanerf/ckpts/model.ckpt
   #big view variation
   python train.py --config ./configs/vanerf_bvv.json --run_val --model_ckpt ./EXPERIMENTS/vanerf/ckpts/model.ckpt
   ```
   Results will be stored in folder `$ROOT/EXPERIMENTS/vanerf/`.
   
3. Visualize the dynamic results:

   ```
   python render_dynamic.py --config ./configs/vanerf.json --model_ckpt ./EXPERIMENTS/vanerf/ckpts/model.ckpt
   ```

## Training on InterHand2.6M dataset
Execute train.py script to train the model on the InterHand2.6M dataset.

   ```
   python train.py --config ./configs/vanerf.json --num_gpus 4
   ```
The output model would be store in `$ROOT//EXPERIMENTS/vanerf/ckpts`.

## Citation

If you find our code or paper useful, please consider citing:

```
@article{huang20243d,
  title={3D Visibility-aware Generalizable Neural Radiance Fields for Interacting Hands},
  author={Huang, Xuan and Li, Hanhui and Yang, Zejun and Wang, Zhisheng and Liang, Xiaodan},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2024}
}
```

