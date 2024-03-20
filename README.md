# VANeRF: 3D Visibility-aware Generalizable Neural Radiance Fields for Interacting Hands (AAAI 2024)

This is an official implementation of "[3D Visibility-aware Generalizable Neural Radiance Fields for Interacting Hands](https://arxiv.org/pdf/2401.00979.pdf)".

<img src="https://github.com/XuanHuang0/VANeRF/assets/13_nvs.gif" alt="8_nvs" style="zoom: 67%;" /> <img src="https://github.com/XuanHuang0/VANeRF/assets/8_nvs.gif" alt="8_nvs" style="zoom: 67%;" />

<img src="https://github.com/XuanHuang0/VANeRF/assets/1567_nvs.gif" alt="1567_nvs" style="zoom: 67%;" /><img src="https://github.com/XuanHuang0/VANeRF/assets/1586_nvs.gif" alt="1586_nvs" style="zoom: 67%;" />

## Installation

Please install python dependencies specified in requirements.txt:

```
conda create -n vanerf python=3.9
conda activate vanerf
pip install -r requirements.txt
```

## Data preparation

1. Download [InterHand2.6M](https://mks0601.github.io/InterHand2.6M/) dataset and unzip it. (**Noted**: we used the `v1.0_5fps` version and `H+M` subset)

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

3. Visualize the dynamic results:

   ```
   python render_dynamic.py --config ./configs/vanerf.json --model_ckpt ./EXPERIMENTS/vanerf/ckpts/model.ckpt
   ```

## Training on InterHand2.6M dataset

```
python train.py --config ./configs/vanerf.json --num_gpus 4
```

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

