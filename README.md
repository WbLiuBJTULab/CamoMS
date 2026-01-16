# CamoMS

This code is for Camouflage Object Detection.

## Prepare Data

The training and testing datasets can be downloaded at [data](https://github.com/Garyson1204/HGINet?tab=readme-ov-file#prepare-data).

## Pretrained Model

Download the SMT from [model](https://pan.baidu.com/s/1BBgnJxONOTxL1dbAyMjnEA) with the fetch code: camo.

## Camouflage Object Detection Results

Download the prediction maps from [results](https://pan.baidu.com/s/1BBgnJxONOTxL1dbAyMjnEA) with the fetch code: camo.

## Installation

```
conda create -n CamoMS python=3.8
conda activate CamoMS
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install openmim==0.3.9
mim install mmcv-full==1.7.0 mmengine==0.8.4
pip install mmsegmentation==0.30.0 timm h5py einops fairscale imageio fvcore pysodmetrics
```

## Training

```
python etrain.py
```

## Testing & Evaluation

```
python etest.py
python eval.py
```

