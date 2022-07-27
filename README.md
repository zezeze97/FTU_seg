# Kaggle 比赛

该仓库用于[HuBMAP + HPA - Hacking the Human Body](https://www.kaggle.com/competitions/hubmap-organ-segmentation)比赛

# 数据集探索


# 环境安装

```sh

conda create -n mmseg-kaggle python=3.10 -y

conda activate mmseg-kaggle

# conda install pytorch=1.11.0 torchvision cudatoolkit=11.3 -c pytorch

pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
pip install segmentation_models_pytorch
git clone https://github.com/zezeze97/FTU_seg.git

cd {path of project}

pip install -e .  

```

# 数据集下载，预处理

从官网下载好数据集后，放在该项目的data目录下，运行[kaggle_segmentation/eda_mask.ipynb](kaggle_segmentation/eda_mask.ipynb)

# 训练，测试

```sh

# 训练

bash run.sh train $GPU

# 测试

bash run.sh test$GPU

```

# 可视化预测

[kaggle_segmentation/inference_demo.ipynb](kaggle_segmentation/inference_demo.ipynb)

# Note
- 虽然有多个类别，但是比赛只要求区分FTU，我先处理成2分类问题
- 使用multilabel segmentor, 最后的激活用sigmoid而不是softmax
- 图片分辨率很大3000x3000, 使用slide模式进行inference，而不是whole!
- baseline使用的convnext-base

# TODO
- 实验结果整理
- 增加smp unet decoder 移植 https://github.com/CarnoZhao/Kaggle-UWMGIT/blob/kaggle_tractseg/mmseg/models/segmentors/smp_models.py

