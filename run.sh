#!/bin/bash
export PATH=/opt/conda/bin:$PATH
export PYTHONPATH=/home/zhangzr/mmsegmentation_kaggle:$PYTHONPATH
GPU=$2



# config=upernet_convnext_base_fp16_512x512_80k_FTU
# config=upernet_swin_base_patch4_window7_512x512_80k_FTU
# config=upernet_swin_base_patch4_window7_512x512_80k_FTU_whole
# config=upernet_convnext_base_fp16_512x512_80k_FTU_whole
config=upernet_swin_base_patch4_window7_512x512_80k_FTU_whole_dice
if [ $1 = "train" ]; then
    # CUDA_VISIBLE_DEVICES=$GPU PORT=23472 ./tools/dist_train.sh configs/convnext/${config}.py 1 --work-dir cache/${config} 
    CUDA_VISIBLE_DEVICES=$GPU PORT=23473 ./tools/dist_train.sh configs/swin/${config}.py 1 --work-dir cache/${config} 
    # CUDA_VISIBLE_DEVICES=$GPU PORT=23472 ./tools/dist_train.sh configs/swin_v2/${config}.py 2 --work-dir cache/${config} 
elif [ $1 = "test" ]; then
    
    # CUDA_VISIBLE_DEVICES=$GPU python ./tools/test.py configs/convnext/${config}.py ../../input/mmsegckpts/iter_1600.pth --format-only --eval-options "imgfile_prefix=./test_results/upernet_convnext_base_fp16_256x256_16k_kaggle_no_crop"
    CUDA_VISIBLE_DEVICES=$GPU ./tools/dist_test.sh configs/convnext/${config}.py ./cache/upernet_originsize_convnext_base_fp16_320x384_160k_kaggle_25d_multilabel/best_mDice_iter_64000.pth 2 --eval mDice # --format-only --eval-options "imgfile_prefix=./test_results/upernet_convnext_base_fp16_256x256_16k_kaggle_no_crop"
fi

