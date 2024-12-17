#!/bin/sh
#SBATCH --time 23:59:59
#SBATCH --cpus-per-gpu=8
#SBATCH --partition=amdgpu
#SBATCH --gres=gpu:4
#SBATCH --array=1
#BATCH --exclusive
/bin/hostname
nvidia-smi
pwd
ls -l

ml load Python/3.11.5-GCCcore-13.2.0
ml load PyTorch/2.5.1-foss-2023b-CUDA-12.4.0
ml load Z3/4.13.0-GCCcore-13.2.0
ml load torchvision/0.20.1-foss-2023b-CUDA-12.4.0
ml load matplotlib/3.8.2-gfbf-2023b
ml load OpenSSL/3
ml load matplotlib/3.8.2-gfbf-2023b
ml load tqdm/4.66.2-GCCcore-13.2.0
python -m wandb online

export OMP_NUM_THREADS=4

image_size=256
vq_model=TiTok-256
bs=128
lr="1e-5"
disc_start=200000
disc_weight=0.5
PYTHONPATH="." bash scripts/tokenizer/train_vq.sh --data-path /mnt/data/Public_datasets/imagenet/imagenet_pytorch/ --image-size $image_size --vq-model $vq_model --cloud-save-path checkpoints --global-batch-size $bs --lr $lr --disc-start $disc_start --disc-weight $disc_weight

vq_ckpt=checkpoints/2024-12-13-19-30-21/023-VQ-8/checkpoints/0175000.pt
# PYTHONPATH="." bash scripts/tokenizer/reconstruction_vq.sh --data-path /mnt/data/Public_datasets/imagenet/imagenet_pytorch/ --vq-model VQ-16 --vq-ckpt $vq_ckpt --image-size $image_size
