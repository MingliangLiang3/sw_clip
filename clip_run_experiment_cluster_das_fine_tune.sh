#!/usr/bin/env bash
#SBATCH --partition=PARTITION_NAME
#SBATCH --account=ACCOUNT_NAME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=6
#SBATCH --job-name=open_clip
#SBATCH --mem=200G
#SBATCH --time=128:00:00
#SBATCH --output=./logs/train/my-experiment-%J.out
#SBATCH --error=./logs/train/my-experiment-%J.err
#SBATCH --mail-user=user

source /user/virtual_environments/tiny-voxceleb-venv/bin/activate

export MASTER_PORT=$(expr 10001 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=8
echo "WORLD_SIZE="$WORLD_SIZE

echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

export PYTHONPATH="$PYTHONPATH:$PWD/src"
cd src
torchrun --nproc_per_node=8 --master_port=25698 training/main.py \
  --save-frequency=1 \
  --report-to=tensorboard \
  --zeroshot-frequency=1 \
  --train-data="../path/to/cc3m/cc3m_train.csv" \
  --imagenet-val="./path/to/imagenet_validation" \
  --csv-img-key=image \
  --csv-caption-key=caption \
  --model=RN50 \
  --pretrained="./path/to/checkpoints/epoch_K.pt" \
  --batch-size=768 \
  --warmup=125 \
  --lr=1e-3 \
  --wd=0.1 \
  --epochs=1 \
  --workers=8 \
  --seed=42 \
  --local-loss \
  --gather-with-grad \
  --force-custom-text \
  --name pretrain_cc3m_train_RN50_subsample_finetune
