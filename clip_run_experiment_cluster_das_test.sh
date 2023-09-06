#!/usr/bin/env bash
#SBATCH --partition=PARTITION_NAME
#SBATCH --account=ACCOUNT_NAME
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --job-name=open_clip
#SBATCH --mem=50G
#SBATCH --time=12:00:00
#SBATCH --output=./logs/test/my-experiment-%J.out
#SBATCH --error=./logs/test/my-experiment-%J.err
#SBATCH --mail-user=user

source /user/virtual_environments/tiny-voxceleb-venv/bin/activate

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

export PYTHONPATH="$PYTHONPATH:$PWD/src"
cd src
python -u training/main.py \
  --report-to tensorboard \
  --imagenet-val="./path/to/imagenet_validation/" \
  --csv-img-key=image \
  --csv-caption-key=caption \
  --batch-size=256 \
  --workers=6 \
  --model=RN50 \
  --pretrained="./path/to/checkpoints/epoch_K.pt" \
  --seed=42 \
  --local-loss \
  --gather-with-grad \
  --force-custom-text
