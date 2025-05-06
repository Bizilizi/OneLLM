#!/bin/sh
#SBATCH --job-name="onellm"
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --partition=mcml-hgx-a100-80x4,mcml-hgx-h100-94x4,mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zverev@in.tum.de
#SBATCH --output=./logs/slurm-%A.out
#SBATCH --error=./logs/slurm-%A.out
 
nvidia-smi
source activate onellm

# Mount squashfs files
cleanup () {
    fusermount -u /tmp/zverev/vggsound
    rmdir /tmp/zverev/vggsound
}

trap cleanup EXIT

echo "Mounting VGGsound"
mkdir -p /tmp/zverev/vggsound
/usr/bin/squashfuse /dss/dssmcmlfs01/pn67gu/pn67gu-dss-0000/zverev/datasets/vggsound.squashfs /tmp/zverev/vggsound

# Activate your conda environment (adjust if needed)
source activate onellm
set -x

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

modality=$1
echo "This is $modality, page $SLURM_PROCID"

CMD="
python process_vggsound.py \
    --master_port \$((23862 + \$SLURM_PROCID)) \
    --gpu_ids \$SLURM_LOCALID \
    --tokenizer_path config/llama2/tokenizer.model \
    --llama_config config/llama2/7B.json \
    --pretrained_path weights/consolidated.00-of-01.pth \
    --dataset_path /tmp/zverev/vggsound \
    --video_csv ../../data/train.csv \
    --output_csv csv/$modality/predictions.csv \
    --page \$SLURM_PROCID \
    --per_page 1000 \
    --modality $modality \
    --prompt_mode single \
    --prompt 'From the given list of classes, which one do you see in this video? Answer only with the class names. Classes: {cl}'
"

SLURM_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    --jobid $SLURM_JOB_ID \
    "

srun $SLURM_ARGS bash -c "$CMD"
