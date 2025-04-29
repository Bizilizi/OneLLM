#!/bin/sh
#SBATCH --job-name="onellm"
#SBATCH --array=1-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=mcml-hgx-a100-80x4,mcml-hgx-h100-94x4,mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zverev@in.tum.de
#SBATCH --output=./logs/slurm-%j-%A_%a.out
#SBATCH --error=./logs/slurm-%j-%A_%a.out
 
nvidia-smi
source activate onellm
conda install conda-forge::squashfs-tools  

# Mount squashfs files
cleanup () {
    fusermount -u /tmp/zverev/$SLURM_ARRAY_TASK_ID/vggsound
    rmdir /tmp/zverev/$SLURM_ARRAY_TASK_ID/vggsound
}

trap cleanup EXIT

echo "Mounting VGGsound"
mkdir -p /tmp/zverev/$SLURM_ARRAY_TASK_ID/vggsound
squashfuse /dss/dssmcmlfs01/pn67gu/pn67gu-dss-0000/zverev/datasets/vggsound.squashfs /tmp/zverev/$SLURM_ARRAY_TASK_ID/vggsound

# Activate your conda environment (adjust if needed)
source activate onellm
set -x

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

modality=$1
echo "This is $modality, page $SLURM_ARRAY_TASK_ID"

# Run the script on each node, assigning each task to a different GPU
srun --exclusive --ntasks=1 python process_vggsound.py \
    --gpu_ids 0 \
    --tokenizer_path config/llama2/tokenizer.model \
    --llama_config config/llama2/7B.json \
    --pretrained_path weights/consolidated.00-of-01.pth \
    --dataset_path /tmp/zverev/$SLURM_ARRAY_TASK_ID/vggsound \
    --video_csv ../../data/train.csv \
    --output_csv csv/$modality/predictions.csv \
    --page $SLURM_ARRAY_TASK_ID \
    --per_page 1000 \
    --modality $modality \
    --prompt_mode single \
    --prompt "From the given list of classes, which one do you see in this video? Answer only with the class names. Classes: {cl}"
