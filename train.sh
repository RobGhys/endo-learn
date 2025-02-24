#!/bin/bash
#
#SBATCH --job-name=training_seg
#SBATCH --array=0-4
#SBATCH --time=48:00:00 # hh:mm:ss
#
#SBATCH --ntasks=1
#SBATCH --gres="gpu:1"
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=7650
#SBATCH --partition=gpu
#
#SBATCH --mail-user=robin.ghyselinck@unamur.be
#SBATCH --mail-type=ALL
#
#SBATCH --account=lysmed


# ------------------------- work -------------------------
# Setting the number of worker for data loading
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

#export PYTHONPATH=$PYTHONPATH:/gpfs/scratch/acad/lysmed/endo-learn/src/

# Check if wandb_api_key is provided as a command line argument
if [ "$#" -ne 1 ]; then
    echo "Error: You must provide the wandb_api_key as an argument."
    echo "Usage: sbatch training.sh <wandb_api_key>"
    exit 1
fi

# Assign the first command line argument to wandb_api_key
wandb_api_key=$1

echo "Starting Task #: $SLURM_ARRAY_TASK_ID"

#resume_path="/gpfs/scratch/acad/lysmed/seg-equi-architectures/outputs/coco/UNet_e2cnn/fold_${SLURM_ARRAY_TASK_ID}/checkpoint_epoch_159.pth"

# Check if the resume file exists
if [ ! -f "$resume_path" ]; then
    echo "Error: Resume file not found at $resume_path"
    exit 1
fi

python src/main.py \
--data /gpfs/scratch/acad/lysmed/data-kvasir/labeled-images \
--wandb_api_key $wandb_api_key \

echo "Finished Task #: $SLURM_ARRAY_TASK_ID"

echo "Exiting the program."
