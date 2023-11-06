#!/bin/sh

#SBATCH --job-name=track
#SBATCH --output=track.out
#SBATCH --time=2-00:00:00

module use /home/exacloud/software/modules
module load cudnn/7.6-10.1
module load cuda/10.1.243

eval "$(conda shell.bash hook)"
conda activate em3

python iou_track_big_cells.py --mask_dir ./data/predicted_cells/ --gt_dir ./data/groundtruth/ --dest_dir ./data/results/tracked_big/

python track_final.py
