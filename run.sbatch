#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=vicreg
#SBATCH --mail-user=s319439@studenti.polito.it
#SBATCH --mail-type=END
#SBATCH --partition=cuda
#SBATCH --gres=gpu:1
#SBATCH --output=/home/nberardo/vicreg/train.log
#SBATCH --error=/home/nberardo/vicreg/train.err
###### 1 Load the module

module load nvidia/cudasdk
module load intel/python/3
# 2 Activate the virtual environment
source /home/nberardo/myenv/bin/activate
# 3 Run the python script
python evaluate.py --data-dir /home/eaiello/ImageNet/ --pretrained /home/nberardo/vicreg/resnet50_fullckpt.pth --exp-dir /home/nberardo/vicreg/experiment/lincls_down --lr-head 0.02
