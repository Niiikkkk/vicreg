#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --job-name=vicreg
#SBATCH --mail-user=s319439@studenti.polito.it
#SBATCH --mail-type=END
#SBATCH --partition=cuda
#SBATCH --gres=gpu:4
#SBATCH --output=/home/nberardo/vicreg/train.log
#SBATCH --error=/home/nberardo/vicreg/train.err
###### 1 Load the module

module load nvidia/cudasdk
module load intel/python/3
# 2 Activate the virtual environment
source /home/nberardo/myenv/bin/activate
# 3 Run the python script
python -m torch.distributed.launch --nproc_per_node=4 main_vicreg.py --data-dir /home/eaiello/ImageNet --exp-dir /home/nberardo/vicreg/experiment/ --arch resnet50 --epochs 100 --batch-size 512 --base-lr 0.3
