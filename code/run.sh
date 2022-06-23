#/bin/bash
#BSUB -J DPMPN100
#BSUB -e /nfsshare/home/dengxiaofeng/log/DPMPN100_%J.err
#BSUB -o /nfsshare/home/dengxiaofeng/log/DPMPN100_%J.out
#BSUB -n 1
#BSUB -q gauss
#BSUB -gpu "num=1:mode=exclusive_process"

python run.py --dataset Beauty
