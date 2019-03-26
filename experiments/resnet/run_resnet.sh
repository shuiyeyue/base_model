echo $*
GLOG_vmodule=MemcachedClient=-1 srun --mpi=pmi2 -p $1 -n$2 --gres=gpu:8 --ntasks-per-node=8 python -u main_resnet.py --arch=resnet50 --save_dir=./checkpoints_resnet50 --epoches=60 --batch_size=256 --learning_rate=0.1 | tee -a log_resnet50
# --load-path=checkpoint/res50/ckpt.pth.tar \
# --resume-opt
# --evaluate \
