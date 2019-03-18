echo $*
GLOG_vmodule=MemcachedClient=-1 srun --mpi=pmi2 -p $1 -n$2 --gres=gpu:1 --ntasks-per-node=1 python -u main.py --arch=vgg19_bn --save_dir=./checkpoints_vgg19_bn --epoches=100 --batch_size=64 --learning_rate=0.0025| tee -a log_vgg19_bn
# --load-path=checkpoint/res50/ckpt.pth.tar \
# --resume-opt
# --evaluate \
