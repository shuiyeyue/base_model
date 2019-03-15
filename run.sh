echo $*
GLOG_vmodule=MemcachedClient=-1 srun --mpi=pmi2 -p $1 -n$2 --gres=gpu:1 --ntasks-per-node=1 python -u main.py --arch=vgg19_bn --save_dir=./checkpoints_vgg19 --epoches=100 --batch_size=128 | tee -a log_vgg19
# --load-path=checkpoint/res50/ckpt.pth.tar \
# --resume-opt
# --evaluate \
