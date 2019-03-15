echo $*
GLOG_vmodule=MemcachedClient=-1 srun --mpi=pmi2 -p $1 -n$2 --gres=gpu:1 --ntasks-per-node=1 python -u main.py --arch=vgg19_bn --save_dir=./checkpoints --epoches=100 | tee -a log
# --load-path=checkpoint/res50/ckpt.pth.tar \
# --resume-opt
# --evaluate \
