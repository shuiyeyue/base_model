echo $*
GLOG_vmodule=MemcachedClient=-1 srun --mpi=pmi2 -p $1 -n$2 --gres=gpu:1 --ntasks-per-node=1 python -u main.py --arch=vgg11 --save_dir=./checkpoints | tee -a log
# --load-path=checkpoint/res50/ckpt.pth.tar \
# --resume-opt
# --evaluate \
