echo $*
GLOG_vmodule=MemcachedClient=-1 srun --mpi=pmi2 -p $1 -n$2 --gres=gpu:8 --ntasks-per-node=8 python -u main_vgg.py --arch=vgg19_bn --save_dir=./checkpoints_vgg19_bn --epoches=70 --batch_size=256 --learning_rate=0.01 | tee -a log_vgg19_bn
# --load-path=checkpoint/res50/ckpt.pth.tar \
# --resume-opt
# --evaluate \
