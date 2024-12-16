# !/bin/bash
set -x

nnodes=1
nproc_per_node=4
node_rank=0

torchrun \
--nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank=$node_rank \
tokenizer/tokenizer_image/vq_train.py "$@"


#torchrun \
#--nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank=$node_rank \
#--master_addr=$master_addr --master_port=$master_port \
#tokenizer/tokenizer_image/vq_train.py "$@"
