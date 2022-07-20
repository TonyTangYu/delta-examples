#!/bin/bash
source /home/laizhiquan/dat01/ytang/common.env
which conda

source /home/laizhiquan/dat01/ytang/delta/delta.env
# source /home/laizhiquan/dat01/ytang/dtr.env
# source /home/laizhiquan/dat01/ytang/delta/delta-examples/imagenet/py.env
which python

PARTITION=$1
JOB_NAME=$2
GPUS=$3
if [ $GPUS -lt 4 ]; then
    GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS}
else
    GPUS_PER_NODE=${GPUS_PER_NODE:-4}
fi
CPUS_PER_TASK=${CPUS_PER_TASK:-4}
SRUN_ARGS=${SRUN_ARGS:-""}

srun -p ${PARTITION} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    ${SRUN_ARGS} \
    python main.py \
    -a resnet50 \
    -b 64 \
    /home/laizhiquan/dat01/newdataset/imagenet \
    2>&1 | tee ${JOB_NAME}.log

