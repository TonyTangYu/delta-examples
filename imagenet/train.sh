#!/bin/bash
# source /home/laizhiquan/dat01/ytang/common.env
# which conda

# source /home/laizhiquan/dat01/ytang/delta/delta.env
# source /home/laizhiquan/dat01/ytang/dtr.env
# source /home/laizhiquan/dat01/ytang/delta/delta-examples/imagenet/py.env
which python
python -m torch.distributed.launch --nproc_per_node=4 main.py -a resnet101 -b 1024 --world-size 4 --use-delta --budget 4000000000
