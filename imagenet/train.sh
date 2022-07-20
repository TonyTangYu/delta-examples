#!/bin/bash
source /home/laizhiquan/dat01/ytang/common.env
which conda

source /home/laizhiquan/dat01/ytang/delta/delta.env
# source /home/laizhiquan/dat01/ytang/dtr.env
# source /home/laizhiquan/dat01/ytang/delta/delta-examples/imagenet/py.env
which python
python main.py -a resnet50 -b 64 /home/laizhiquan/dat01/newdataset/imagenet
