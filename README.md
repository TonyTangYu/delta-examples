# DELTA Examples

DELTA examples of paper [DELTA: Dynamically Optimizing GPU Memory beyond Tensor Recomputation](https://arxiv.org/abs/2203.15980).

## DELTA-PyTorch

[DELTA-PyTorch](https://github.com/TonyTangYu/pytorch) is implemented as a fork of [DTR-PyTorch](https://github.com/uwsampl/pytorch). You could build and install this version of PyTorch from source or follow the instructions.

```
conda create -n delta python=3.6
conda activate delta
git clone --recursive https://github.com/TonyTangYu/pytorch -b delta pytorch
python setup.py install
```


## Running Examples 

You could run DELTA-examples from the following instructions.

```
git clone https://github.com/TonyTangYu/delta-examples
```

### Running ResNet-50

Running ResNet-50 might require `torchvision`.
`torchvision` needs also installed from source. 
```
git clone https://github.com/pytorch/vision -b v0.8.0-rc1
python setup.py install
``` 
training ResNet-50
```
cd imagenet
python main.py -a resnet50 -b 64 path/to/your/imagenet --use-delta --budget 10000000000
```

### Running BERT

```
cd BingBertSquad
sh run_squad_baseline.sh
```
You could modify the argument `budget` to set the memory budget in your experiments.

## Acknowledgement

[DELTA-PyTorch](https://github.com/TonyTangYu/pytorch) is implemented as a fork of [DTR-PyTorch](https://github.com/uwsampl/pytorch). Thanks for the help from [Marisa Kirisame](https://github.com/MarisaKirisame). 

## Citation

Please cite our paper:

```
@article{tang2022delta,
  title={DELTA: Dynamically Optimizing GPU Memory beyond Tensor Recomputation},
  author={Tang, Yu and Wang, Chenyu and Zhang, Yufan and Liu, Yuliang and Zhang, Xingcheng and Qiao, Linbo and Lai, Zhiquan and Li, Dongsheng},
  journal={arXiv preprint arXiv:2203.15980},
  year={2022}
}
```
