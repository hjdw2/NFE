# NFE (Network Fission Ensembles)

## Introduction

This is the Pytorch code for the paper [Network Fission Ensembles for Low-Cost Self-Ensembles]. 

These codes are examples for CIFAR-100 with Wide ResNet28-10.

## Dependencies

* Python 3.6.13 (Anaconda)
* Pytorch 1.7.1
* CUDA 10.1

## Run

For N=2
```
python3 train_exit1.py --sparsity 0.5
```
For N=3
```
python3 train_exit12.py --sparsity 0.5
```

## Citation 

```latex
@article{NFE,
  title={Network Fission Ensembles for Low-Cost Self-Ensembles},
  author ={H. {Lee} and J. -S. {Lee}},
  journal = {arXiv preprint	arXiv:2408.02301}
  year={2024},
}
```

