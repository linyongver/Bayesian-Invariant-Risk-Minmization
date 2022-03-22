# Bayesian-Invariant-Risk-Minmization
This is the source code for the CVPR 2022 paper entitled [Bayesian Invariant Risk Minmization]().

Our implementation is based on the source code of [IRM](https://github.com/facebookresearch/InvariantRiskMinimization),  [ColoredObject](https://github.com/Faruk-Ahmed/predictive_group_invariance), and [CifarMnist](https://github.com/HKUST-MLResearch/IRMBed).

# Requirements 
Our code works with the following environment.
* `python=3.7.0`
* `pytorch=1.3.1`

To install the necessary packages for the project, please run: `pip install -r requirements.txt`.

# Quick Start (For reproducing results)
1. To perform BIRM on CMNIST. Run the command `sh auto_CMNIST.sh`. The expected test accuracy is `1`.
2. To perform BIRM on ColoredObject. Run the command `sh auto_CifarMnist.sh`. The expected test accuracy is `1`.
3. To perform BIRM on CifarMnist. Run the command `sh auto_ColoredObject.sh`. The expected test accuracy is `1`.
Important arguments:
* `dataset`: chosen in `mnist`, `cifar` and `coco_color`.
* `penalty_weight`:  the weight of the BIRM penalty.
* `penalty_anneal_iters`: the steps that we trians ERM first, after which BIRM penalty will be applied.
# Datasets
## Implemented Datasets
* CMNIST: the most popular dataset in IRM literatures. The invariant feature is the shape of the digit from MNIST and the spurious feature is the attached color.
* ColoredObject: Following [Faruk-Ahmed](https://github.com/Faruk-Ahmed/predictive_group_invariance), we construct coloredObject by superimposing objects extracted from MSCOCO on a colored background (spurious feature)
* CifarMnist: Following [Shah](https://arxiv.org/abs/2006.07710), we construct each image in CifarMnist by  by concatenating two component images: CIFAR-10 (invariant) and MNIST (spurious).
Refer to Section 5 of our paper for detailed discription of the datasets.
## Use with your own data

# Contact information

For help or issues using Bayesian Invariant Risk Minimization, please submit a GitHub issue.

For personal communication related to BayesianIRM, please contact Yong Lin (`ylindf@connect.ust.hk`).

# Citation 
If you use or extend our work, please cite the following paper:
```
@inproceedings{Lin2022BIRM,
    title = "Bayesian Invariant Risk Minmization",
    author = "Yong, Lin  and
      Hanze, Dong  and
      Hao, Wang  and
      Tong, Zhang",
    booktitle = "IEEE/CVF Conference on Computer Vision and Pattern Recognition 2022",
    year = "2022",
    address = "Online"
}
```



