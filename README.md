# Bayesian Invariant Risk Minmization (BIRM)
The repo for [Bayesian Invariant Risk Minimization](https://openaccess.thecvf.com/content/CVPR2022/papers/Lin_Bayesian_Invariant_Risk_Minimization_CVPR_2022_paper.pdf), CVPR2022 (oral).

Our implementation is based on the source code of [IRM](https://github.com/facebookresearch/InvariantRiskMinimization),  [ColoredObject](https://github.com/Faruk-Ahmed/predictive_group_invariance), and [CifarMnist](https://github.com/HKUST-MLResearch/IRMBed).

# Requirements 
Our code works with the following environment.
* `python=3.7.0`
* `torch=1.3.1`
* `h5py==2.8.0`

To install the necessary packages for the project, please run: `pip install -r requirements.txt`.

# Quick Start (For Reproducing Results)
1. To perform BIRM on CMNIST (with 20K training data). Run the command `sh auto_CMNIST.sh`. The expected test accuracy is `67.0±1.8`.
2. To perform BIRM on ColoredObject. First run `sh prepare_coco_dataset.sh` to download MSCOCO dataset and preprocess the images (it may take several hours, please be patient). Second run the command `sh auto_CifarMnist.sh` to train BRIM. The expected test accuracy is `78.1±0.6`.
3. To perform BIRM on CifarMnist. Run the command `sh auto_ColoredObject.sh`. The expected test accuracy is `59.3±2.3`.

Important arguments:
* `dataset`: chosen in `CMNIST`, `ColoredObject` and `CifarMnist`;
* `l2_regularizer_weight`: weight decay coeffient;
* `lr`: learning rate;
* `opt`: the optimizer. By default, we use `adam` for CMNIST and use `sgd` for ColoredObject and CifarMnist; 
* `data_num`: (only valid for dataset CMNIST) the number of training data;
* `penalty_weight`:  the weight of the BIRM penalty;
* `penalty_anneal_iters`: the steps that we trians ERM first, after which BIRM penalty will be applied.
* `step_gamma`: the ratio of step decay, `0.1` means the learning rate will decay by `0.1` at the middle of the training steps.
# Datasets
## Implemented Datasets
* CMNIST: the most popular dataset in IRM literatures. The invariant feature is the shape of the digit from MNIST and the spurious feature is the attached color.
* ColoredObject: Following [Faruk-Ahmed](https://github.com/Faruk-Ahmed/predictive_group_invariance), we construct coloredObject by superimposing objects extracted from MSCOCO on a colored background (spurious feature)
* CifarMnist: Following [Shah](https://arxiv.org/abs/2006.07710), we construct each image in CifarMnist by  by concatenating two component images: CIFAR-10 (invariant) and MNIST (spurious).

Refer to Section 5 of our paper for detailed discription of the datasets.

## Use BIRM on Your Own Data
We provider interface for you to include your own data. You need to inherit the 
 class `IRMDataProvider`, and re-implement the function `fetch_train` and `fetch_test`. The main function will call `fetch_train` to get training data for each step. `fetch_train` should return the following values:

* `train_x`: the feature tensor;
* `train_y`: the label tensor;
* `train_g`: the tensor contains values indicating which environmnets the data are from;
* `train_c`(optional): the tensor contains values indicating whether the spurious features align with the labels.

The structure of the return value of `fetch_test` are similar with `fetch_train`.
# Contact Information

For help or issues using Bayesian Invariant Risk Minimization, please submit a GitHub issue.

For personal communication related to BayesianIRM, please contact Yong Lin (`ylindf@connect.ust.hk`).

# Reference 
If you use or extend our work, please cite the following paper:
```
@inproceedings{lin2022bayesian,
  title={Bayesian Invariant Risk Minimization},
  author={Lin, Yong and Dong, Hanze and Wang, Hao and Zhang, Tong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={16021--16030},
  year={2022}
}
```

