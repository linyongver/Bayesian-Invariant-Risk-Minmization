import argparse
import random
from cm_spurious_dataset import get_data_loader_cifarminst
from coco_dataset import get_spcoco_dataset
import math
import numpy as np
import torch
from torchvision import datasets
from data import CowCamels
from data import AntiReg
import os
import sys
from torch import nn, optim, autograd

def return_model(flags):
    model_type = None
    if flags.irm_type == "erm":
        model_type = "erm"
    elif flags.irm_type == "irmv1":
        if flags.dataset == "CMNIST":
            model_type="irmv1"
        elif flags.dataset == "ColoredObject":
            model_type="irmv1b"
        elif flags.dataset == "CifarMnist":
            model_type="irmv1b"
        else:
            raise("Please specify the irm model for this dataset!")
    elif flags.irm_type == "birm":
        if flags.dataset == "CMNIST":
            model_type="bayes_fullbatch"
        elif flags.dataset == "ColoredObject":
            model_type="bayes_variance"
        elif flags.dataset == "CifarMnist":
            model_type="bayes_batch"
        else:
            raise("Please specify the bayesian irm model for this dataset!")
        flags = update_flags(flags)
    else:
        raise Exception
    return flags, model_type


def update_flags(flags):
    assert flags.irm_type == "birm"
    if flags.dataset == "CMNIST":
        if flags.data_num == "5000":
            flags.prior_sd_coef =1350
        else:
            flags.prior_sd_coef =1200
    elif flags.dataset == "ColoredObject":
        flags.prior_sd_coef=1000
    elif flags.dataset == "CifarMnist":
        flags.prior_sd_coef=1500
    else:
        raise("Please specify the bayesian irm model for this dataset!")
    return flags

def torch_bernoulli(p, size):
    return (torch.rand(size) < p).float()

def torch_xor(a, b):
    return (a-b).abs()

def concat_envs(con_envs):
    con_x = torch.cat([env["images"] for env in con_envs])
    con_y = torch.cat([env["labels"] for env in con_envs])
    con_g = torch.cat([
        ig * torch.ones_like(env["labels"])
        for ig,env in enumerate(con_envs)])
    # con_2g = torch.cat([
    #     (ig < (len(con_envs) // 2)) * torch.ones_like(env["labels"])
    #     for ig,env in enumerate(con_envs)]).long()
    con_c = torch.cat([env["color"] for env in con_envs])
    # con_yn = torch.cat([env["noise"] for env in con_envs])
    # return con_x, con_y, con_g, con_c
    return con_x.cuda(), con_y.cuda(), con_g.cuda(), con_c.cuda()


def merge_env(original_env, merged_num):
    merged_envs = merged_num
    a = original_env
    interval = (a.max() - a.min()) // merged_envs + 1
    b = (a - a.min()) // interval
    return b

def eval_acc_class(logits, labels, colors):
    acc  = mean_accuracy_class(logits, labels)
    minacc = mean_accuracy_class(
      logits[colors!=1],
      labels[colors!=1])
    majacc = mean_accuracy_class(
      logits[colors==1],
      labels[colors==1])
    return acc, minacc, majacc

def mean_accuracy_class(logits, y):
    preds = (logits > 0.).float()
    return ((preds - y).abs() < 1e-2).float().mean()

def eval_acc_multi_class(logits, labels, colors):
    acc  = mean_accuracy_multi_class(logits, labels)
    minacc = mean_accuracy_multi_class(
      logits[colors.view(-1)!=1],
      labels[colors.view(-1)!=1])
    majacc = mean_accuracy_multi_class(
      logits[colors.view(-1)==1],
      labels[colors.view(-1)==1])
    return acc, minacc, majacc

def mean_accuracy_multi_class(output, target):
    probs = torch.softmax(output, dim=1)
    winners = probs.argmax(dim=1)
    corrects = (winners == target.view(-1))
    accuracy = corrects.sum().float() / float(corrects.size(0))
    return accuracy

def eval_acc_reg(logits, labels, colors):
    acc  = mean_nll_reg(logits, labels)
    minacc = torch.tensor(0.0)
    majacc = torch.tensor(0.0)
    return acc, minacc, majacc


def get_strctured_penalty(strctnet, ebd, envs_num, xis):
    x0, x1, x2 = xis
    assert envs_num > 2
    x2_ebd = ebd(x2).view(-1, 1) - 1
    x1_ebd = ebd(x1).view(-1, 1) - 1
    x0_ebd = ebd(x0).view(-1, 1) - 1
    x01_ebd = (x0_ebd-x1_ebd)[:, None]
    x12_ebd = (x1_ebd-x2_ebd)[:, None]
    x12_ebd_logit = strctnet(x01_ebd)
    return 10**13 * (x12_ebd_logit - x12_ebd).pow(2).mean()


def make_environment(images, labels, e):
    # 2x subsample for computational convenience
    images = images.reshape((-1, 28, 28))[:, ::2, ::2]
    # Assign a binary label based on the digit; flip label with probability 0.25
    labels = (labels < 5).float()
    labels = torch_xor(labels, torch_bernoulli(0.25, len(labels)))
    # Assign a color based on the label; flip the color with probability e
    color_mask = torch_bernoulli(e, len(labels))
    colors = torch_xor(labels, color_mask)
    # colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
    # Apply the color to the image by zeroing out the other color channel
    images = torch.stack([images, images], dim=1)
    images[torch.tensor(range(len(images))), (1-colors).long(), :, :] *= 0
    return {
      'images': (images.float() / 255.),
      'labels': labels[:, None],
      'color': (1- color_mask[:, None])
    }


def make_environment_fullcolor(images, labels, sp_ratio, noise_ratio):
    colors = [(1, 1, 0), (1, 0, 1), (0, 1, 1),
                (1, 0, 0), (0, 1, 0), (1, 0.5, 0),
                (0, 0, 1), (1, 1, 1),
                (0, 0.4, 0.8), (0.8,0,0.4)]
    images = images.reshape((-1, 28, 28))[:, ::2, ::2]
    images = torch.stack([images, images, images], dim=1).long()
    NUM_CLASSES = 10
    assert len(colors) == NUM_CLASSES
    sp_list = []
    ln_list = []
    for i in range( images.shape[0]): #
        if np.random.random() < noise_ratio: # 0.25
            label_ = np.random.choice([
            x for x in list(range(NUM_CLASSES))
            if x != labels[i]])
            ln = 0
        else:
            label_ = labels[i]
            ln = 1
        ln_list.append(ln) # label noise
        if np.random.random() < sp_ratio: # 0.1
            color_ = np.random.choice([
            x for x in list(range(NUM_CLASSES))
            if x != label_])
            sp = 0
        else:
            color_ = label_
            sp = 1
        sp_list.append(sp)
        bc = torch.Tensor(colors[torch.tensor(color_).long()])[:, None, None]
        images[i] = (images[i] * bc).long().clone()
        labels[i] = torch.tensor(label_).long().clone()
    return {
      'images': (images.float() / 255.),
      'labels': labels[:, None],
      'color': torch.Tensor(sp_list)[:, None]
    }


def make_fullmnist_envs(flags):
    mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
    mnist_train = (mnist.data[:flags.data_num], mnist.targets[:flags.data_num])
    mnist_val = (mnist.data[flags.data_num:], mnist.targets[flags.data_num:])
    rng_state = np.random.get_state()
    np.random.shuffle(mnist_train[0].numpy())
    np.random.set_state(rng_state)
    np.random.shuffle(mnist_train[1].numpy())
    # Build environments
    sp_ratio_list = [ 1- float(x) for x in flags.cons_ratio.split("_")]
    envs_num = len(sp_ratio_list) - 1
    envs = []
    for i in range(envs_num):
        envs.append(
          make_environment_fullcolor(
              mnist_train[0][i::envs_num],
              mnist_train[1][i::envs_num],
              sp_ratio=sp_ratio_list[i],
              noise_ratio=flags.noise_ratio))
    envs.append(
        make_environment_fullcolor(
            mnist_val[0],
            mnist_val[1],
            sp_ratio=sp_ratio_list[-1],
            noise_ratio=flags.noise_ratio))
    return envs

def make_mnist_envs(flags):
    mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
    mnist_train = (mnist.data[:flags.data_num], mnist.targets[:flags.data_num])
    mnist_val = (mnist.data[flags.data_num:], mnist.targets[flags.data_num:])
    rng_state = np.random.get_state()
    np.random.shuffle(mnist_train[0].numpy())
    np.random.set_state(rng_state)
    np.random.shuffle(mnist_train[1].numpy())
    # Build environments
    envs_num = flags.envs_num
    envs = []
    if flags.env_type == "linear":
        for i in range(envs_num):
            envs.append(
              make_environment(
                  mnist_train[0][i::envs_num],
                  mnist_train[1][i::envs_num],
                  (0.2 - 0.1)/(envs_num-1) * i + 0.1))
    elif flags.env_type == "sin":
        for i in range(envs_num):
            envs.append(
                make_environment(mnist_train[0][i::envs_num], mnist_train[1][i::envs_num], (0.2 - 0.1) * math.sin(i * 2.0 * math.pi / (envs_num-1)) * i + 0.1))
    elif flags.env_type == "step":
        lower_coef = 0.1
        upper_coef = 0.2
        env_per_group = flags.envs_num // 2
        for i in range(envs_num):
            env_coef = lower_coef if i < env_per_group else upper_coef
            envs.append(
                make_environment(
                    mnist_train[0][i::envs_num],
                    mnist_train[1][i::envs_num],
                    env_coef))
    else:
        raise Exception
    envs.append(make_environment(mnist_val[0], mnist_val[1], 0.9))
    return envs

def make_one_logit(num, sp_ratio, dim_inv, dim_spu):
    cc = CowCamels(
        dim_inv=dim_inv, dim_spu=dim_spu, n_envs=1,
        p=[sp_ratio], s= [0.5])
    inputs, outputs, colors, inv_noise= cc.sample(
        n=num, env="E0")
    return {
        'images': inputs,
        'labels': outputs,
        'color': colors[:, None]
    }

def make_one_reg(num, sp_cond, inv_cond, dim_inv, dim_spu):
    ar = AntiReg(
        dim_inv=dim_inv, dim_spu=dim_spu, n_envs=1,
        s=[sp_cond], inv= [inv_cond])
    inputs, outputs, colors, inv_noise= ar.sample(
        n=num, env="E0")
    return {
        'images': inputs,
        'labels': outputs,
        'color': colors,
        'noise': None,
    }

def make_logit_envs(total_num, flags):
    envs_num = flags.envs_num
    envs = []
    if flags.env_type == "linear":
        lower_coef = 0.8
        upper_coef = 0.9
        for i in range(envs_num):
            envs.append(
                make_one_logit(
                    total_num // envs_num,
                    (upper_coef - lower_coef)/(envs_num-1) * i + lower_coef,
                    flags.dim_inv,
                    flags.dim_spu))
    elif flags.env_type == "cos":
        lower_coef = 0.8
        upper_coef = 0.9
        for i in range(envs_num):
            envs.append(
                make_one_logit(
                    total_num // envs_num,
                    (upper_coef - lower_coef) * math.cos(i * 2.0 * math.pi / envs_num) + lower_coef,
                    flags.dim_inv,
                    flags.dim_spu))
    elif flags.env_type == "sin":
        lower_coef = 0.8
        upper_coef = 0.9
        for i in range(envs_num):
            envs.append(
                make_one_logit(
                    total_num // envs_num,
                    (upper_coef - lower_coef) * math.sin(i * 2.0 * math.pi / envs_num) + lower_coef,
                    flags.dim_inv,
                    flags.dim_spu))
    elif flags.env_type == "2cos":
        lower_coef = 0.8
        upper_coef = 0.9
        for i in range(envs_num):
            envs.append(
                make_one_logit(
                    total_num // envs_num,
                    (upper_coef - lower_coef) * math.cos(i * 4.0 * math.pi / envs_num) + lower_coef,
                    flags.dim_inv,
                    flags.dim_spu))
    elif flags.env_type == "2sin":
        lower_coef = 0.8
        upper_coef = 0.9
        for i in range(envs_num):
            envs.append(
                make_one_logit(
                    total_num // envs_num,
                    (upper_coef - lower_coef) * math.sin(i * 4.0 * math.pi / envs_num) + lower_coef,
                    flags.dim_inv,
                    flags.dim_spu))
    else:
        raise Exception
    envs.append(make_one_logit(total_num, 0.1, flags.dim_inv, flags.dim_spu))
    return envs

def make_reg_envs(total_num, flags):
    envs_num = flags.envs_num
    envs = []
    sp_ratio_list = [float(x) for x in flags.cons_ratio.split("_")]
    if flags.env_type == "linear":
        upper_coef = sp_ratio_list[0]
        lower_coef = sp_ratio_list[1]
        inv_cond = 1.0
        for i in range(envs_num):
            envs.append(
                make_one_reg(
                    total_num // envs_num,
                    (upper_coef - lower_coef)/(envs_num-1) * i + lower_coef,
                    inv_cond,
                    flags.dim_inv,
                    flags.dim_spu))
    else:
        raise Exception
    envs.append(make_one_reg(total_num, sp_ratio_list[-1], inv_cond, flags.dim_inv, flags.dim_spu))
    return envs


def mean_nll_class(logits, y):
    return nn.functional.binary_cross_entropy_with_logits(logits, y)

def mean_nll_multi_class(logits, y):
    nll = nn.CrossEntropyLoss()
    return nll(logits, y.view(-1).long())

def mean_nll_reg(logits, y):
    l2loss = nn.MSELoss()
    return l2loss(logits, y)

def mean_accuracy_reg(logits, y, colors=None):
    return mean_nll_reg(logits, y)


def pretty_print(*values):
    col_width = 13
    def format_val(v):
        if not isinstance(v, str):
            v = np.array2string(v, precision=5, floatmode='fixed')
        return v.ljust(col_width)
    str_values = [format_val(v) for v in values]
    print("   ".join(str_values))


class LYDataProvider(object):
    def __init__(self):
        pass

    def preprocess_data(self):
        pass

    def fetch_train(self):
        pass

    def fetch_test(self):
        pass

class IRMDataProvider(LYDataProvider):
    def __init__(self, flags):
        super(IRMDataProvider, self).__init__()

    def preprocess_data(self):
        self.train_x, self.train_y, self.train_g, self.train_c= concat_envs(self.envs[:-1])
        self.test_x, self.test_y, self.test_g, self.test_c= concat_envs(self.envs[-1:])

    def fetch_train(self):
        return self.train_x, self.train_y, self.train_g, self.train_c

    def fetch_test(self):
        return self.test_x, self.test_y, self.test_g, self.test_c

class CMNIST_LYDP(IRMDataProvider):
    def __init__(self, flags):
        super(CMNIST_LYDP, self).__init__(flags)
        self.flags = flags
        self.envs = make_mnist_envs(flags)
        self.preprocess_data()

class CMNISTFULL_LYDP(IRMDataProvider):
    def __init__(self, flags):
        super(CMNISTFULL_LYDP, self).__init__(flags)
        self.flags = flags
        self.envs = make_fullmnist_envs(flags)
        self.preprocess_data()

class LOGIT_LYDP(IRMDataProvider):
    def __init__(self, flags):
        super(LOGIT_LYDP, self).__init__(flags)
        self.flags = flags
        self.envs = make_logit_envs(flags.data_num, flags)
        self.preprocess_data()

class REG_LYDP(IRMDataProvider):
    def __init__(self, flags):
        super(REG_LYDP, self).__init__(flags)
        self.flags = flags
        self.envs = make_reg_envs(flags.data_num, flags)
        self.preprocess_data()

class CIFAR_LYPD(LYDataProvider):
    def __init__(self, flags):
        super(CIFAR_LYPD, self).__init__()
        self.flags = flags
        np.random.seed(flags.seed)
        random.seed(1) # Fix the random seed of dataset
        self.preprocess_data()

    def preprocess_data(self):
        train_num=10000
        test_num=1000 #1800
        cons_list = [0.999,0.7,0.1]
        train_envs = len(cons_list) - 1
        ratio_list = [1. / train_envs] * (train_envs)
        spd, self.train_loader, self.val_loader, self.test_loader, self.train_data, self.val_data, self.test_data = get_data_loader_cifarminst(
            batch_size=self.flags.batch_size,
            train_num=train_num,
            test_num=test_num,
            cons_ratios=cons_list,
            train_envs_ratio=ratio_list,
            label_noise_ratio=0.1,
            color_spurious=False,
            transform_data_to_standard=0,
            oracle=0)
        self.train_loader_iter = iter(self.train_loader)

    def fetch_train(self):
        try:
            batch_data = self.train_loader_iter.__next__()
        except:
            self.train_loader_iter = iter(self.train_loader)
            batch_data = self.train_loader_iter.__next__()
        batch_data = tuple(t.cuda() for t in batch_data)
        x, y, g, sp = batch_data
        return x, y.float().cuda(), g, sp

    def fetch_test(self):
        ds = self.test_data.val_dataset
        batch = ds.x_array, ds.y_array, ds.env_array, ds.sp_array
        batch = tuple(
            torch.Tensor(t).cuda()
            for t in batch)
        x, y, g, sp = batch
        return x, y.float(), g, sp

    def test_batchs(self):
        return math.ceil(self.test_data.val_dataset.x_array.shape[0] / self.flags.batch_size)

    # def train_batchs(self):
    #     return math.ceil(self.train_dataset.x_array.shape[0] / self.flags.batch_size)



class COCOcolor_LYPD(LYDataProvider):
    def __init__(self, flags):
        super(COCOcolor_LYPD, self).__init__()
        self.flags = flags
        self.preprocess_data()

    def preprocess_data(self):
        sp_ratio_list = [float(x) for x in self.flags.cons_ratio.split("_")]
        self.train_dataset, self.test_dataset = get_spcoco_dataset(
            sp_ratio_list=sp_ratio_list,
            noise_ratio=self.flags.noise_ratio,
            num_classes=2,
            flags=self.flags)
        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.flags.batch_size,
            shuffle=False,
            num_workers=4)
        self.test_loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.flags.batch_size,
            shuffle=False,
            num_workers=4)
        self.train_loader_iter = iter(self.train_loader)
        self.test_loader_iter = iter(self.test_loader)

    def fetch_train(self):
        try:
            batch_data = self.train_loader_iter.__next__()
        except:
            self.train_loader_iter = iter(self.train_loader)
            batch_data = self.train_loader_iter.__next__()
        batch_data = tuple(t.cuda() for t in batch_data)
        x, y, g, sp = batch_data
        return x, y.float().cuda(), g, sp

    def fetch_test(self):
        ds = self.test_dataset
        batch = ds.x_array, ds.y_array, ds.env_array, ds.sp_array
        batch = tuple(
            torch.Tensor(t).cuda()
            for t in batch)
        x, y, g, sp = batch
        return x, y.float(), g, sp

    def test_batchs(self):
        return math.ceil(self.test_dataset.x_array.shape[0] / self.flags.batch_size)

    def train_batchs(self):
        return math.ceil(self.train_dataset.x_array.shape[0] / self.flags.batch_size)

    def fetch_test_batch(self):
        try:
            batch_data = self.test_loader_iter.__next__()
        except:
            self.test_loader_iter = iter(self.test_loader)
            batch_data = self.test_loader_iter.__next__()
        batch_data = tuple(t.cuda() for t in batch_data)
        x, y, g, sp = batch_data
        return x, y.float().cuda(), g, sp
