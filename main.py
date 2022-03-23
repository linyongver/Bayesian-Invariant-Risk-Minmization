import argparse
import random
import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import datasets
import pandas as pd
# import wandb
import os
import sys
from torch import nn, optim, autograd
from model import EBD
from model import resnet18_sepfc_us
from model import MLP

sys.path.append('dataset_scripts')
from utils import concat_envs,eval_acc_class,eval_acc_reg,mean_nll_class,mean_accuracy_class,mean_nll_reg,mean_accuracy_reg,pretty_print, return_model
from utils import CMNIST_LYDP
from utils import CIFAR_LYPD, COCOcolor_LYPD
from utils import mean_nll_multi_class,eval_acc_multi_class,mean_accuracy_multi_class


parser = argparse.ArgumentParser(description='Colored MNIST')
parser.add_argument('--envs_num', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--dataset', type=str, default="CMNIST", choices=["CifarMnist","ColoredObject", "CMNIST"])
parser.add_argument('--opt', type=str, default="adam", choices=["adam", "sgd"])
parser.add_argument('--l2_regularizer_weight', type=float,default=0.001)
parser.add_argument('--print_every', type=int,default=100)
parser.add_argument('--prior_sd_coef', type=float,default=50)
parser.add_argument('--data_num', type=int, default=2000)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--env_type', default="linear", type=str, choices=["2_group", "cos", "linear"])
parser.add_argument('--irm_type', default="birm", type=str, choices=["birm", "irmv1", "erm"])
parser.add_argument('--n_restarts', type=int, default=1)
parser.add_argument('--image_scale', type=int, default=64)
parser.add_argument('--hidden_dim', type=int, default=16)
parser.add_argument('--cons_ratio', type=str, default="0.999_0.7_0.1")
parser.add_argument('--noise_ratio', type=float, default=0)
parser.add_argument('--step_gamma', type=float, default=0.1)
parser.add_argument('--penalty_anneal_iters', type=int, default=200)
parser.add_argument('--penalty_weight', type=float, default=10000.0)
parser.add_argument('--steps', type=int, default=501)
parser.add_argument('--grayscale_model', type=int, default=0)
flags = parser.parse_args()
irm_type = flags.irm_type

torch.manual_seed(flags.seed)
np.random.seed(flags.seed)
random.seed(1) # Fix the random seed of dataset
# Because random package is used to generate CifarMnist dataset
# We fix the randomness of the dataset.


final_train_accs = []
final_test_accs = []
flags, model_type = return_model(flags)
for restart in range(flags.n_restarts):

    if flags.dataset == "CMNIST":
        dp = CMNIST_LYDP(flags)
        test_batch_num = 1
        test_batch_fetcher = dp.fetch_test
        mlp = MLP(flags).cuda()
        mean_nll = mean_nll_class
        mean_accuracy = mean_accuracy_class
        eval_acc = eval_acc_class
        flags.env_type = "linear"
    elif flags.dataset == "CifarMnist":
        dp = CIFAR_LYPD(flags)
        test_batch_num = 1
        test_batch_fetcher = dp.fetch_test
        mlp = resnet18_sepfc_us(
            pretrained=False,
            num_classes=1).cuda()
        mean_nll = mean_nll_class
        mean_accuracy = mean_accuracy_class
        eval_acc = eval_acc_class
    elif flags.dataset == "ColoredObject":
        dp = COCOcolor_LYPD(flags)
        test_batch_num = dp.test_batchs()
        test_batch_fetcher = dp.fetch_test_batch
        mlp = resnet18_sepfc_us(
            pretrained=False,
            num_classes=1).cuda()
        mean_nll = mean_nll_class
        mean_accuracy = mean_accuracy_class
        eval_acc = eval_acc_class
    else:
        raise Exception
    if flags.opt == "adam":
        optimizer = optim.Adam(
          mlp.parameters(),
          lr=flags.lr)
    elif flags.opt == "sgd":
        optimizer = optim.SGD(
          mlp.parameters(),
          momentum=0.9,
          lr=flags.lr)
    else:
        raise Exception

    ebd = EBD(flags).cuda()
    lr_schd = lr_scheduler.StepLR(
        optimizer,
        step_size=int(flags.steps/2),
        gamma=flags.step_gamma)

    pretty_print('step', 'train loss', 'train penalty', 'test acc')
    if flags.irm_type == "cirm_sep":
        pred_env_haty_sep.init_sep_by_share(pred_env_haty)
    for step in range(flags.steps):
        mlp.train()
        train_x, train_y, train_g, train_c= dp.fetch_train()
        if model_type == "irmv1":
            train_logits = ebd(train_g).view(-1, 1) * mlp(train_x)
            train_nll = mean_nll(train_logits, train_y)
            grad = autograd.grad(
                train_nll * flags.envs_num, ebd.parameters(),
                create_graph=True)[0]
            train_penalty =  torch.mean(grad**2)
        elif model_type == "irmv1b":
            e1 = (train_g == 0).view(-1).nonzero().view(-1)
            e2 = (train_g == 1).view(-1).nonzero().view(-1)
            e1 = e1[torch.randperm(len(e1))]
            e2 = e2[torch.randperm(len(e2))]
            s1 = torch.cat([e1[::2], e2[::2]])
            s2 = torch.cat([e1[1::2], e2[1::2]])
            train_logits = ebd(train_g).view(-1, 1) * mlp(train_x)

            train_nll1 = mean_nll(train_logits[s1], train_y[s1])
            train_nll2 = mean_nll(train_logits[s2], train_y[s2])
            train_nll = train_nll1 + train_nll2
            grad1 = autograd.grad(
                train_nll1 * flags.envs_num, ebd.parameters(),
                create_graph=True)[0]
            grad2 = autograd.grad(
                train_nll2 * flags.envs_num, ebd.parameters(),
                create_graph=True)[0]
            train_penalty = torch.mean(grad1 * grad2)
        elif model_type == "bayes_variance":
            sampleN = 10
            train_penalty = 0
            train_logits = mlp(train_x)
            train_nll = mean_nll(train_logits, train_y)
            for i in range(sampleN):
                ebd.re_init_with_noise(flags.prior_sd_coef/flags.data_num)
                loss_list = []
                for i in range(int(train_g.max())+1):
                    ei = (train_g == i).view(-1)
                    ey = train_y[ei]
                    el= train_logits[ei]
                    enll = mean_nll(el, ey)
                    loss_list.append(enll)
                loss_t = torch.stack(loss_list)
                train_penalty0 = ((loss_t - loss_t.mean())** 2).mean()
                train_penalty +=  1/sampleN * train_penalty0
        elif model_type == "bayes_fullbatch":
            sampleN = 10
            train_penalty = 0
            train_logits = mlp(train_x)
            for i in range(sampleN):
                ebd.re_init_with_noise(flags.prior_sd_coef/flags.data_num)
                train_logits_w = ebd(train_g).view(-1, 1)*train_logits
                train_nll = mean_nll(train_logits_w, train_y)
                grad = autograd.grad(
                    train_nll * flags.envs_num, ebd.parameters(),
                    create_graph=True)[0]
                train_penalty +=  1/sampleN * torch.mean(grad**2)
        elif model_type == "bayes_batch":
            sampleN = 10
            train_penalty = 0
            train_logits = mlp(train_x)
            e1 = (train_g == 0).view(-1).nonzero().view(-1)
            e2 = (train_g == 1).view(-1).nonzero().view(-1)
            e1 = e1[torch.randperm(len(e1))]
            e2 = e2[torch.randperm(len(e2))]
            s1 = torch.cat([e1[::2], e2[::2]])
            s2 = torch.cat([e1[1::2], e2[1::2]])
            train_nll = mean_nll(train_logits, train_y)
            for i in range(sampleN):
                ebd.re_init_with_noise(flags.prior_sd_coef/flags.data_num)
                train_logits_w1 = ebd(train_g[s1]).view(-1, 1)*train_logits[s1]
                train_logits_w2 = ebd(train_g[s2]).view(-1, 1)*train_logits[s2]
                train_nll1 = mean_nll(train_logits_w1, train_y[s1])
                train_nll2 = mean_nll(train_logits_w2, train_y[s2])
                grad1 = autograd.grad(
                    train_nll1 * flags.envs_num, ebd.parameters(),
                    create_graph=True)[0]
                grad2 = autograd.grad(
                    train_nll2 * flags.envs_num, ebd.parameters(),
                    create_graph=True)[0]
                train_penalty +=  1./sampleN * torch.mean(grad1*grad2)
        elif irm_type == "erm":
            train_logits = mlp(train_x)
            train_nll = mean_nll(train_logits, train_y)
            train_penalty = torch.tensor(0.0)
        else:
            raise Exception
        train_acc, train_minacc, train_majacc = eval_acc(train_logits, train_y, train_c)
        weight_norm = torch.tensor(0.).cuda()
        for w in mlp.parameters():
            weight_norm += w.norm().pow(2)

        loss = train_nll.clone()
        loss += flags.l2_regularizer_weight * weight_norm
        penalty_weight = (flags.penalty_weight
            if step >= flags.penalty_anneal_iters else 0.0)
        loss += penalty_weight * train_penalty
        if penalty_weight > 1.0:
          loss /= (1. + penalty_weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_schd.step()

        if step % flags.print_every == 0:
            if flags.dataset != 'CifarMnist':
                mlp.eval()
            test_acc_list = []
            test_minacc_list = []
            test_majacc_list = []
            data_num = []
            for ii in range(test_batch_num):
                test_x, test_y, test_g, test_c= test_batch_fetcher()
                test_logits = mlp(test_x)
                test_acc_, test_minacc_, test_majacc_ = eval_acc(test_logits, test_y, test_c)
                test_acc_list.append(test_acc_ * test_x.shape[0])
                test_minacc_list.append(test_minacc_ * test_x.shape[0])
                test_majacc_list.append(test_majacc_ * test_x.shape[0])
                data_num.append(test_x.shape[0])
            total_data = torch.Tensor(data_num).sum()
            test_acc, test_minacc, test_majacc = torch.Tensor(test_acc_list).sum()/total_data, torch.Tensor(test_minacc_list).sum()/total_data, torch.Tensor(test_majacc_list).sum()/total_data
            pretty_print(
                np.int32(step),
                loss.detach().cpu().numpy(),
                train_penalty.detach().cpu().numpy(),
                test_acc.detach().cpu().numpy(),
            )
    final_train_accs.append(train_acc.detach().cpu().numpy())
    final_test_accs.append(test_acc.detach().cpu().numpy())
    print('Final test acc: %s' % np.mean(final_test_accs))
