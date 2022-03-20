# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import datasets
import pandas as pd
# import wandb
import pdb
import os
import sys
from torch import nn, optim, autograd
from model import ENV_EBD, PredYEnvHatY
from model import EBD
from model import resnet18_sepfc_us
from model import BayesW
from model import Y_EBD, PredEnvHatY, PredEnvHatYSep, PredEnvYY
from model import InferEnv
from model import MLP, MLPFull

# sys.path.append('/home/ylindf/projects/tools')
# sys.path.append('/home/ylindf/projects/invariant/CMNIST/common')
from utils import concat_envs,eval_acc_class,eval_acc_reg,mean_nll_class,mean_accuracy_class,mean_nll_reg,mean_accuracy_reg,pretty_print
from utils import LOGIT_LYDP, REG_LYDP, CMNIST_LYDP
from utils import CIFAR_LYPD, COCOcolor_LYPD
from utils import CMNISTFULL_LYDP
from utils import mean_nll_multi_class,eval_acc_multi_class,mean_accuracy_multi_class
from global_utils import args2header, save_args, save_cmd, LYCSVLogger



parser = argparse.ArgumentParser(description='Colored MNIST')
parser.add_argument('--envs_num', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--classes_num', type=int, default=2)
parser.add_argument('--dataset', type=str, default="mnist", choices=["cifar","coco_color", "mnist","mnistfull", "logit", "reg"])
parser.add_argument('--opt', type=str, default="adam", choices=["adam", "sgd"])
parser.add_argument('--l2_regularizer_weight', type=float,default=0.001)
parser.add_argument('--print_every', type=int,default=100)
parser.add_argument('--prior_sd_coef', type=float,default=50)
parser.add_argument('--dim_inv', type=int, default=2)
parser.add_argument('--variance_gamma', type=float, default=1.0)
parser.add_argument('--data_num', type=int, default=2000)
parser.add_argument('--dim_spu', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--sfx', default="adhfoadf", type=str)
parser.add_argument('--env_type', default="linear", type=str, choices=["2_group", "cos", "linear"])
parser.add_argument('--irm_type', default="irmv1", type=str, choices=["bayes_irmv1b", "bayes_rex", "irmv1b","bayes_irmv1", "rex", "rvp", "erm","irmv1", "invrat", "irmgame"])
parser.add_argument('--n_restarts', type=int, default=10)
parser.add_argument('--image_scale', type=int, default=64)
parser.add_argument('--hidden_dim', type=int, default=16)
parser.add_argument('--cons_ratio', type=str, default="0.999_0.7_0.1")
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--noise_ratio', type=float, default=0)
parser.add_argument('--step_gamma', type=float, default=0.1)
parser.add_argument('--step_round', type=int, default=3)
parser.add_argument('--inner_steps', type=int, default=1)
parser.add_argument('--penalty_anneal_iters', type=int, default=200)
parser.add_argument('--penalty_weight', type=float, default=10000.0)
parser.add_argument('--steps', type=int, default=501)
parser.add_argument('--grayscale_model', type=int, default=0)
flags = parser.parse_args()
print("batch_size is", flags.batch_size)
irm_type = flags.irm_type

torch.manual_seed(flags.seed)
np.random.seed(flags.seed)


default_dict = {"inner_steps": 1, "step_round":3, "step_gamma":0.1, "hidden_dim":390,  "data_num":3400, "grayscale_model": False, "l2_regularizer_weight":0.001, "penalty_anneal_iters":200, "lr": 0.0004, "steps":1500, "envs_num":2, "dim_inv":2, "penalty_weight":10000, "cons_ratio": "0.9_0.8_0.1", "noise_ratio":0.25}
exclude_names = [
    "print_every",
    "variance_gamma",
    "dim_inv",
    "dim_spu",
    "env_type",
    "data_num",
    "hidden_dim",
    "step_round",
    "inner_steps",
    "grayscale_model"
]

logger_key= args2header(
    flags, default_dict=default_dict, exclude_names=exclude_names)
print('Flags:')

# wandb.init(project=flags.dataset + "_bayes", name=logger_key)
# wandb.config.update(flags)
logger_path = "logs/%s" % logger_key
if not os.path.exists(logger_path):
    os.makedirs(logger_path)
save_args(flags, logger_path)
save_cmd(sys.argv, logger_path)
mode='w'
csv_logger = LYCSVLogger(os.path.join(logger_path, 'res.csv'),  mode=mode)

for k,v in sorted(vars(flags).items()):
    print("\t{}: {}".format(k, v))


final_train_accs = []
final_test_accs = []
for restart in range(flags.n_restarts):
    print("Restart", restart)

    if flags.dataset == "mnist":
        dp = CMNIST_LYDP(flags)
        test_batch_num = 1
        test_batch_fetcher = dp.fetch_test
        mlp = MLP(flags).cuda()
        assert flags.num_classes == 2
        mean_nll = mean_nll_class
        mean_accuracy = mean_accuracy_class
        eval_acc = eval_acc_class
        # else:
        # mean_nll = mean_nll_multi_class
        # mean_accuracy = mean_accuracy_multi_class
        # eval_acc = eval_acc_multi_class
    elif flags.dataset == "mnistfull":
        dp = CMNISTFULL_LYDP(flags)
        test_batch_num = 1
        test_batch_fetcher = dp.fetch_test
        mlp = MLPFull(flags).cuda()
        assert flags.num_classes == 10
        mean_nll = mean_nll_multi_class
        mean_accuracy = mean_accuracy_multi_class
        eval_acc = eval_acc_multi_class
    elif flags.dataset == "cifar":
        dp = CIFAR_LYPD(flags)
        test_batch_num = 1
        test_batch_fetcher = dp.fetch_test
        mlp = resnet18_sepfc_us(
            pretrained=False,
            num_classes=1).cuda()
        mean_nll = mean_nll_class
        mean_accuracy = mean_accuracy_class
        eval_acc = eval_acc_class
    elif flags.dataset == "coco_color":
        dp = COCOcolor_LYPD(flags)
        test_batch_num = dp.test_batchs()
        test_batch_fetcher = dp.fetch_test_batch
        if flags.num_classes == 2:
            mlp = resnet18_sepfc_us(
                pretrained=False,
                num_classes=1).cuda()
            mean_nll = mean_nll_class
            mean_accuracy = mean_accuracy_class
            eval_acc = eval_acc_class
        else:
            mlp = resnet18_sepfc_us(
                pretrained=False,
                num_classes=flags.num_classes).cuda()
            mean_nll = mean_nll_multi_class
            mean_accuracy = mean_accuracy_multi_class
            eval_acc = eval_acc_multi_class
    elif flags.dataset == "logit":
        dp = LOGIT_LYDP(flags)
        mlp = nn.Linear(in_features=12, out_features=1).cuda()
        mean_nll = mean_nll_class
        mean_accuracy = mean_accuracy_class
        eval_acc = eval_acc_class
    elif flags.dataset == "reg":
        dp = REG_LYDP(flags)
        test_batch_num = 1
        test_batch_fetcher = dp.fetch_test
        mlp = nn.Linear(in_features=flags.dim_inv + flags.dim_spu, out_features=1).cuda()
        mean_nll = mean_nll_reg
        mean_accuracy = mean_accuracy_reg
        eval_acc = eval_acc_reg
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
    if flags.irm_type == "invrat":
        ebd_opt = optim.SGD(
          ebd.parameters(),
          lr=flags.lr)
    if flags.irm_type == "irmgame":
        ebd1 = EBD(flags).cuda()
        ebd2 = EBD(flags).cuda()
        ebd_opt1 = optim.SGD(
          ebd.parameters(),
          lr=flags.lr)
        ebd_opt2 = optim.SGD(
          ebd.parameters(),
          lr=flags.lr)

    lr_schd = lr_scheduler.StepLR(
        optimizer,
        step_size=int(flags.steps/2),
        gamma=flags.step_gamma)

    pretty_print('step', 'train acc', 'train penalty', 'test acc', "test_minacc", "test_majacc")
    # train_x, train_y, train_g, train_c= concat_envs(envs[:-1])
    # normed_g = (train_g - train_g.mean())/train_g.std()
    if flags.irm_type == "cirm_sep":
        pred_env_haty_sep.init_sep_by_share(pred_env_haty)
    for step in range(flags.steps):
        mlp.train()
        train_x, train_y, train_g, train_c= dp.fetch_train()
        if irm_type == "irmv1":
            train_logits = ebd(train_g).view(-1, 1) * mlp(train_x)
            train_nll = mean_nll(train_logits, train_y)
            grad = autograd.grad(
                train_nll * flags.envs_num, ebd.parameters(),
                create_graph=True)[0]
            train_penalty =  torch.mean(grad**2)
        elif irm_type == "irmv1b":
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
        elif irm_type == "rex":
            loss_list = []
            train_logits = mlp(train_x)
            train_nll = 0
            for i in range(int(train_g.max())+1):
                ei = (train_g == i).view(-1)
                ey = train_y[ei]
                el= train_logits[ei]
                enll = mean_nll(el, ey)
                train_nll += enll / (train_g.max() + 1)
                loss_list.append(enll)
            loss_t = torch.stack(loss_list)
            train_penalty = ((loss_t - loss_t.mean())** 2).mean()
        elif irm_type == "bayes_rex":
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
        elif irm_type == "irmgame":
            train_logits = mlp(train_x)
            ei = (train_g == 1).view(-1)
            ey = train_y[ei]
            el= train_logits[ei]
            eg = train_g[ei]
            ebd1(eg).view(-1, 1) * el,
            enll = mean_nll(el, ey)
            loss_t = torch.stack(loss_list)
            train_penalty = ((loss_t - loss_t.mean())** 2).mean()
        elif irm_type == "rvp":
            loss_list = []
            train_logits = mlp(train_x)
            train_nll = 0
            for i in range(int(train_g.max())+1):
                ei = train_g == i
                ey = train_y[ei]
                el= train_logits[ei]
                enll = mean_nll(el, ey)
                train_nll += enll / (train_g.max() + 1)
                loss_list.append(enll)
            loss_t = torch.stack(loss_list)
            train_penalty = torch.sqrt(((loss_t - loss_t.mean())** 2).mean())
        elif flags.irm_type == "bayes_irmv1":
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
        elif flags.irm_type == "bayes_irmv1b":
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
                # if flags.num_classes == 2:
                train_logits_w1 = ebd(train_g[s1]).view(-1, 1)*train_logits[s1]
                train_logits_w2 = ebd(train_g[s2]).view(-1, 1)*train_logits[s2]
                # else:
                # train_logits_w1 = ebd(train_g[s1]).view(-1, flags.num_classes)*train_logits[s1]
                # train_logits_w2 = ebd(train_g[s2]).view(-1, flags.num_classes)*train_logits[s2]
                train_nll1 = mean_nll(train_logits_w1, train_y[s1])
                train_nll2 = mean_nll(train_logits_w2, train_y[s2])
                grad1 = autograd.grad(
                    train_nll1 * flags.envs_num, ebd.parameters(),
                    create_graph=True)[0]
                grad2 = autograd.grad(
                    train_nll2 * flags.envs_num, ebd.parameters(),
                    create_graph=True)[0]
                train_penalty +=  1./sampleN * torch.mean(grad1*grad2)
        elif irm_type == "invrat":
            # knowing env does not help predicting Y
            ebd.re_init()
            train_logits = mlp(train_x).detach()
            train_nll_sep = mean_nll(
                ebd(train_g).view(-1, 1) * train_logits,
                train_y)
            ebd_opt.zero_grad()
            train_nll_sep.backward()
            ebd_opt.step()
            # print(ebd.embedings.weight.detach().cpu().numpy())
            # calculate penalty
            train_logits = mlp(train_x)
            train_nll_sep= mean_nll(
              ebd(train_g).view(-1, 1) * train_logits,
              train_y)
            train_nll= mean_nll(
              train_logits, train_y)
            train_penalty = 4 / flags.lr * (train_nll - train_nll_sep)
        elif irm_type == "erm":
            train_logits = mlp(train_x)
            train_nll = mean_nll(train_logits, train_y)
            train_penalty = torch.tensor(0.0)
        else:
            raise Exception


        # if step % 100 == 0:
        #     train_nll0 = mean_nll(
        #         train_logits[train_g==0],
        #         train_y[train_g==0])
        #     train_nll1 = mean_nll(
        #         train_logits[train_g==1],
        #         train_y[train_g==1])
        #     # print("env loss", train_nll0.item(), train_nll1.item(), mlp.weight.detach().cpu().numpy())
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
            if flags.dataset != 'cifar':
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
                # train_nll.detach().cpu().numpy(),
                train_acc.detach().cpu().numpy(),
                train_penalty.detach().cpu().numpy(),
                test_acc.detach().cpu().numpy(),
                test_minacc.detach().cpu().numpy(),
                test_majacc.detach().cpu().numpy()
            )
            stats_dict = {
                "train_nll": train_nll.detach().cpu().numpy(),
                "train_acc": train_acc.detach().cpu().numpy(),
                "train_minacc": train_minacc.detach().cpu().numpy(),
                "train_majacc": train_majacc.detach().cpu().numpy(),
                "train_penalty": train_penalty.detach().cpu().numpy(),
                "test_acc": test_acc.detach().cpu().numpy(),
                "test_minacc": test_minacc.detach().cpu().numpy(),
                "test_majacc": test_majacc.detach().cpu().numpy(),
            }
            # if flags.dataset == "reg":
            #     weight_list = mlp.weight.view(-1).detach().cpu().numpy().tolist()
            #     weight_name = ["weight_%s"%s for s in range(len(weight_list))]
            #     stats_dict.update(dict(zip(weight_name, weight_list)))
            # wandb.log(stats_dict)

            csv_logger.log(
                epoch=step,
                batch=step,
                stats_dict=stats_dict,
                restart=restart)

    final_train_accs.append(train_acc.detach().cpu().numpy())
    final_test_accs.append(test_acc.detach().cpu().numpy())
    print('Final train acc (mean/std across restarts so far):')
    print(np.mean(final_train_accs), np.std(final_train_accs))
    print('Final test acc (mean/std across restarts so far):')
    print(np.mean(final_test_accs), np.std(final_test_accs))
csv_logger.close()
