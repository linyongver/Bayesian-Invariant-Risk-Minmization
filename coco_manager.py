from collections import defaultdict
import os
import datetime
CUDA_num = 1
cmd_list = []
sfx = "CMNIST_vframe" + datetime.datetime.now().strftime('%m%d%H%M%S')
para_dict = {
"l2_regularizer_weight": [0.001], "lr": [0.01], "hidden_dim": [390], "cons_ratio": ["0.999_0.7_0.1"], "noise_ratio": [0.05], "image_scale": [32], "batch_size": [1000], "penalty_anneal_iter": [80], "opt": ["sgd"], "print_every": [10], "penalty_weight": [10000], "steps": [800], "envs_num": [2], "n_restarts": [1], "dataset": ["coco_color"], "dim_inv": [2], "step_gamma": [0.1], "dim_spu": [2], "inner_step": [3], "irm_type": ["irmv1", "irmv1b", "bayes_rex"], "prior_sd_coef": [0, 1000], "grayscale_model": [0], "data_num": [12000], "seed": [0, 1, 2],'sfx': [sfx]
}
name_list = list(para_dict.keys())
cmd_list = ["python main.py"]

def duplicate_cmd(cmd_list, num):
    ncl = []
    for cmd in cmd_list:
        for i in range(num):
            ncl.append(cmd)
    return ncl

for iname in name_list:
    ilist = para_dict[iname]
    cmd_list = duplicate_cmd(
        cmd_list, len(ilist))
    for icmd in range(len(cmd_list)):
        value = ilist[icmd % len(ilist)]
        if isinstance(value, bool):
            if value == True:
                cmd_list[icmd] = cmd_list[icmd] + " --%s" % iname
            elif value == False:
                pass
        else:
            cmd_list[icmd] = cmd_list[icmd] + " --%s %s" % (iname, value)

import random
random.shuffle(cmd_list)
print("cmd number=%s" % len(cmd_list))
sh_dict = defaultdict(list)
for icmd in range(len(cmd_list)):
    cuda_device = icmd % CUDA_num
    sh_dict[cuda_device].append(cmd_list[icmd])

run_sh_cmds = []

dir_path = "meta_run/%s" % sfx
if not os.path.exists(dir_path):
    os.mkdir(dir_path)
import json
exDict = para_dict
exDict["time"] = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
with open(dir_path + "/args", 'w') as file:
    file.write(json.dumps(exDict))
print("args json to", dir_path + "/args")

for i in range(CUDA_num):
    sh_cmd = sh_dict[i]
    file_name = '%s/%s.sh' % (dir_path, i)
    print("sh  %s"% file_name)
    run_sh_cmds.append("sh " + file_name)
    with open(file_name, 'w') as f:
        for item in sh_cmd:
            f.write("%s\n" % item)
