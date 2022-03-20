import pandas as pd
import os
import numpy as np

def args2header(args, default_dict=None, exclude_names=[]):
    from datetime import datetime
    time_header = datetime.strftime(
        datetime.now(),
        "%H:%M:%S")
    date_header = datetime.strftime(
        datetime.now(),
        "%Y%m%d")
    exDict = vars(args)
    loggers = [date_header, time_header]
    def convert_args_to_logger(k):
        return ("".join([x[:2] for x in k.split("_")])).upper()
    for k, v in exDict.items():
        if default_dict is not None:
            if k in default_dict.keys():
                if default_dict[k] == v:
                    continue
            if k in exclude_names:
                continue
        if k != "sfx":
            loggers.append(
                "%s:%s" %(convert_args_to_logger(k), v))
    if "sfx" in exDict.keys():
        loggers.append(
                "sfx:%s" %(exDict["sfx"]))
    return "_".join(loggers)

def save_args(args, logger_path):
    import json
    exDict = vars(args)
    with open(logger_path+ "/args", 'w') as f:
        f.write(json.dumps(exDict))

class LYCSVLogger(object):
    def __init__(self, csv_path, mode="w", log_every=10):
        self.csv_path = csv_path
        self.mode = mode
        self.states_list = []
        self.log_every = log_every
        self.count = 0

    def log(self, epoch, batch, stats_dict, restart=None):
        self.count += 1
        if "epoch" not in stats_dict:
            stats_dict.update({"epoch": epoch})
        if restart is not None:
            if "restart" not in stats_dict:
                stats_dict.update(
                    {"restart": restart})
        if "batch" not in stats_dict:
            stats_dict.update({"batch": batch})
        self.states_list.append(stats_dict)
        if self.count % self.log_every == 0:
            self.form_and_output()

    def form_and_output(self):
        self.stats_df = pd.DataFrame(self.states_list)
        self.stats_df.to_csv(self.csv_path, index=False)

    def close(self):
        self.form_and_output()

class LYCSVStepLogger(object):
    def __init__(self, csv_path=None, save_interval=20):
        if csv_path is not None:
            self.set_path(csv_path)
        self.states_list = []
        self.save_interval = save_interval
        self.log_count = 0

    def set_path(self, csv_path):
        self.csv_path = csv_path

    def log(self, stats_dict):
        self.states_list.append(stats_dict)
        self.log_count += 1
        if self.log_count % self.save_interval == 0:
            self.form_and_out_df()

    def form_and_out_df(self):
        self.stats_df = pd.DataFrame(self.states_list)
        self.stats_df.to_csv(self.csv_path, index=False)

    def close(self):
        self.form_and_out_df()
        del self.stats_df
        del self.states_list


class EpochStat(object):
    def __init__(self, sfx):
        self.attrs = []
        self.sfx = sfx

    def update_stats(self, stats):
        if isinstance(stats, list):
            for es in stats:
                self.attrs.append(es)
        elif isinstance(stats, dict):
            self.attrs.append(stats)
        else:
            raise Exception

    def get_summary(self):
        summary_dict = {}
        full_df = pd.DataFrame(self.attrs)
        self.envs = list(np.unique(full_df.env))
        for ie in self.envs:
            edf = full_df[full_df.env == ie]
            summary_dict[ie] = dict(edf.mean())
        self.summary_dict = summary_dict
        return summary_dict

    def get_log_summary(self):
        self.get_summary()
        log_dict = {}
        for ie in self.envs:
            env_dict = self.summary_dict[ie]
            log_dict["avg_acc_group:%s"%ie] = env_dict["acc"]
            log_dict["avg_loss_group:%s"%ie] = env_dict["loss"]
            try:
                log_dict["penalty:%s"%ie] = env_dict["penalty"]
                log_dict["total_loss:%s"%ie] = env_dict["loss"]
                log_dict["opt_loss:%s"%ie] = env_dict["main_loss"]
            except:
                pass
        self.log_dict = log_dict
        return log_dict


    def echo(self):
        summary_dict = self.get_summary()
        echo_str = [self.sfx + "\n"]
        for ienv in range(len(self.envs)):
            env = self.envs[ienv]
            echo_str += ["Env%s"%env]
            for k,v in summary_dict[env].items():
                if k != "env" and not np.isnan(v):
                    echo_str += ["%s=%.4f"%(k, v)]
            if ienv < len(self.envs) - 1:
                echo_str += ["\n"]
        print(" ".join(echo_str))

def env_stat(x, outputs, y, g, model, criterion):
    env_stats = []
    for ie in range(4):
        eindex = (g == ie)
        if eindex.sum() > 0:
            ex = x[eindex]
            ey = y[eindex]
            env_stats.append(
            {"env": ie,
            "loss": criterion(
                outputs[eindex].view(-1),
                ey.float()).item(),
            "acc": mean_accuracy(
                outputs[eindex].view(-1),
                ey.float()).item()})
    return env_stats

def save_cmd(sys_argv, logger_path): # sys.argv
    with open(os.path.join(logger_path, "cmd.tex"), "w") as f:
        f.write('python ' + ' '.join(sys_argv))
        print("outputing cmd to cmd.txt")

