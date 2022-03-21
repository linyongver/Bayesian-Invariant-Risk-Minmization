import torch
import pdb
import math
class CowCamels:
    """
    Cows and camels
    """
    def __init__(self, dim_inv, dim_spu, n_envs, p, s):
        self.scramble = torch.eye(dim_inv + dim_spu)
        self.dim_inv = dim_inv
        self.dim_spu = dim_spu
        self.dim = dim_inv + dim_spu

        self.task = "classification"
        self.envs = {}
        assert len(p) == n_envs
        assert len(s) == n_envs

        for i in range(n_envs):
            self.envs["E" + str(i)] = {"p": p[i], "s": s[i]}

        # foreground is 100x noisier than background
        self.snr_fg = 1
        self.snr_bg = 1
        # foreground (fg) denotes animal (cow / camel)
        cow = torch.ones(1, self.dim_inv)
        self.avg_fg = torch.cat((cow, cow, -cow, -cow))

        # background (bg) denotes context (grass / sand)
        grass = torch.ones(1, self.dim_spu)
        self.avg_bg = torch.cat((grass, -grass, -grass, grass))

    def sample(self, n=1000, env="E0"):
        def torch_bernoulli(p, size):
            return (torch.rand(size) < p).float()
        def torch_xor(a, b):
            return (a-b).abs()
        p = self.envs[env]["p"]
        s = self.envs[env]["s"]
        y = torch_bernoulli(s, n)
        inv_noise_ratio = 0.1
        sp_noise_ratio = 1- p
        inv_noise = torch_bernoulli(inv_noise_ratio, n)
        sp_noise = torch_bernoulli(sp_noise_ratio, n)
        inv_feature = torch_xor(y, inv_noise) * 2 - 1
        sp_feature = torch_xor(y, sp_noise) * 2 - 1
        x = torch.cat((
            (torch.randn(n, self.dim_inv) + inv_feature[:, None]) * self.snr_fg,
            (torch.randn(n, self.dim_spu) + sp_feature[:, None]) * self.snr_bg), -1)

        inputs = x @ self.scramble
        outputs = y[:, None].float()
        colors = sp_noise[:, None]

        return inputs, outputs, colors, inv_noise[:, None]

class AntiReg:
    """
    Cows and camels
    """
    def __init__(self, dim_inv, dim_spu, n_envs, s, inv):
        self.scramble = torch.eye(dim_inv + dim_spu)
        self.dim_inv = dim_inv
        self.dim_spu = dim_spu
        self.dim = dim_inv + dim_spu

        self.task = "regression"
        self.envs = {}
        assert len(s) == n_envs
        assert len(inv) == n_envs

        for i in range(n_envs):
            self.envs["E" + str(i)] = {"s": s[i], "inv": inv[i]}

    def sample(self, n=1000, env="E0"):
        sp_cond = self.envs["E0"]["s"]
        inv_cond = self.envs["E0"]["inv"]
        x = torch.randn(n, self.dim_inv)
        x_noise = torch.randn(n, 1) * inv_cond
        y = x.sum(1, keepdim=True) + x_noise
        z_noise = torch.randn(n, self.dim_spu) * sp_cond
        z = y + z_noise

        inputs = torch.cat((x, z), 1) @ self.scramble
        outputs = y.sum(1, keepdim=True)
        colors = (z_noise.abs().mean(1, keepdim=True) > z_noise.abs().mean()).float()
        inv_noise = (x_noise.abs().mean(1, keepdim=True) > x_noise.abs().mean()).float()

        return inputs, outputs, colors, inv_noise


if __name__ == "__main__":
    exp2 = CowCamels(dim_inv=2, dim_spu=10, n_envs=3, 
                     p=[0.97, 0.9, 0.1], s= [0.5, 0.5, 0.5])
    inputs, outputs, colors = exp2.sample(n=1000, env="E0")
    inputs.shape, outputs.shape
    for i in range(12):
        # print("-" * 12, i, "-" * 12)
        inputs, outputs, colors = exp2.sample(n=1000, env="E2")
        # print(np.corrcoef(inputs.numpy()[:, i], outputs.numpy()[:, 0])[0, 1])
