from torch import nn, optim, autograd
import pdb
import torch
from torchvision import datasets

class ENV_EBD(nn.Module):
    def __init__(self, flags):
      super(ENV_EBD, self).__init__()
      self.embedings = torch.nn.Embedding(flags.envs_num, 4)
      self.re_init()

    def re_init(self):
      pass
      # self.embedings.weight.data.fill_(1.)

    def forward(self, e):
      return self.embedings(e.long())

class EBD(nn.Module):
    def __init__(self, flags):
      super(EBD, self).__init__()
      self.flags = flags
      # if self.flags.num_classes == 2:
      self.embedings = torch.nn.Embedding(flags.envs_num, 1)
      # else:
      #     self.embedings = torch.nn.Embedding(flags.envs_num, self.flags.num_classes)
      self.re_init()

    def re_init(self):
      self.embedings.weight.data.fill_(1.)

    def re_init_with_noise(self, noise_sd):
      # if self.flags.num_classes == 2:
      rd = torch.normal(
         torch.Tensor([1.0] * self.flags.envs_num),
         torch.Tensor([noise_sd] * self.flags.envs_num))
      self.embedings.weight.data = rd.view(-1, 1).cuda()
      # else:
      # rd = torch.normal(
      #    torch.Tensor([1.0] * self.flags.envs_num * self.flags.num_classes),
      #    torch.Tensor([noise_sd] * self.flags.envs_num* self.flags.num_classes))
      # self.embedings.weight.data = rd.view(-1, self.flags.num_classes).cuda()

    def forward(self, e):
      return self.embedings(e.long())


class Y_EBD(nn.Module):
    def __init__(self, flags):
      super(Y_EBD, self).__init__()
      self.embedings = torch.nn.Embedding(flags.classes_num, 4)
      self.re_init()

    def re_init(self):
      pass
      # self.embedings.weight.data.fill_(1.)

    def forward(self, e):
      return self.embedings(e.long())


class BayesW(nn.Module):
    def __init__(self, prior, flags, update_w=True):
        super(BayesW, self).__init__()
        self.pw, self.psigma = prior
        self.flags = flags
        self.vw = torch.nn.Parameter(self.pw.clone(), requires_grad=update_w)
        self.vsigma= torch.nn.Parameter(self.psigma.clone())
        self.nll = nn.MSELoss()
        self.re_init()

    def reset_prior(self, prior):
        self.pw, self.psigma = prior
        print("resetting prior", self.pw.item(), self.psigma.item())

    def reset_posterior(self, prior):
        new_w, new_sigma = prior
        self.vw.data, self.vsigma.data = new_w.clone(), new_sigma.clone()
        print("resetting posterior", self.pw.item(), self.psigma.item())


    def generate_rand(self, N):
        self.epsilon = list()
        for i in range(N):
            self.epsilon.append(
                torch.normal(
                    torch.tensor(0.0),
                    torch.tensor(1.0)))

    def variational_loss(self, xb, yb, N):
        pw, psigma = self.pw, self.psigma
        vw, vsigma = self.vw, self.vsigma
        kl = torch.log(psigma/vsigma) + (vsigma ** 2 + (vw - pw) ** 2) / (2 * psigma ** 2)
        lk_loss = 0
        assert N == len(self.epsilon)
        for i in range(N):
            epsilon_i = self.epsilon[i]
            wt_ei = vw + vsigma * epsilon_i
            loss_i = self.nll(wt_ei * xb, yb)
            lk_loss += 1./N * loss_i
        return lk_loss + 1./self.flags.data_num  * kl

    def forward(self, x):
        return self.vw * x

    def re_init(self):
      pass

    def init_sep_by_share(self, share_bayes_net):
        self.vw.data = share_bayes_net.vw.data.clone()
        self.vsigma.data = share_bayes_net.vsigma.data.clone()
        self.epsilon = share_bayes_net.epsilon


class MLP(nn.Module):
    def __init__(self, flags):
        super(MLP, self).__init__()
        self.flags = flags
        if flags.grayscale_model:
          lin1 = nn.Linear(14 * 14, flags.hidden_dim)
        else:
          lin1 = nn.Linear(2 * 14 * 14, flags.hidden_dim)
        lin2 = nn.Linear(flags.hidden_dim, flags.hidden_dim)
        if flags.num_classes == 2:
            lin3 = nn.Linear(flags.hidden_dim, 1)
        else:
            raise Exception
        for lin in [lin1, lin2, lin3]:
          nn.init.xavier_uniform_(lin.weight)
          nn.init.zeros_(lin.bias)
        self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)

    def forward(self, input):
        if self.flags.grayscale_model:
          out = input.view(input.shape[0], 2, 14 * 14).sum(dim=1)
        else:
          out = input.view(input.shape[0], 2 * 14 * 14)
        out = self._main(out)
        return out


class MLPFull(nn.Module):
    def __init__(self, flags):
        super(MLPFull, self).__init__()
        self.flags = flags
        if flags.grayscale_model:
          lin1 = nn.Linear(14 * 14, flags.hidden_dim)
        else:
          lin1 = nn.Linear(3 * 14 * 14, flags.hidden_dim)
        lin2 = nn.Linear(flags.hidden_dim, flags.hidden_dim)
        lin3 = nn.Linear(flags.hidden_dim, flags.num_classes)
        for lin in [lin1, lin2, lin3]:
          nn.init.xavier_uniform_(lin.weight)
          nn.init.zeros_(lin.bias)
        self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)

    def forward(self, input):
        if self.flags.grayscale_model:
          out = input.view(input.shape[0], 3, 14 * 14).sum(dim=1)
        else:
          out = input.view(input.shape[0], 3 * 14 * 14)
        out = self._main(out)
        return out


class PredEnvHatY(nn.Module):
    def __init__(self, flags):
        super(PredEnvHatY, self).__init__()
        self.lin1 = nn.Linear(1, flags.hidden_dim)
        self.lin2 = nn.Linear(flags.hidden_dim, 1)
        for lin in [self.lin1, self.lin2]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self._main = nn.Sequential(
            self.lin1, nn.ReLU(True), self.lin2)

    def forward(self, input):
        out = self._main(input)
        return out


class InferEnv(nn.Module):
    def __init__(self, flags):
        super(InferEnv, self).__init__()
        self.lin1 = nn.Linear(1, flags.hidden_dim)
        self.lin2 = nn.Linear(flags.hidden_dim, 1)
        for lin in [self.lin1, self.lin2]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self._main = nn.Sequential(
            self.lin1, nn.ReLU(True), self.lin2, nn.Sigmoid())

    def forward(self, input):
        out = self._main(input)
        return out


class PredEnvYY(nn.Module):
    def __init__(self, flags):
        super(PredEnvYY, self).__init__()
        lin1 = nn.Linear(5, flags.hidden_dim)
        lin2 = nn.Linear(flags.hidden_dim, 1)
        for lin in [lin1, lin2]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self._main = nn.Sequential(
            lin1, nn.ReLU(True), lin2)

    def forward(self, input):
        out = self._main(input)
        return out

class PredEnvHatYSep(nn.Module):
    def __init__(self, flags):
        super(PredEnvHatYSep, self).__init__()
        self.lin1_1 = nn.Linear(1, flags.hidden_dim)
        self.lin1_2 = nn.Linear(flags.hidden_dim, 1)
        self.lin2_1 = nn.Linear(1, flags.hidden_dim)
        self.lin2_2 = nn.Linear(flags.hidden_dim, 1)
        for lin in [self.lin1_1, self.lin1_2, self.lin2_1, self.lin2_2]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self._main1 = nn.Sequential(
           self. lin1_1, nn.ReLU(True), self.lin1_2)
        self._main2 = nn.Sequential(
            self.lin2_1, nn.ReLU(True), self.lin2_2)

    def init_sep_by_share(self, share_net):
        self.lin1_1.weight.data = share_net.lin1.weight.data.clone()
        self.lin1_2.weight.data = share_net.lin2.weight.data.clone()
        self.lin1_1.bias.data = share_net.lin1.bias.data.clone()
        self.lin1_2.bias.data = share_net.lin2.bias.data.clone()
        self.lin2_1.weight.data = share_net.lin1.weight.data.clone()
        self.lin2_2.weight.data = share_net.lin2.weight.data.clone()
        self.lin2_1.bias.data = share_net.lin1.bias.data.clone()
        self.lin2_2.bias.data = share_net.lin2.bias.data.clone()

    def forward(self, g, input):
        output = torch.zeros_like(g).cuda()
        output[g == 0]= self._main1(
            input[g == 0].view(-1, 1)).view(-1)
        output[g == 1]= self._main2(
            input[g == 1].view(-1, 1)).view(-1)
        return output


class PredYEnvHatY(nn.Module):
    def __init__(self, flags):
        super(PredYEnvHatY, self).__init__()
        lin1 = nn.Linear(5, flags.hidden_dim)
        lin2 = nn.Linear(flags.hidden_dim, 1)
        for lin in [lin1, lin2]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self._main = nn.Sequential(
            lin1, nn.ReLU(True), lin2)

    def forward(self, input):
        out = self._main(input)
        return out


__all__ = [ 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        num_classes = 1
        self.num_classes = 1
        self.class_classifier = nn.Linear(512 * block.expansion, num_classes)
        self.sep=False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def encoder(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    def forward(self, x):

        x = self.encoder(x)
        self.fp = x
        return self.class_classifier(x)

    def sep_param_id(self):
        sep_params = [
            p[1] for p
            in self.named_parameters()
            if 'classifier' in p[0] and
                'sep' in p[0]]
        sep_param_id = [id(i) for i in sep_params]
        return sep_param_id

    def rep_param_id(self):
        rep_param_id = [
            id(p) for p in self.parameters()
            if id(p) not in  self.sep_param_id()
                and id(p) not in self.share_param_id()]
        return rep_param_id

    def get_optimizer_schedule(self, args):
        if args.irm_penalty_weight > 0:
            if args.opt == "Adam":
                opt_fun = optim.Adam
                optimizer_rep = opt_fun(
                    filter(lambda p:id(p) in self.rep_param_id(), self.parameters()),
                    lr=args.lr)
                optimizer_share = opt_fun(
                    filter(lambda p:id(p) in self.share_param_id(), self.parameters()),
                    lr=args.lr* args.penalty_wlr)
                optimizer_sep = optim.SGD(
                    filter(lambda p:id(p) in self.sep_param_id(), self.parameters()),
                    lr=args.lr* args.penalty_welr)
            elif args.opt == "SGD":
                opt_fun = optim.SGD
                optimizer_rep = opt_fun(
                    filter(lambda p:id(p) in self.rep_param_id(), self.parameters()),
                    momentum=0.9,
                    lr=args.lr)
                optimizer_share = opt_fun(
                    filter(lambda p:id(p) in self.share_param_id(), self.parameters()),
                    momentum=args.w_momentum,
                    lr=args.lr * args.penalty_wlr)
                optimizer_sep = opt_fun(
                    filter(lambda p:id(p) in self.sep_param_id(), self.parameters()),
                    momentum=args.w_momentum,
                    lr=args.lr * args.penalty_welr)
            else:
                raise Exception
            if args.lr_schedule_type == "step":
                print("step_gamma=%s" % args.step_gamma)
                scheduler_rep = lr_scheduler.StepLR(optimizer_rep, step_size=int(args.n_epochs/2.5), gamma=args.step_gamma)
                scheduler_sep = lr_scheduler.StepLR(optimizer_sep, step_size=int(args.n_epochs), gamma=args.step_gamma)
                scheduler_share = lr_scheduler.StepLR(optimizer_share, step_size=int(args.n_epochs/2.5), gamma=args.step_gamma)

            return [optimizer_rep, optimizer_share, optimizer_sep], [scheduler_rep, scheduler_share, scheduler_sep]
        else:
            if args.opt == "Adam":
                opt_fun = optim.Adam
                optimizer= opt_fun(
                    filter(lambda p:id(p) in self.rep_param_id(), self.parameters()),
                    lr=args.lr)
            elif args.opt == "SGD":
                opt_fun = optim.SGD
                optimizer= opt_fun(
                    filter(lambda p:id(p) in self.rep_param_id(), self.parameters()),
                    momentum=0.9,
                    lr=args.lr)
            else:
                raise Exception
            scheduler= lr_scheduler.StepLR(
                optimizer,
                step_size=int(args.n_epochs/3.),
                gamma=args.step_gamma)
            return [optimizer], [scheduler]



class ResNetUS(ResNet):
    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        nn.Module.__init__(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        ## CIFAR10: kernel_size 7 -> 3, stride 2 -> 1, padding 3->1
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        ## END

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # num_classes = 1
        self.num_classes = num_classes
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.class_classifier = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


def _resnet_sepfc_us(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNetUS(block, layers, **kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],progress=progress)
        model_dict = model.state_dict()
        state_dict['conv1.weight'] = model_dict['conv1.weight']
        pretrained_state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(pretrained_state_dict)
        model.load_state_dict(model_dict)
    return model

def resnet18_sepfc_us(pretrained=False, progress=True, **kwargs):
    return _resnet_sepfc_us('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet50_sepfc_us(pretrained=False, progress=True, **kwargs):
    return _resnet_sepfc_us(
        'resnet50', Bottleneck,
        [3, 4, 6, 3], pretrained, progress,
        **kwargs)


