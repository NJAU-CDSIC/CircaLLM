import torch
from torch import nn


def nanvar(tensor, dim=None, keepdim=False):
    tensor_mean = tensor.nanmean(dim=dim, keepdim=True)
    output = (tensor - tensor_mean).square().nanmean(dim=dim, keepdim=keepdim)
    return output


def nanstd(tensor, dim=None, keepdim=False):
    output = nanvar(tensor, dim=dim, keepdim=keepdim)
    output = output.sqrt()
    return output


class RevIN(nn.Module):#Reversible Instance Normalization,一条数据一条数据进行处理和整体没关系
    #nn.Module 的内部实现中，__call__() 方法会自动调用 forward() 方法。
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        RevIN 作为类的名称传递给 super()，是为了确保在继承关系中正确地调用父类方法。
        它不是在 super() 中“重复”类本身，而是显式地指出父类是 nn.Module, 并调用其构造函数进行初始化。
        """
        
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x: torch.Tensor, mode: str = "norm", mask: torch.Tensor = None):
        """
        :param x: input tensor of shape (batch_size, n_channels, seq_len)
        :param mode: 'norm' or 'denorm'
        :param mask: input mask of shape (batch_size, seq_len)
        :return: RevIN transformed tensor
        """
        if mode == "norm":
            self._get_statistics(x, mask=mask)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(1, self.num_features, 1))
        self.affine_bias = nn.Parameter(torch.zeros(1, self.num_features, 1))

    def _get_statistics(self, x, mask=None):
        """
        x    : batch_size x n_channels x seq_len
        mask : batch_size x seq_len
        """
        if mask is None:
            mask = torch.ones((x.shape[0], x.shape[-1]))
        n_channels = x.shape[1]
        
        mask = mask.unsqueeze(1).repeat(1, n_channels, 1).bool()

        # Set masked positions to NaN, and unmasked positions are taken from x
        masked_x = torch.where(mask, x, torch.nan)
        self.mean = torch.nanmean(masked_x, dim=-1, keepdim=True).detach()
        self.stdev = nanstd(masked_x, dim=-1, keepdim=True).detach() + self.eps
        # self.stdev = torch.sqrt(
        #     torch.var(masked_x, dim=-1, keepdim=True) + self.eps).get_data().detach()
        # NOTE: By default not bessel correction

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev

        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x
