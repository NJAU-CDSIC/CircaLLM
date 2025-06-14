import logging
import warnings
from argparse import Namespace
from copy import deepcopy
from math import ceil

import torch
from huggingface_hub import PyTorchModelHubMixin 
from torch import nn
from transformers import T5Config, T5EncoderModel, T5Model
import numpy as np
import torch.fft as fft
from utils.common import TASKS
from data_provider.base import TimeseriesOutputs
from layers.embed import PatchEmbedding, Patching,CircaDataEmbedding
from layers.revin import RevIN
from utils.masking import Masking
from layers.FANLayer import FANLayer
from scipy.interpolate import CubicSpline
from scipy.signal import lombscargle,detrend
import math

from utils.utils import (
    NamespaceWithDefaults,
    get_anomaly_criterion,
    get_huggingface_model_dimensions,
)

SUPPORTED_HUGGINGFACE_MODELS = [
    "google/flan-t5-small",
    "google/flan-t5-base",
    "google/flan-t5-large",
    "google/flan-t5-xl",
    "google/flan-t5-xxl",
]


class PretrainHead(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        patch_len: int = 8,
        head_dropout: float = 0.1,
        orth_gain: float = 1.41,
    ):
        super().__init__()
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(d_model, patch_len)

        if orth_gain is not None:
            torch.nn.init.orthogonal_(self.linear.weight, gain=orth_gain)
            self.linear.bias.data.zero_()

    def forward(self, x):
        x = self.linear(self.dropout(x))
        x = x.flatten(start_dim=2, end_dim=3)
        return x

class ClassificationHead(nn.Module):
    def __init__(
        self,
        n_channels: int = 1,
        d_model: int = 768,
        n_classes: int = 2,
        head_dropout: int = 0.1,
        reduction: str = "mean",
    ):
        super().__init__()
        self.dropout = nn.Dropout(head_dropout) 
        if reduction == "mean":
            # self.linear =nn.Sequential(
            #     nn.Linear(d_model, d_model),  # 隐藏层
            #     nn.ReLU(),  # 激活函数
            #     nn.Dropout(head_dropout),  # Dropout层
            #     nn.Linear(d_model, n_classes)  # 输出层
            # )
            self.linear = nn.Linear(d_model, n_classes-1)  # 输出层
        elif reduction == "concat":
            self.linear = nn.Sequential(
                nn.Linear(n_channels * d_model, d_model),
                nn.ReLU(),
                nn.Dropout(head_dropout),
                nn.Linear(d_model // 2, n_classes-1)
            )
        else:
            raise ValueError(f"Reduction method {reduction} not implemented. Only 'mean' and 'concat' are supported.")
    def forward(self, x, input_mask: torch.Tensor = None):
        x = torch.mean(x, dim=1)#将时间点池化
        x = self.dropout(x)
        y = self.linear(x)
        return y
    
class DiffRhythmHead(nn.Module):
    def __init__(
        self,
        n_channels: int = 2,
        d_model: int = 512,
        n_classes: int = 4,
        head_dropout: int = 0.1,
        reduction: str = "concat",
    ):
        super().__init__()
        self.dropout = nn.Dropout(head_dropout)
        self.linear=nn.Sequential(
            nn.Linear(n_channels * d_model+4, d_model),  # 输入层到隐藏层
            nn.ReLU(),  # 激活函数
            nn.Linear(d_model,n_classes)
        )

    def forward(self, x,feature):
        x = torch.mean(x, dim=1)#将时间点池化
        x = self.dropout(x)

        x = torch.cat((x, feature), dim=1)
        y = self.linear(x)
        return y
    
def fft_feature(signal,t):
    # 进行 FFT 变换
    fft = torch.fft.fft(signal,dim=-1)

    # 计算幅度和相位
    amplitude = torch.abs(fft)
    phase = torch.angle(fft)
    # 频率轴
    n = signal.shape[-1]
    dt = t[1] - t[0]
    freq = torch.fft.fftfreq(n, d=dt)

    # 只取正频率部分
    positive_freq = freq[:n//2]
    positive_amplitude = amplitude[..., :n // 2]
    positive_phase = phase[..., :n // 2]

    # 找到每个批次的主要频率分量
    main_frequency_index = torch.argmax(positive_amplitude, dim=-1, keepdim=True)
    main_frequency = torch.gather(positive_freq.expand_as(positive_amplitude), -1, main_frequency_index)
    main_amplitude = torch.gather(positive_amplitude, -1, main_frequency_index) / (n * 0.5)
    main_phase = torch.abs(torch.gather(positive_phase, -1, main_frequency_index))

    # 计算周期
    main_period = 1 / main_frequency

    return main_phase, main_period

class HighPrecisionPhaseGrad(nn.Module):
    def __init__(self, time_aware=True):
        super().__init__()
        self.time_aware = time_aware  # 是否支持非均匀时间戳
    def pytorch_hilbert(self,x):
        """PyTorch可导的希尔伯特变换实现"""
        # FFT计算
        n = x.size(-1)
        X = fft.fft(x, dim=-1)
        
        # 构建希尔伯特核
        h = torch.zeros_like(X)
        h[..., 0] = 1
        h[..., 1:(n+1)//2] = 2
        if n % 2 == 0:
            h[..., n//2] = 1
        
        # 应用核并逆变换
        analytic = fft.ifft(X * h, dim=-1)
        return analytic
    def differentiable_unwrap(self,phase, discont=np.pi):
        dd = torch.diff(phase, dim=-1)
        ddmod = torch.remainder(dd + np.pi, 2*np.pi) - np.pi
        
        # 检测跳变点
        ddmod[(ddmod == -np.pi) & (dd > 0)] = np.pi
        ph_correct = ddmod - dd
        
        # 累积修正量
        ph_cumsum = torch.cat([
            torch.zeros_like(phase[..., :1]),
            torch.cumsum(ph_correct, dim=-1)
        ], dim=-1)
        
        return phase + ph_cumsum
    def torch_gradient(self,y, t, edge_order=1):
        """支持非均匀时间戳的梯度计算（可导）"""
        # 计算时间间隔
        dt = torch.diff(t, dim=-1)
        
        # 内部点中心差分
        grad_center = (y[..., 2:] - y[..., :-2]) / (dt[..., 1:] + dt[..., :-1])
        
        # 边界处理
        if edge_order == 1:
            grad_start = (y[..., 1] - y[..., 0]) / dt[..., 0]
            grad_end = (y[..., -1] - y[..., -2]) / dt[..., -1]
        else:  # 二阶边界
            grad_start = (-3*y[..., 0] + 4*y[..., 1] - y[..., 2]) / (dt[..., 0] + dt[..., 1])
            grad_end = (3*y[..., -1] - 4*y[..., -2] + y[..., -3]) / (dt[..., -2] + dt[..., -1])
        
        # 拼接结果
        return torch.cat([
            grad_start.unsqueeze(-1),
            grad_center,
            grad_end.unsqueeze(-1)
        ], dim=-1)
    def forward(self, x,x_mark):
        """
        输入：
            signal: [batch_size, seq_len] 时序信号
            t: [batch_size, seq_len] 时间戳（可选）
        输出：
            inst_freq: [batch_size, seq_len] 瞬时频率
            phase_grad: [batch_size, seq_len] 相位梯度
        """
        # pos=torch.nonzero(input_mask[0] == 1).squeeze()
        t=x_mark[0][:,0].float().to(x.device)
        # t=t[pos].to(x.device)
        # x1=x[:,:,pos].float()
        # signal=x1.mean(dim=1)
        signal=x.squeeze()
        # 希尔伯特变换
        analytic = self.pytorch_hilbert(signal)
        
        # 计算相位
        phase = torch.atan2(analytic.imag, analytic.real)
        
        # 相位解卷绕
        phase_unwrapped = self.differentiable_unwrap(phase)
        
        # 计算相位梯度
        if self.time_aware and t is not None:
            phase_grad = self.torch_gradient(phase_unwrapped, t)
        else:  # 均匀采样假设
            phase_grad = torch.diff(phase_unwrapped, dim=-1, prepend=phase_unwrapped[..., :1])
            phase_grad = phase_grad / (t[:,1]-t[:,0]).unsqueeze(-1) if t is not None else phase_grad
        
        # 计算瞬时频率
        inst_freq = phase_grad / (2 * torch.pi)
        return inst_freq
class FourierProjector(nn.Module):
    def __init__(self, n_freq=3):
        super().__init__()
        self.n_freq = n_freq  # 频率成分数量
    def forward(self, x,x_mark,input_mask):
        # 输入: [batch, 6] (假设6个时序特征)
        # 输出: [batch, 2n_freq] (各频率的sin/cos投影)
        pos=torch.nonzero(input_mask[0] == 1).squeeze()
        t=x_mark[0][:,0].float()
        t=t[pos].to(x.device)
        T_total = t.max()
        x1=x[:,:,pos].float()
        x1=x1.mean(dim=1)
        basis = []
        for k in range(self.n_freq):
            basis.append(torch.sin(2*np.pi*k*t/T_total))
            basis.append(torch.cos(2*np.pi*k*t/T_total))
        basis = torch.stack(basis, dim=1)  # [6, 2n_freq]
        proj = x1 @ basis  # [batch, 2n_freq]
        return proj
def interpolate_signal_torch(x_enc1, input_mask1,x_mark1):
    """
    对 PyTorch 张量进行三次样条插值。

    参数:
    x_torch (torch.Tensor): 形状为 [batchsize, channels, len_data] 的输入张量。
    new_length (int): 插值后希望达到的新长度。

    返回:
    torch.Tensor: 形状为 [batchsize, channels, new_length] 的插值后张量。
    """
    device=x_enc1.device
    x_enc1=x_enc1.cpu()
    input_mask1=input_mask1.cpu()
    x_mark1=x_mark1.cpu()
    pos=torch.nonzero(input_mask1[0] == 1).squeeze()
    x1=x_enc1[:,:,pos].float()
    batchsize, channels, len_data = x1.shape
    x_reshaped = x1.view(batchsize * channels, len_data)

    pos=torch.nonzero(input_mask1[0] == 1).squeeze()
    t=x_mark1[0,:,0].to(int)
    t_orig = t[pos].view(1, -1).expand(batchsize * channels, -1)

    l_new=28
    t_new = torch.linspace(0, 29, l_new).view(1, -1).expand(batchsize * channels, -1)  # 生成新的时间点

    # 对每个通道进行插值
    interpolated = torch.zeros(batchsize * channels, l_new)
    for i in range(batchsize * channels):
        cs = CubicSpline(t_orig[i].numpy(), x_reshaped[i].numpy())
        interpolated[i] = torch.tensor(cs(t_new[i].numpy()))

    interpolated = interpolated.view(batchsize, channels, l_new)  # 将张量重塑回原始形状
    interpolated = np.pad(interpolated.numpy(), ((0,0),(0,0),(72 - l_new, 0)))

    time_stamp=np.array([[math.floor(t), (t - math.floor(t))*60] for t in t_new[0]])
    time_stamp = np.tile(time_stamp, (batchsize, 1, 1))
    x_mark = np.pad(time_stamp, ((0,0),(72 - l_new, 0),(0,0)),constant_values=0)

    input_mask=np.pad(time_stamp, ((0,0),(72 - l_new, 0),(0,0)),constant_values=0)
    input_mask = np.ones(72)
    input_mask[: 72 - l_new]=0
    input_mask=np.tile(input_mask, (batchsize,1))
    
    return torch.from_numpy(interpolated).to(device).to(x_enc1.dtype), torch.from_numpy(x_mark).to(device).to(x_mark1.dtype), torch.from_numpy(input_mask).to(device).to(input_mask1.dtype)

def lombscargle_batch(x_enc1: torch.Tensor, x_mark1: torch.Tensor,input_mask1: torch.Tensor,freq_start=0.035714, freq_end=0.05, n_freq=2000) -> torch.Tensor:
    """
    输入: 
        x_tensor - 形状 [batch_size, n_channels, len_data] 的时序数据
    输出: 
        periods - 形状 [batch_size, 1] 的周期估计值（单位：小时）
    """
    # 生成频率序列
    frequencies = np.linspace(freq_start, freq_end, n_freq) 


    pos=torch.nonzero(input_mask1[0] == 1).squeeze()
    x_tensor=x_enc1[:,:,pos].float().cpu()

    t=x_mark1[0,:,0].to(int)
    t_orig = t[pos].cpu()
    
    # 逐批次处理
    periods = []
    for batch in x_tensor.unbind(dim=0):  # [batch_size, n_channels, L]
        channel_periods = []
        for channel in batch.unbind(dim=0):  
            # channel=detrend(channel.numpy(), type='constant')
            # 计算功率谱
            power = lombscargle(t_orig.numpy(), channel.numpy(), 2 * np.pi * frequencies,precenter=True)
            max_power_index = np.argmax(power)
            best_frequency = frequencies[max_power_index]
            channel_periods.append(1 / best_frequency)
            # # 获取前三个峰值频率‌
            # sorted_idx = np.argsort(power)[::-1][:3]
            # top3_freqs = frequencies[sorted_idx]
            # channel_periods.append(1 / top3_freqs)
        
        # 多通道聚合 [n_channels,3]
        batch_top3 = np.mean(channel_periods, axis=0)
        periods.append(batch_top3)
    periods=np.array(periods)
    return torch.tensor(periods).to(x_enc1.device).to(x_enc1.dtype)  # [batch_size,3]

class CIRCALLM(nn.Module):
    def __init__(self, config: Namespace | dict, **kwargs: dict):
        super().__init__()
        config = self._update_inputs(config, **kwargs)#config类型变为NamespaceWithDefaults
        config = self._validate_inputs(config)#检查一些关键参数
        self.config = config
        self.task_name = config.task_name
        self.seq_len = config.seq_len
        self.patch_len = config.patch_len
        self.normalizer = RevIN(num_features=1, affine=config.getattr("revin_affine", False))

        ##################################################################################################
        # self.tokenizer = Patching(
        #     patch_len=config.patch_len, stride=config.patch_stride_len
        # )
        # self.patch_embedding = PatchEmbedding(
        #     d_model=config.d_model,
        #     seq_len=config.seq_len,
        #     patch_len=config.patch_len,
        #     stride=config.patch_stride_len,
        #     patch_dropout=config.getattr("patch_dropout", 0.1),
        #     add_positional_embedding=config.getattr("add_positional_embedding", True),
        #     value_embedding_bias=config.getattr("value_embedding_bias", False),
        #     orth_gain=config.getattr("orth_gain", 1.41),
        # )
        # self.mask_generator = Masking(mask_ratio=config.getattr("mask_ratio", 0.0),patch_len=config.patch_len)

        self.data_embedding=CircaDataEmbedding(
            d_model=config.d_model,
            patch_dropout=config.getattr("patch_dropout", 0.1),
            add_positional_embedding=config.getattr("add_positional_embedding", True),
            value_embedding_bias=config.getattr("value_embedding_bias", False),
            orth_gain=config.getattr("orth_gain", 1.41),
        )
        ##################################################################################################

        self.encoder = self._get_transformer_backbone(config)
        self.head = self._get_head(self.task_name)
        # print(config.getattr("freeze_embedder", True))
        # print(config.getattr("freeze_encoder", True))
        # print(config.getattr("freeze_head", True))
        # Frozen parameters
        self.freeze_embedder = config.getattr("freeze_embedder", True)
        self.freeze_encoder = config.getattr("freeze_encoder", True)
        self.freeze_head = config.getattr("freeze_head", False)

        if self.freeze_embedder:
            # self.patch_embedding = freeze_parameters(self.patch_embedding)
            self.data_embedding = freeze_parameters(self.data_embedding)
        if self.freeze_encoder:
            self.encoder = freeze_parameters(self.encoder)
        if self.freeze_head:
            self.head = freeze_parameters(self.head)

    def _update_inputs(
        self, config: Namespace | dict, **kwargs: dict
    ) -> NamespaceWithDefaults:
        #isinstance(object, classinfo)检查object是不是classinfo类的实例
        if isinstance(config, dict) and "model_kwargs" in kwargs:
            '''
                假设
                config = {'learning_rate': 0.01, 'batch_size': 32}
                kwargs = {'model_kwargs': {'epochs': 10, 'optimizer': 'adam'}}
                字典解包用的符号为**
                {**config, **kwargs["model_kwargs"]}之后就会变成{'learning_rate': 0.01, 'batch_size': 32, 'epochs': 10, 'optimizer': 'adam'}
                然后Namespace(**{**config, **kwargs["model_kwargs"]}),就会变成Namespace(learning_rate=0.01, batch_size=32, epochs=10, optimizer='adam')
            '''
            return NamespaceWithDefaults(**{**config, **kwargs["model_kwargs"]})
        else:
            return NamespaceWithDefaults.from_namespace(config)

    def _validate_inputs(self, config: NamespaceWithDefaults) -> NamespaceWithDefaults:
        if (
            config.d_model is None
            and config.transformer_backbone in SUPPORTED_HUGGINGFACE_MODELS
        ):
            #d_model表示模型中每个嵌入（embedding）的维度，表示每个 token（单词或字符）的向量表示的维度
            config.d_model = config.t5_config['d_model']
            logging.info(f"Setting d_model to {config.d_model}")
        elif config.d_model is None:
            raise ValueError(
                "d_model must be specified if transformer backbone "
                "unless transformer backbone is a Huggingface model."
            )

        if config.transformer_type not in [
            "encoder_only",
            "decoder_only",
            "encoder_decoder",
        ]:
            raise ValueError(
                "transformer_type must be one of "
                "['encoder_only', 'decoder_only', 'encoder_decoder']"
            )

        if config.patch_stride_len != config.patch_len:
            warnings.warn("Patch stride length is not equal to patch length.")
        return config

    def _get_head(self, task_name: str) -> nn.Module:
        if task_name == TASKS.CLASSIFICATION:
            return ClassificationHead(
                self.config.n_channels,
                self.config.d_model,
                self.config.num_class,
                self.config.getattr("head_dropout", 0.1),
                reduction = self.config.getattr("reduction", "concat"),
            )
        elif task_name == TASKS.DIFFRHYTHM:
            return DiffRhythmHead(
                self.config.n_channels,
                self.config.d_model,
                self.config.num_class,
                self.config.getattr("head_dropout", 0.1),
                reduction = self.config.getattr("reduction", "concat"),
            )
        elif task_name == TASKS.EMBED:
            return nn.Identity()
        else:
            raise NotImplementedError(f"Task {task_name} not implemented.")

    def _get_transformer_backbone(self, config) -> nn.Module:
        model_config = T5Config.from_dict(config.t5_config)

        if config.getattr("randomly_initialize_backbone", False):
            transformer_backbone = T5Model(model_config)
            logging.info(
                f"Initializing randomly initialized transformer from {config.transformer_backbone}."
            )
        else:
            transformer_backbone = T5EncoderModel(model_config)
            logging.info(
                f"Initializing pre-trained transformer from {config.transformer_backbone}."
            )

        transformer_backbone = transformer_backbone.get_encoder()
        # print(transformer_backbone)
        if config.getattr("enable_FAN", False):
            if model_config.dense_act_fn=="gelu":
                for block in transformer_backbone.block:
                    MLPlayer=block.layer[1]
                    MLPlayer.DenseReluDense.wi=FANLayer(input_dim=MLPlayer.DenseReluDense.wi.in_features,output_dim=MLPlayer.DenseReluDense.wi.out_features,with_gate=config.getattr("enable_FAN_gate", True))
                    MLPlayer.DenseReluDense.wo=FANLayer(input_dim=MLPlayer.DenseReluDense.wo.in_features,output_dim=MLPlayer.DenseReluDense.wo.out_features,with_gate=config.getattr("enable_FAN_gate", True))
            elif model_config.dense_act_fn=="gelu_new":
                for block in transformer_backbone.block:
                    MLPlayer=block.layer[1]
                    MLPlayer.DenseReluDense.wi_0=FANLayer(input_dim=MLPlayer.DenseReluDense.wi_0.in_features,output_dim=MLPlayer.DenseReluDense.wi_0.out_features,with_gate=config.getattr("enable_FAN_gate", True))
                    MLPlayer.DenseReluDense.wi_1=FANLayer(input_dim=MLPlayer.DenseReluDense.wi_1.in_features,output_dim=MLPlayer.DenseReluDense.wi_1.out_features,with_gate=config.getattr("enable_FAN_gate", True))
                    MLPlayer.DenseReluDense.wo=FANLayer(input_dim=MLPlayer.DenseReluDense.wo.in_features,output_dim=MLPlayer.DenseReluDense.wo.out_features,with_gate=config.getattr("enable_FAN_gate", True))
        if config.getattr("enable_gradient_checkpointing", True):
            transformer_backbone.gradient_checkpointing_enable()
            logging.info("Enabling gradient checkpointing.")

        return transformer_backbone

    def __call__(self, *args, **kwargs) -> TimeseriesOutputs:
        return self.forward(*args, **kwargs)

    def embed(
        self,
        *,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        reduction: str = "mean",
        **kwargs,
    ) -> TimeseriesOutputs:
        batch_size, n_channels, seq_len = x_enc.shape

        if input_mask is None:
            input_mask = torch.ones((batch_size, seq_len)).to(x_enc.device)

        x_enc = self.normalizer(x=x_enc, mask=input_mask, mode="norm")
        x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)

        input_mask_patch_view = Masking.convert_seq_to_patch_view(
            input_mask, self.patch_len
        )

        x_enc = self.tokenizer(x=x_enc)
        enc_in = self.patch_embedding(x_enc, mask=input_mask)

        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape(
            (batch_size * n_channels, n_patches, self.config.d_model)
        )

        patch_view_mask = Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
        attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0)
        outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
        enc_out = outputs.last_hidden_state

        enc_out = enc_out.reshape((-1, n_channels, n_patches, self.config.d_model))
        # [batch_size x n_channels x n_patches x d_model]

        if reduction == "mean":
            enc_out = enc_out.mean(dim=1, keepdim=False)  # Mean across channels
            # [batch_size x n_patches x d_model]
            input_mask_patch_view = input_mask_patch_view.unsqueeze(-1).repeat(
                1, 1, self.config.d_model
            )
            enc_out = (input_mask_patch_view * enc_out).sum(
                dim=1
            ) / input_mask_patch_view.sum(dim=1)
        else:
            raise NotImplementedError(f"Reduction method {reduction} not implemented.")

        return TimeseriesOutputs(
            embeddings=enc_out, input_mask=input_mask, metadata=reduction
        )

    def reconstruction(
        self,
        *,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        mask: torch.Tensor = None,
        **kwargs,
    ) -> TimeseriesOutputs:
        batch_size, n_channels, _ = x_enc.shape

        if mask is None:
            mask = self.mask_generator.generate_mask(x=x_enc, input_mask=input_mask)
            mask = mask.to(x_enc.device)  # mask: [batch_size x seq_len]

        x_enc = self.normalizer(x=x_enc, mask=mask * input_mask, mode="norm")
        # Prevent too short time-series from causing NaNs
        x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)

        x_enc = self.tokenizer(x=x_enc)
        enc_in = self.patch_embedding(x_enc, mask=mask)

        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape(
            (batch_size * n_channels, n_patches, self.config.d_model)
        )

        patch_view_mask = Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
        attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0)
        if self.config.transformer_type == "encoder_decoder":
            outputs = self.encoder(
                inputs_embeds=enc_in,
                decoder_inputs_embeds=enc_in,
                attention_mask=attention_mask,
            )
        else:
            outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
        enc_out = outputs.last_hidden_state

        enc_out = enc_out.reshape((-1, n_channels, n_patches, self.config.d_model))

        dec_out = self.head(enc_out)  # [batch_size x n_channels x seq_len]
        dec_out = self.normalizer(x=dec_out, mode="denorm")

        if self.config.getattr("debug", False):
            illegal_output = self._check_model_weights_for_illegal_values()
        else:
            illegal_output = None

        return TimeseriesOutputs(
            input_mask=input_mask,
            reconstruction=dec_out,
            pretrain_mask=mask,
            illegal_output=illegal_output,
        )

    def classify(
        self,
        *,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        x_mark: torch.Tensor = None,
        reduction: str = "concat",
        **kwargs,
    ) -> TimeseriesOutputs:
        batch_size, n_channels, seq_len = x_enc.shape

        if input_mask is None:
            input_mask = torch.ones((batch_size, seq_len)).to(x_enc.device)
        # print(f'input_mask.shape:{input_mask.shape}')
        # print(f'input_mask:{input_mask}')
        x_enc = self.normalizer(x=x_enc, mask=input_mask, mode="norm")
        x_enc = torch.nan_to_num(x_enc, nan=0, posinf=0, neginf=0)
        
        ##################################################################################################
        # x_enc = self.tokenizer(x=x_enc)
        # enc_in = self.patch_embedding(x_enc, mask=input_mask)
        
        # n_patches = enc_in.shape[2]
        # enc_in = enc_in.reshape(
        #     (batch_size * n_channels, n_patches, self.config.d_model)
        # )

        # patch_view_mask = Masking.convert_seq_to_patch_view(input_mask, self.patch_len)
        # attention_mask = patch_view_mask.repeat_interleave(n_channels, dim=0)

        enc_in = self.data_embedding(x_enc, mask=input_mask,x_mark=x_mark)
        enc_in = enc_in.reshape(
            (batch_size * n_channels, seq_len, self.config.d_model)
        )

        attention_mask = input_mask.repeat_interleave(n_channels, dim=0)

        ##################################################################################################

        outputs = self.encoder(inputs_embeds=enc_in, attention_mask=attention_mask)
        enc_out = outputs.last_hidden_state
        # print(f'enc_out.shape1:{enc_out.shape}')
        enc_out = enc_out.reshape((-1, n_channels, seq_len, self.config.d_model))
        # print(f'enc_out.shape2:{enc_out.shape}')
        # [batch_size x n_channels x n_patches x d_model]
        # enc_out = enc_out.mean(dim=1, keepdim=False)
        # # Mean across channels
        if reduction=="mean":
            enc_out=enc_out.mean(dim=1,keepdim=False)# [batch_size x n_patches x d_model]
        elif reduction=="concat":
            # # permute(0,2,3,1)会把[batch_size x n_channels x n_patches x d_model]变成[batch_size x n_patches x d_model * n_channels],n_patches为patch的数量
            enc_out=enc_out.permute(0,2,3,1).reshape(batch_size, seq_len, self.config.d_model * n_channels)
        else:
            raise NotImplementedError(f"Reduction method {reduction} not implemented.")
        logits = self.head(enc_out, input_mask=input_mask)

        return TimeseriesOutputs(embeddings=enc_out, logits=logits, metadata=reduction)

    def diffthyrhm_classify(
        self,
        *,
        x_enc1: torch.Tensor,
        input_mask1: torch.Tensor = None,
        x_mark1: torch.Tensor = None,
        
        x_enc2: torch.Tensor,
        input_mask2: torch.Tensor = None,
        x_mark2: torch.Tensor = None,
        reduction: str = "concat",
        **kwargs,
    ) -> TimeseriesOutputs:
        batch_size1, n_channels1, seq_len1 = x_enc1.shape
        batch_size2, n_channels2, seq_len2 = x_enc2.shape

        if input_mask1 is None:
            input_mask1 = torch.ones((batch_size1, seq_len1)).to(x_enc1.device)
        if input_mask2 is None:
            input_mask2 = torch.ones((batch_size2, seq_len2)).to(x_enc2.device)

        # fourier = FourierProjector()
        # fourier1,fourier2=fourier(x_enc1,x_mark1,input_mask1),fourier(x_enc2,x_mark2,input_mask2)
        # pos=torch.nonzero(input_mask1[0] == 1).squeeze()
        # t=x_mark1[0][pos,0].int()
        # x1,x2=x_enc1[:,:,pos],x_enc2[:,:,pos]
        # x_enc1_phase,x_enc1_period=fft_feature(x1.cpu(),t.cpu())
        # x_enc1,x_mark1,input_mask1=interpolate_signal_torch(x_enc1,input_mask1,x_mark1)
        # x_enc2,x_mark2,input_mask2=interpolate_signal_torch(x_enc2,input_mask2,x_mark2)
        # print(f'x_enc1.shape:{x_enc1.shape}')
        # print(f'input_mask1.shape:{input_mask1.shape}')
        # print(f'x_mark1.shape:{x_mark1.shape}')
        # batch_size1, n_channels1, seq_len1 = x_enc1.shape
        # batch_size2, n_channels2, seq_len2 = x_enc2.shape

        # period1=lombscargle_batch(x_enc1,x_mark1,input_mask1)
        # period2=lombscargle_batch(x_enc2,x_mark2,input_mask2)

        x_enc1 = self.normalizer(x=x_enc1, mask=input_mask1, mode="norm")
        mean1 = self.normalizer.mean.mean(dim=1)
        stdev1 = self.normalizer.stdev.mean(dim=1)
        
        x_enc2 = self.normalizer(x=x_enc2, mask=input_mask2, mode="norm")
        mean2 = self.normalizer.mean.mean(dim=1)
        stdev2 = self.normalizer.stdev.mean(dim=1)
        
        # PhaseGrad=HighPrecisionPhaseGrad()
        # PhaseGrad1,PhaseGrad2=PhaseGrad(x_enc1,x_mark1),PhaseGrad(x_enc2,x_mark2)
        feature = torch.cat((mean1, stdev1,mean2, stdev2), dim=1)

        x_enc1, x_enc2 = torch.nan_to_num(x_enc1, nan=0, posinf=0, neginf=0), torch.nan_to_num(x_enc2, nan=0, posinf=0, neginf=0)
        

        enc_in1,enc_in2 = self.data_embedding(x_enc1, mask=input_mask1,x_mark=x_mark1),self.data_embedding(x_enc2, mask=input_mask2,x_mark=x_mark2)
        enc_in1,enc_in2 = enc_in1.reshape(
            (batch_size1 * n_channels1, seq_len1, self.config.d_model)
        ),enc_in2.reshape(
            (batch_size2 * n_channels2, seq_len2, self.config.d_model)
        )
        attention_mask1,attention_mask2 = input_mask1.repeat_interleave(n_channels1, dim=0),input_mask2.repeat_interleave(n_channels2, dim=0)

        outputs1, outputs2 = self.encoder(inputs_embeds=enc_in1, attention_mask=attention_mask1),self.encoder(inputs_embeds=enc_in2, attention_mask=attention_mask2)
        enc_out1, enc_out2 = outputs1.last_hidden_state,outputs2.last_hidden_state

        enc_out1,enc_out2 = enc_out1.reshape((-1, n_channels1, seq_len1, self.config.d_model)),enc_out2.reshape((-1, n_channels2, seq_len2, self.config.d_model))

        # [batch_size x n_channels x n_patches x d_model]
        enc_out1, enc_out2 = enc_out1.mean(dim=1,keepdim=False), enc_out2.mean(dim=1,keepdim=False)
        
        #将enc_out1和enc_out2通过concat方式拼接起来
        enc_out = torch.cat((enc_out1, enc_out2), dim=-1)

        logits = self.head(enc_out,feature)

        return TimeseriesOutputs(embeddings=enc_out, logits=logits, metadata=reduction,embeddings1=enc_out1,embeddings2=enc_out2)
    
    def forward(
        self,
        *,
        x_enc: torch.Tensor,
        input_mask: torch.Tensor = None,
        x_mark:torch.Tensor = None,

        x_enc2:torch.Tensor= None,
        input_mask2: torch.Tensor = None,
        x_mark2:torch.Tensor = None,

        mask: torch.Tensor = None,
        **kwargs,
    ) -> TimeseriesOutputs:
        if input_mask is None:
            input_mask = torch.ones_like(x_enc[:, 0, :])

        if self.task_name == TASKS.RECONSTRUCTION:
            return self.reconstruction(
                x_enc=x_enc, mask=mask, input_mask=input_mask, **kwargs
            )
        elif self.task_name == TASKS.EMBED:
            return self.embed(x_enc=x_enc, input_mask=input_mask, **kwargs)
        elif self.task_name == TASKS.CLASSIFICATION:
            return self.classify(x_enc=x_enc, input_mask=input_mask, x_mark=x_mark, **kwargs)
        elif self.task_name == TASKS.DIFFRHYTHM:
            return self.diffthyrhm_classify(x_enc1=x_enc, input_mask1=input_mask, x_mark1=x_mark,
                                            x_enc2=x_enc2, input_mask2=input_mask2, x_mark2=x_mark2, **kwargs)
        else:
            raise NotImplementedError(f"Task {self.task_name} not implemented.")


class CIRCALLMPipeline(CIRCALLM, PyTorchModelHubMixin):
    def __init__(self, config: Namespace | dict, **kwargs: dict):
        # print(config)
        # print(kwargs)
        self._validate_model_kwargs(**kwargs)
        self.new_task_name = kwargs.get("model_kwargs", {}).pop(
            "task_name", TASKS.CLASSIFICATION
        )
        super().__init__(config, **kwargs)

    def _validate_model_kwargs(self, **kwargs: dict) -> None:
        kwargs = deepcopy(kwargs)
        kwargs.setdefault("model_kwargs", {"task_name": TASKS.CLASSIFICATION})
        kwargs["model_kwargs"].setdefault("task_name", TASKS.CLASSIFICATION)
        config = Namespace(**kwargs["model_kwargs"])
        
        if config.task_name == TASKS.CLASSIFICATION or config.task_name == TASKS.DIFFRHYTHM:
            if not hasattr(config, "num_class"):
                raise ValueError("num_class must be specified for classification.")
    def init(self) -> None:
        if self.new_task_name != TASKS.CLASSIFICATION:
            self.task_name = self.new_task_name
            self.head = self._get_head(self.new_task_name)

def freeze_parameters(model):
    """
    Freeze parameters of the model
    """
    # Freeze the parameters
    for name, param in model.named_parameters():
        param.requires_grad = False

    return model
