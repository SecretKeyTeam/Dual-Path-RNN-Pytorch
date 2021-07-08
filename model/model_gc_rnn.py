import sys
sys.path.append('../')

import torch.nn.functional as F
from torch import nn
import torch
from utils.util import check_parameters

import warnings

warnings.filterwarnings('ignore')

class GlobalLayerNorm(nn.Module):
    '''
       Calculate Global Layer Normalization
       dim: (int or list or torch.Size) –
          input shape from an expected input of size
       eps: a value added to the denominator for numerical stability.
       elementwise_affine: a boolean value that when set to True, 
          this module has learnable per-element affine parameters 
          initialized to ones (for weights) and zeros (for biases).
    '''

    def __init__(self, dim, shape, eps=1e-8, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            if shape == 3:
                self.weight = nn.Parameter(torch.ones(self.dim, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1))
            if shape == 4:
                self.weight = nn.Parameter(torch.ones(self.dim, 1, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        # x = N x C x K x S or N x C x L
        # N x 1 x 1
        # cln: mean,var N x 1 x K x S
        # gln: mean,var N x 1 x 1
        if x.dim() == 4:
            mean = torch.mean(x, (1, 2, 3), keepdim=True)
            var = torch.mean((x-mean)**2, (1, 2, 3), keepdim=True)
            if self.elementwise_affine:
                x = self.weight*(x-mean)/torch.sqrt(var+self.eps)+self.bias
            else:
                x = (x-mean)/torch.sqrt(var+self.eps)
        if x.dim() == 3:
            mean = torch.mean(x, (1, 2), keepdim=True)
            var = torch.mean((x-mean)**2, (1, 2), keepdim=True)
            if self.elementwise_affine:
                x = self.weight*(x-mean)/torch.sqrt(var+self.eps)+self.bias
            else:
                x = (x-mean)/torch.sqrt(var+self.eps)
        return x


class CumulativeLayerNorm(nn.LayerNorm):
    '''
       Calculate Cumulative Layer Normalization
       dim: you want to norm dim
       elementwise_affine: learnable per-element affine parameters 
    '''

    def __init__(self, dim, elementwise_affine=True):
        super(CumulativeLayerNorm, self).__init__(
            dim, elementwise_affine=elementwise_affine, eps=1e-8)

    def forward(self, x):
        # x: N x C x K x S or N x C x L
        # N x K x S x C
        if x.dim() == 4:
           x = x.permute(0, 2, 3, 1).contiguous()
           # N x K x S x C == only channel norm
           x = super().forward(x)
           # N x C x K x S
           x = x.permute(0, 3, 1, 2).contiguous()
        if x.dim() == 3:
            x = torch.transpose(x, 1, 2)
            # N x L x C == only channel norm
            x = super().forward(x)
            # N x C x L
            x = torch.transpose(x, 1, 2)
        return x


def select_norm(norm, dim, shape):
    if norm == 'gln':
        return GlobalLayerNorm(dim, shape, elementwise_affine=True)
    if norm == 'cln':
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    if norm == 'ln':
        return nn.GroupNorm(1, dim, eps=1e-8)
    else:
        return nn.BatchNorm1d(dim)

class Encoder(nn.Module):
    '''
       Conv-Tasnet Encoder part
       kernel_size: the length of filters
       out_channels: the number of filters
    '''

    def __init__(self, kernel_size=2, out_channels=64):
        super(Encoder, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=out_channels,
                                kernel_size=kernel_size, stride=kernel_size//2, groups=1, bias=False)

    def forward(self, x):
        """
          Input:
              x: [B, T], B is batch size, T is times
          Returns:
              x: [B, C, T_out]
              T_out is the number of time steps
        """
        # B x T -> B x 1 x T
        x = torch.unsqueeze(x, dim=1)
        # B x 1 x T -> B x C x T_out
        x = self.conv1d(x)
        x = F.relu(x)
        return x


class Decoder(nn.ConvTranspose1d):
    '''
        Decoder of the TasNet
        This module can be seen as the gradient of Conv1d with respect to its input. 
        It is also known as a fractionally-strided convolution 
        or a deconvolution (although it is not an actual deconvolution operation).
    '''

    def __init__(self, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        x: [B, N, L]
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 3/4D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))

        if torch.squeeze(x).dim() == 1:
            x = torch.squeeze(x, dim=1)
        else:
            x = torch.squeeze(x)
        return x


class Grouo_Comm_Dual_RNN_Block(nn.Module):
    '''
       Implementation of the intra-RNN and the inter-RNN
       input:
            in_channels: The number of expected features in the input x
            out_channels: The number of features in the hidden state h
            rnn_type: RNN, LSTM, GRU
            norm: gln = "Global Norm", cln = "Cumulative Norm", ln = "Layer Norm"
            dropout: If non-zero, introduces a Dropout layer on the outputs 
                     of each LSTM layer except the last layer, 
                     with dropout probability equal to dropout. Default: 0
            bidirectional: If True, becomes a bidirectional LSTM. Default: False
    '''

    def __init__(self, out_channels,
                 hidden_channels, group_num, rnn_type='LSTM', norm='ln',
                 dropout=0, bidirectional=False, num_spks=2):
        super(Grouo_Comm_Dual_RNN_Block, self).__init__()
        # RNN model
        self.inter_group_rnn = getattr(nn, rnn_type)(
            out_channels // group_num, hidden_channels // group_num, 1, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.intra_block_rnn = getattr(nn, rnn_type)(
            out_channels // group_num, hidden_channels // group_num, 1, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.inter_block_rnn = getattr(nn, rnn_type)(
            out_channels // group_num, hidden_channels // group_num, 1, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        # Norm
        self.inter_group_norm = select_norm(norm, out_channels, 4)
        self.intra_block_norm = select_norm(norm, out_channels, 4)
        self.inter_block_norm = select_norm(norm, out_channels, 4)
        # Linear
        self.inter_group_linear = nn.Linear(
            hidden_channels*2 // group_num if bidirectional else hidden_channels // group_num, out_channels // group_num)
        self.intra_block_linear = nn.Linear(
            hidden_channels*2 // group_num if bidirectional else hidden_channels // group_num, out_channels // group_num)
        self.inter_block_linear = nn.Linear(
            hidden_channels*2 // group_num if bidirectional else hidden_channels // group_num, out_channels // group_num)

        self.group_num = group_num        

    def forward(self, x):
        '''
           x: [B, N, K, S]
           out: [Spks, B, N, K, S]
        '''
        B, N, T, S = x.shape
        K = self.group_num
        M = N // K
        assert M * K == N
        x = x.contiguous().view(B, K, M, T, S)

        # inter group RNN
        # B, K, M, 2T, S => B  S  2T  K, M
        # 0  1  2   3  4 => 0  4  3   1  2

        # [B x S x 2T, K, M]
        inter_grouo_rnn = x.permute(0, 4, 3, 1, 2).contiguous().view(B*S*T, K, M)
        # [B x S x 2T, K, M']
        inter_grouo_rnn, _ = self.inter_group_rnn(inter_grouo_rnn)
        # [B, S, 2T, K, M]
        inter_grouo_rnn = self.inter_group_linear(inter_grouo_rnn.contiguous().view(B*S*T*K, -1)).view(B, S, T, K * M)
        # B, S, 2T, N => B, N, 2T, S
        # 0  1  2   3 => 0  3   2  1
        inter_grouo_rnn = inter_grouo_rnn.permute(0, 3, 2, 1)
        # B, K, M, 2T, S
        inter_grouo_rnn = self.inter_group_norm(inter_grouo_rnn).contiguous().view(B, K, M, T, S)
        # B, K, M, 2T, S
        inter_grouo_rnn = inter_grouo_rnn + x


        # intra block RNN
        # B, K, M, 2T, S => B, S, K, 2T, M => B x S x K, 2T, M
        # 0  1  2   3  4 => 0  4  1   3  2 =>  
        intra_rnn = inter_grouo_rnn.permute(0, 4, 1, 3, 2).contiguous().view(B*S*K, T, M)

        # [B x S x K, 2T, M']
        intra_rnn, _ = self.intra_block_rnn(intra_rnn)
        # [B x S x K, 2T, M'] => [B, S, K, T, M]
        intra_rnn = self.intra_block_linear(intra_rnn.contiguous().view(B*S*K*T, -1)).view(B, S, K, T, -1)
        # B, S, K, 2T, M => B, K, M, 2T, S => B, N, 2T, S
        # 0  1  2   3  4 => 0  2  4   3  1 

        # [B, N, 2T, S]
        intra_rnn = intra_rnn.permute(0, 2, 4, 3, 1).contiguous().view(B, N, T, S)
        # B, K, M, 2T, S
        intra_rnn = self.intra_block_norm(intra_rnn).contiguous().view(B, K, M, T, S)
        
        # [B, N, 2T, S]
        intra_rnn = intra_rnn + inter_grouo_rnn

        
        # inter block RNN
        # B, K, M, 2T, S => B, 2T, K, S, M
        # 0  1  2   3  4 => 0   3  1  4  2
        # [B x 2T x K, S, M]
        inter_rnn = intra_rnn.permute(0, 3, 1, 4, 2).contiguous().view(B*T*K, S, M)
        # [B x 2T x K, S, M']
        inter_rnn, _ = self.inter_block_rnn(inter_rnn)
        # [B x 2T x K, S, M'] => [B x 2T x K, S, M] => B x 2T, K, S, M
        inter_rnn = self.inter_block_linear(inter_rnn.contiguous().view(B*T*K*S, -1)).view(B, T, K, S, M)
        # B, 2T, K, S, M ==> B, K, M, 2T, S ==> B, N, 2T, S
        # 0   1  2  3  4 ==> 0  2  4   1  3
        # [B x 2T, S, N]
        inter_rnn = inter_rnn.permute(0, 2, 4, 1, 3).contiguous().view(B, N, T, S)

        inter_rnn = self.inter_block_norm(inter_rnn).contiguous().view(B, K, M, T, S)
        # [B, K, M, 2T, S]
        out = inter_rnn + intra_rnn

        out = out.contiguous().view(B, N, T, S)
        return out


class Dual_Path_RNN(nn.Module):
    '''
       Implementation of the Dual-Path-RNN model
       input:
            in_channels: The number of expected features in the input x
            out_channels: The number of features in the hidden state h
            rnn_type: RNN, LSTM, GRU
            norm: gln = "Global Norm", cln = "Cumulative Norm", ln = "Layer Norm"
            dropout: If non-zero, introduces a Dropout layer on the outputs 
                     of each LSTM layer except the last layer, 
                     with dropout probability equal to dropout. Default: 0
            bidirectional: If True, becomes a bidirectional LSTM. Default: False
            num_layers: number of Dual-Path-Block
            K: the length of chunk
            num_spks: the number of speakers
    '''

    def __init__(self, in_channels, out_channels, hidden_channels,
                 group_num=4, 
                 rnn_type='LSTM', norm='ln', dropout=0, 
                 bidirectional=False, num_layers=4, K=200, num_spks=2):
        super(Dual_Path_RNN, self).__init__()
        self.K = K
        self.num_spks = num_spks
        self.num_layers = num_layers
        self.norm = select_norm(norm, in_channels, 3)
        self.conv1d = nn.Conv1d(in_channels, out_channels, 1, bias=False)

        self.dual_rnn = nn.ModuleList([])
        for i in range(num_layers):
            self.dual_rnn.append(Grouo_Comm_Dual_RNN_Block(out_channels, hidden_channels, group_num=group_num,
                                                           rnn_type=rnn_type, norm=norm, dropout=dropout,
                                                           bidirectional=bidirectional))

        self.conv2d = nn.Conv2d(
            out_channels, out_channels*num_spks, kernel_size=1)
        self.end_conv1x1 = nn.Conv1d(out_channels, in_channels, 1, bias=False)
        self.prelu = nn.PReLU()
        self.activation = nn.ReLU()
         # gated output layer
        self.output = nn.Sequential(nn.Conv1d(out_channels, out_channels, 1),
                                    nn.Tanh()
                                    )
        self.output_gate = nn.Sequential(nn.Conv1d(out_channels, out_channels, 1),
                                         nn.Sigmoid()
                                         )

    def forward(self, x):
        '''
           x: [B, N, L]

        '''
        # [B, N, L]
        x = self.norm(x)
        # [B, N, L]
        x = self.conv1d(x)
        # [B, N, 2T, S]
        x, gap = self._Segmentation(x, self.K)
        # [B, N*spks, 2T, S]
        for i in range(self.num_layers):
            x = self.dual_rnn[i](x)
        x = self.prelu(x)
        x = self.conv2d(x)
        # [B*spks, N, 2T, S]
        B, _, T, S = x.shape
        x = x.view(B*self.num_spks,-1, T, S)
        # [B*spks, N, L]
        x = self._over_add(x, gap)
        x = self.output(x)*self.output_gate(x)
        # [spks*B, N, L]
        x = self.end_conv1x1(x)
        # [B*spks, N, L] -> [B, spks, N, L]
        _, N, L = x.shape
        x = x.view(B, self.num_spks, N, L)
        x = self.activation(x)
        # [spks, B, N, L]
        x = x.transpose(0, 1)
        
        return x

    def _padding(self, input, K):
        '''
           padding the audio times
           K: chunks of length
           P: hop size
           input: [B, N, L]
        '''
        B, N, L = input.shape
        P = K // 2
        gap = K - (P + L % K) % K
        if gap > 0:
            pad = torch.Tensor(torch.zeros(B, N, gap)).type(input.type())
            input = torch.cat([input, pad], dim=2)

        _pad = torch.Tensor(torch.zeros(B, N, P)).type(input.type())
        input = torch.cat([_pad, input, _pad], dim=2)

        return input, gap

    def _Segmentation(self, input, K):
        '''
           the segmentation stage splits
           K: chunks of length
           P: hop size
           input: [B, N, L]
           output: [B, N, K, S]
        '''
        B, N, L = input.shape
        P = K // 2
        input, gap = self._padding(input, K)
        # [B, N, K, S]
        input1 = input[:, :, :-P].contiguous().view(B, N, -1, K)
        input2 = input[:, :, P:].contiguous().view(B, N, -1, K)
        input = torch.cat([input1, input2], dim=3).view(
            B, N, -1, K).transpose(2, 3)

        return input.contiguous(), gap

    def _over_add(self, input, gap):
        '''
           Merge sequence
           input: [B, N, K, S]
           gap: padding length
           output: [B, N, L]
        '''
        B, N, K, S = input.shape
        P = K // 2
        # [B, N, S, K]
        input = input.transpose(2, 3).contiguous().view(B, N, -1, K * 2)

        input1 = input[:, :, :, :K].contiguous().view(B, N, -1)[:, :, P:]
        input2 = input[:, :, :, K:].contiguous().view(B, N, -1)[:, :, :-P]
        input = input1 + input2
        # [B, N, L]
        if gap > 0:
            input = input[:, :, :-gap]

        return input

class Dual_RNN_model(nn.Module):
    '''
       model of Dual Path RNN
       input:
            in_channels: The number of expected features in the input x
            out_channels: The number of features in the hidden state h
            hidden_channels: The hidden size of RNN
            kernel_size: Encoder and Decoder Kernel size
            rnn_type: RNN, LSTM, GRU
            norm: gln = "Global Norm", cln = "Cumulative Norm", ln = "Layer Norm"
            dropout: If non-zero, introduces a Dropout layer on the outputs 
                     of each LSTM layer except the last layer, 
                     with dropout probability equal to dropout. Default: 0
            bidirectional: If True, becomes a bidirectional LSTM. Default: False
            num_layers: number of Dual-Path-Block
            K: the length of chunk
            num_spks: the number of speakers
    '''
    def __init__(self, in_channels, out_channels, hidden_channels,
                 kernel_size=2, rnn_type='LSTM', norm='ln', dropout=0,
                 bidirectional=False, num_layers=4, K=200, num_spks=2):
        super(Dual_RNN_model,self).__init__()
        self.encoder = Encoder(kernel_size=kernel_size,out_channels=in_channels)
        self.separation = Dual_Path_RNN(in_channels, out_channels, hidden_channels,
                 rnn_type=rnn_type, norm=norm, dropout=dropout,
                 bidirectional=bidirectional, num_layers=num_layers, K=K, num_spks=num_spks)
        self.decoder = Decoder(in_channels=in_channels, out_channels=1, kernel_size=kernel_size, stride=kernel_size//2, bias=False)
        self.num_spks = num_spks
    
    def forward(self, x):
        '''
           x: [B, L]
        '''
        # [B, N, L]
        e = self.encoder(x)
        # [spks, B, N, L]
        s = self.separation(e)
        # [B, N, L] -> [B, L]
        out = [s[i]*e for i in range(self.num_spks)]
        audio = [self.decoder(out[i]) for i in range(self.num_spks)]
        return audio

if __name__ == "__main__":
    rnn = Dual_RNN_model(256, 64, 128,bidirectional=True, norm='ln', num_layers=6).cuda()
    #encoder = Encoder(16, 512)
    x = torch.ones(1, 16000).cuda()
    out = rnn(x)
    print("{:.3f}".format(check_parameters(rnn)*1000000))
    print(rnn)
