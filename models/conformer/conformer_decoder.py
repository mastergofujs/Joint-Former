from ast import Str
import imp
import torch
from torch import nn
import math
from models.conformer.conformer_block import ConformerBlock
from models.conformer.repeat import repeat
from models.conformer.masked_positional_encoding import MaskedPositionalEncoding

class Unsqueeze(torch.nn.Module):
    def __init__(self, dim=1):
        super(Unsqueeze, self).__init__()
        self.dim = dim
    def forward(self, x):
        return x.unsqueeze(self.dim)

class Squeeze(torch.nn.Module):
    def __init__(self, dim=1):
        super(Squeeze, self).__init__()
        self.dim = dim
    def forward(self, x):
        return x.squeeze(self.dim)

class Permute(torch.nn.Module):
    def __init__(self, dim):
        super(Permute, self).__init__()
        self.dim = dim
    def forward(self, x):
        return x.permute(self.dim)

class GLU(nn.Module):
    def __init__(self, input_num):
        super(GLU, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(input_num, input_num)

    def forward(self, x):
        lin = self.linear(x.permute(0, 2, 3, 1))
        lin = lin.permute(0, 3, 1, 2)
        sig = self.sigmoid(x)
        res = lin * sig
        return res

class ContextGating(nn.Module):
    def __init__(self, input_num):
        super(ContextGating, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(input_num, input_num)

    def forward(self, x):
        lin = self.linear(x.permute(0, 2, 3, 1))
        lin = lin.permute(0, 3, 1, 2)
        sig = self.sigmoid(lin)
        res = x * sig
        return res

class CNNUpsampler(torch.nn.Module):
    def __init__(
        self,
        n_in_channel,
        activation="Relu",
        conv_dropout=0,
        kernel_size=[3, 3, 3],
        padding=[1, 1, 1],
        stride=[2, 2, 2],
        nb_filters=[128, 128, 64],
    ):
        super(CNNUpsampler, self).__init__()
        self.nb_filters = nb_filters
        cnn = torch.nn.Sequential()

        def conv(i, batchNormalization=False, dropout=None, activ="relu"):
            nIn = n_in_channel if i == 0 else nb_filters[i - 1]
            nOut = nb_filters[i]
            cnn.add_module("conv{0}".format(i), nn.ConvTranspose1d(nIn, nOut, kernel_size[i], stride[i], padding[i]))
            if batchNormalization:
                cnn.add_module("batchnorm{0}".format(i), nn.BatchNorm1d(nOut, eps=0.001, momentum=0.99))
            if activ.lower() == "leakyrelu":
                cnn.add_module("relu{0}".format(i), nn.LeakyReLU(0.2))
            elif activ.lower() == "relu":
                cnn.add_module("relu{0}".format(i), nn.ReLU())
            elif activ.lower() == "glu":
                cnn.add_module("glu{0}".format(i), GLU(nOut))
            elif activ.lower() == "cg":
                cnn.add_module("cg{0}".format(i), ContextGating(nOut))
            else:
                pass
            if dropout is not None:
                cnn.add_module("dropout{0}".format(i), nn.Dropout(dropout))

        batch_norm = True
        for i in range(len(nb_filters)):
            conv(i, batch_norm, conv_dropout, activ=activation)
        self.cnn = cnn

    def forward(self, x):
        # input size : (batch_size, n_channels, n_frames, n_freq)
        # conv features
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        return x


class ConformerMaskedDecoder(torch.nn.Module):
    # def __init__(self, n_stacks, d_model, d_ff, n_head, dropout):
    def __init__(
        self,
        idim: int,
        adim: int = 144,
        fdim: int = 64,
        dropout_rate: float = 0.1,
        elayers: int = 2,
        eunits: int = 576,
        aheads: int = 4,
        kernel_size: int = 7,
        cnn_upsampler: dict = None
    ):
        super(ConformerMaskedDecoder, self).__init__()
        assert adim % aheads == 0
        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(idim, adim),
            torch.nn.LayerNorm(adim),
            torch.nn.Dropout(dropout_rate),
            torch.nn.ReLU())
        self.pe = MaskedPositionalEncoding(adim, dropout_rate)
        self.conformer_blocks = repeat(elayers, lambda: ConformerBlock(adim, eunits, aheads, dropout_rate, kernel_size))
        self.upsampling = CNNUpsampler(n_in_channel=144, **cnn_upsampler)
        self.token_trans = torch.nn.Linear(adim, fdim)
        self.out_layer = nn.Sequential(
            Permute((0, 2, 1)),
            torch.nn.BatchNorm1d(fdim),
            Permute((0, 2, 1)),
            torch.nn.Linear(fdim, fdim)
        )
        self.mask_embed = torch.nn.Parameter(torch.zeros([1, 1, idim])) # 144 is out-dim of backbone
        self.mask_embed = torch.nn.init.xavier_normal_(self.mask_embed)

    def forward(self, x, inds, cls_token=True):
        masked_id, unmasked_id = inds[0], inds[1]
        b, nf, d = x.size()
        if cls_token:
            # Note: here input x is unmasked with cls_token
            inds_all = torch.cat([masked_id + 1, torch.zeros(1, dtype=torch.int64), unmasked_id + 1]) 
        else:
            inds_all = torch.cat([masked_id, unmasked_id])
        x_u = torch.zeros((b, len(inds_all), d)).cuda()
        unmasked_ = torch.cat([torch.zeros(1, dtype=torch.int64), unmasked_id.sort()[0] + 1])
        x_u[:, unmasked_, :] = x
        x_u[:, masked_id.sort()[0] + 1, :] = self.mask_embed
        x_u = self.input_layer(x_u)
        x_u = self.pe(x_u, inds_all.sort()[0])
        x_u, _ = self.conformer_blocks(x_u, None)
        x_u_f = self.upsampling(x_u[:, 1:, :])
        x_u_cls = self.token_trans(x_u[:, 0, :].unsqueeze(1))
        x_u = self.out_layer(x_u_f)
        # x_u = torch.tanh(x_u)
        return x_u

class MaskedLinearDecoder(torch.nn.Module):
    def __init__(
        self,
        idim: int,
        adim: int,
        dropout_rate: float = 0.1
    ):
        super(MaskedLinearDecoder, self).__init__()
        self.decoder_layer = torch.nn.Sequential(
            torch.nn.Linear(idim, adim),
            torch.nn.LayerNorm(adim),
            torch.nn.Dropout(dropout_rate),
            torch.nn.ReLU(),
            torch.nn.Linear(adim, adim),
            torch.nn.Tanh()
        )
        self.mask_embed = torch.nn.Parameter(torch.zeros([1, 1, idim])) # 144 is out-dim of backbone
        self.mask_embed = torch.nn.init.xavier_normal_(self.mask_embed)

    def forward(self, x, masked_inds, unmasked_inds):
        b, nf, d = x.size()
        shuffled_inds = torch.cat([masked_inds, unmasked_inds])
        x = torch.cat([self.mask_embed.repeat((b, len(masked_inds), 1)), x], dim=1)
        unshuffled_x = torch.zeros_like(x).cuda()
        unshuffled_x[:, shuffled_inds, :] = x
        x = self.decoder_layer(unshuffled_x)
        return x
    
    