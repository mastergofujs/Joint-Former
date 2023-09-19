import torch
from torch import nn
from models.conformer.attention import PositionalEncoding, RelMultiHeadAttn
from models.conformer.convolution import ConvolutionModule
from models.conformer.macaron_feed_forward import MacaronFeedForward
from fvcore.nn import FlopCountAnalysis

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
        
class CNN(nn.Module):
    def __init__(
        self,
        n_in_channel,
        activation="Relu",
        conv_dropout=0,
        kernel_size=[3, 3, 3],
        padding=[1, 1, 1],
        stride=[1, 1, 1],
        nb_filters=[64, 64, 64],
        pooling=[(1, 4), (1, 4), (1, 4)],
    ):
        super(CNN, self).__init__()
        self.nb_filters = nb_filters
        cnn = nn.Sequential()

        def conv(i, batchNormalization=False, dropout=None, activ="relu"):
            nIn = n_in_channel if i == 0 else nb_filters[i - 1]
            nOut = nb_filters[i]
            cnn.add_module("conv{0}".format(i), nn.Conv2d(nIn, nOut, kernel_size[i], stride[i], padding[i]))
            if batchNormalization:
                cnn.add_module("batchnorm{0}".format(i), nn.BatchNorm2d(nOut, eps=0.001, momentum=0.99))
            if activ.lower() == "leakyrelu":
                cnn.add_module("relu{0}".format(i), nn.LeakyReLU(0.2))
            elif activ.lower() == "relu":
                cnn.add_module("relu{0}".format(i), nn.ReLU())
            elif activ.lower() == "glu":
                cnn.add_module("glu{0}".format(i), GLU(nOut))
            elif activ.lower() == "cg":
                cnn.add_module("cg{0}".format(i), ContextGating(nOut))
            if dropout is not None:
                cnn.add_module("dropout{0}".format(i), nn.Dropout(dropout))

        batch_norm = True
        for i in range(len(nb_filters)):
            conv(i, batch_norm, conv_dropout, activ=activation)
            if i < len(nb_filters) -1:
                cnn.add_module("pooling{0}".format(i), nn.AvgPool2d(pooling[i]))  # bs x tframe x mels
        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.cnn = cnn

    def forward(self, x):
        # input size : (batch_size, n_channels, n_frames, n_freq)
        # conv features
        x = self.cnn(x)
        x = self.gap(x)
        return x


class CNNLocalDownsampler(torch.nn.Module):
    def __init__(self, patchsize=8, **cnn_kwargs):
        super(CNNLocalDownsampler, self).__init__()
        self.patchsize = patchsize
        self.cnn = CNN(**cnn_kwargs)
        
    def temporal_patchifying(self, input):
        b, nf, d = input.squeeze(1).size()
        npatchs = nf // self.patchsize
        patchs = input.view(b, npatchs, 1, self.patchsize, d)
        return patchs

    def patch_masking(self, x, mask_size=1, mask_ratio=0.75):
        b, np, c, ps, d = x.size()
        masked_p = int((np // mask_size) * mask_ratio)
        shuffled_ind = torch.randperm(np // mask_size)
        masked_inds = shuffled_ind[:masked_p]
        unmasked_inds = shuffled_ind[masked_p:]
        unmasked_patchs = x[:, unmasked_inds, :]
        return unmasked_patchs, masked_inds, unmasked_inds

    def forward(self, x, mask=False, **mask_kwargs):
        # x = x.repeat(1, 1, 2, 1)
        x = self.temporal_patchifying(x)
        if mask:
            x, masked_inds, unmasked_inds = self.patch_masking(x, **mask_kwargs)
        b, p, c, t, d = x.size() # batch size, p
        x = x.view(b * p, c, t, d)
        # flops = FlopCountAnalysis(self.cnn, x)
        x = self.cnn(x)
        x = x.view(b, p, -1)
        if mask:
            return x, masked_inds, unmasked_inds
        else:
            return x

class CNNDownsamplerWithMask(torch.nn.Module):
    def __init__(self, patchsize=8, **cnn_kwargs):
        super(CNNDownsamplerWithMask, self).__init__()
        self.patchsize = patchsize
        self.cnn = CNN(**cnn_kwargs)

    def patch_masking(self, x, mask_size=1, mask_ratio=0.75):
        b, nf, d = x.size()
        shuffled_ind = torch.randperm(nf)
        masked_inds = shuffled_ind[:int(mask_ratio * nf)]
        unmasked_inds = shuffled_ind[int(mask_ratio * nf):]
        unmasked_patchs = x[:, unmasked_inds, :]
        return unmasked_patchs, masked_inds, unmasked_inds

    def forward(self, x, mask=False, **mask_kwargs):
        x = self.cnn(x)
        x = x.permute((0, 2, 1, 3)).squeeze()
        if mask:
            x, masked_inds, unmasked_inds = self.patch_masking(x, **mask_kwargs)
        if mask:
            return x, masked_inds, unmasked_inds
        else:
            return x

class ConformerDownsamplerBlock(torch.nn.Module):
    def __init__(self, d_model, d_ff, n_head, dropout, kernel_size, pool_size=1):
        super(ConformerDownsamplerBlock, self).__init__()
        self.ffn1 = MacaronFeedForward(d_model, d_ff, dropout)
        self.mhsa = RelMultiHeadAttn(n_head, d_model, dropout)
        self.conv = ConvolutionModule(d_model, dropout, kernel_size)
        self.pool = torch.nn.AvgPool2d(kernel_size=(pool_size, 1), stride=(pool_size, 1))
        self.ffn2 = MacaronFeedForward(d_model, d_ff, dropout)
        self.norm = torch.nn.LayerNorm(d_model)
        self.pe = PositionalEncoding(d_model, dropout)
        
    def forward(self, x, mask=None):
        x = 0.5 * self.ffn1(x) + x
        x = x.permute(1, 0, 2)  # (B, T, D)
        x = self.mhsa(x, mask)  # (T, B, D)
        x = x.permute(1, 0, 2)  # (B, T, D)
        x = self.conv(x) + x
        token = x[:, 0]
        x = self.pool(x[:, 1:])
        x = torch.cat([token.unsqueeze(1), x], dim=1)
        x = 0.5 * self.ffn2(x) + x
        x = self.norm(x)
        return x, mask