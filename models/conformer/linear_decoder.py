import torch
from torch.nn.functional import interpolate

class LinearDecoder(torch.nn.Module):
    def __init__(
        self,
        idim: int,
        adim: int,
        dropout_rate: float = 0.1
    ):
        super(LinearDecoder, self).__init__()
        self.decoder_layer = torch.nn.Sequential(
            torch.nn.Linear(idim, adim),
            torch.nn.LayerNorm(adim),
            torch.nn.Dropout(dropout_rate),
            torch.nn.ReLU(),
            torch.nn.Linear(adim, adim),
            torch.nn.Tanh()
        )
    
    def forward(self, x):
        x = interpolate(
            x.transpose(1, 2),
            x.shape[1] * 8,
            mode='linear',
            align_corners=False).transpose(1, 2)
        x = self.decoder_layer(x)
        return x

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
        inds = torch.cat([masked_inds, unmasked_inds]).sort()[0]
        # x = torch.cat([self.mask_embed.repeat((b, len(masked_inds), 1)), x], dim=1)
        emb_x = torch.zeros((b, len(inds) + 1, d)).cuda()
        emb_x[:, masked_inds.sort()[0] + 1, :] = self.mask_embed
        emb_x[:, unmasked_inds.sort()[0] + 1, :] = x[:, 1:, :]
        emb_x[:, 0, :] = x[:, 0]
        x = self.decoder_layer(emb_x)[:, 1:, :]
        return x
