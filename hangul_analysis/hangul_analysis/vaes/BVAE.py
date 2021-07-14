import torch
import torch.nn as nn
from torch.nn import Sequential, ModuleList

class Reparameterize(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.h_dim = h_dim
    def forward(self, x):
        mu, logvar = x[:, :self.h_dim], x[:, self.h_dim:]
        std = torch.exp(0.5 * logvar) + 1e-5
        eps = torch.randn_like(std)
        sample = mu + std * eps
        return sample, mu, logvar
    
class BVAE(nn.Module):
    def __init__(self, h_dim, enc_conv, enc_lin, dec_lin, dec_conv, in_layers):
        super().__init__()
        self.h_dim = h_dim
        self.enc_conv = enc_conv
        self.enc_lin = enc_lin
        self.reparam = Reparameterize(h_dim)
        self.dec_lin = dec_lin
        self.dec_conv = dec_conv
        self.enc_lin_input = in_layers[0]
        self.dec_conv_input = [in_layers[1][1], in_layers[1][2], in_layers[1][3]]
    def encode(self, x):
        x = x
        indices = []
        sizes = []
        if type(self.enc_conv) == ModuleList:
            indice = 0
            for l in self.enc_conv:
                if type(l) == ModuleList:
                    for lay in l:
                        x = lay(x)
                else:
                    result = l(x)
                    sizes.append(x.size())
                    x, indice = result
                    indices.append(indice)
        else:        
            x = self.enc_conv(x)
        x = x.view(-1, self.enc_lin_input)
        for l in self.enc_lin:
            x = l(x)
        return x, indices, sizes
    def decode(self, sample, indices, sizes):
        x = sample
        for l in self.dec_lin:
            x = l(x)
        x = x.view(-1, *self.dec_conv_input)
        for l in self.dec_conv:
            if type(l) == ModuleList:
                for lay in l:
                    x = lay(x)
            else:
                indice = indices.pop()
                s = sizes.pop()
                x = l(x, indices=indice, output_size=s)
        return x
    def forward(self, x):
        
        x_size = x.size()
        x, indices, sizes = self.encode(x)
        sample, mu, logvar = self.reparam(x)
        x = self.decode(sample, indices, sizes)
        
        x = x.view(x_size)
        return x, mu, logvar
    def bce_loss(self, reconstruction, x):
        criterion = nn.MSELoss(reduction='mean')
#         criterion = nn.BCEWithLogitsLoss(reduction='mean')
        bce_loss = criterion(reconstruction, x)
        return bce_loss