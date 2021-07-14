import torch
import torch.nn as nn
from torchsummary import summary


class Reparameterize(nn.Module):
    def forward(self, x):
        mu, logvar = x[:, :self.h_dim], x[:, self.h_dim:]
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        sample = mu + std * eps
        return sample, mu, logvar


class VAE(nn.Module):
    """Reimplmentation of paper"""

    def __init__(self, h_dim):
        super().__init__()

        # Encoder
        self.h_dim = h_dim
        self.enc_convLayer = nn.Sequential(
            nn.Conv2d(1, 32, 4, padding=3),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, stride=1),  # B, 32, 28, 28
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, stride=1),  # B, 32, 24, 24
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, stride=1),  # B, 32, 20, 20
            nn.ReLU(),
            nn.Conv2d(32, 16, 4, stride=2, padding=1),  # B, 16, 10, 10
            nn.ReLU(),
            nn.Conv2d(16, 4, 4, stride=2, padding=1),  # B, 4, 5, 5
            nn.ReLU()
        )

        self.reparam_layer = Reparameterize()

        # Decoder
        self.dec_convLayer = nn.Sequential(
            nn.ConvTranspose2d(2, 16, 4, stride=2, padding=1),  # B, 16, 10, 10
            nn.ReLU(),
            nn.ConvTranspose2d(16, 32, 4, stride=2, padding=1),  # B, 32, 20, 20
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 5, stride=1),  # B, 32, 24, 24
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 5, stride=1),  # B, 32, 28, 28
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 5, stride=1),  # B, 32, 32, 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, padding=3)  # B, 32, 29, 29
        )

    def encode(self, x):
        x = self.enc_convLayer(x)
        x = x.view(-1, 100)
        return x

    def decode(self, sample):
        x = sample.view(-1, 2, 5, 5)
        x = self.dec_convLayer(x)
        return x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        sample = mu + std * eps
        return sample

    def forward(self, x):
        x_size = x.size()
        # for layer in self.enc_convLayer:
        #    x = layer(x)
        #    print(x.size())

        x = self.encode(x)
        # split latent layer in half
        mu, logvar = x[:, :self.h_dim], x[:, self.h_dim:]
        sample = self.reparam_layer(mu, logvar)
        x = self.decode(sample)
        x = x.view(x_size)
        return x, mu, logvar

    def final_loss(self, reconstruction, x, mu, logvar, gamma, C):
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        bce_loss = criterion(reconstruction, x)
        kld = gamma * \
            torch.abs((-0.5 * torch.mean(1 + logvar -
                                         mu.pow(2) - logvar.exp(), dim=1) - C).mean(dim=0))
        return bce_loss, kld, bce_loss + kld

class VAEFC(nn.Module):
    """Reimplmentation of paper"""

    def __init__(self, h_dim):
        super().__init__()

        # Encoder
        self.h_dim = h_dim
        self.enc_convLayer = nn.Sequential(
            nn.Conv2d(1, 32, 4, padding=3),
#             nn.Conv2d(1, 32, 4),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, stride=2, padding=1),  # B, 32, 16, 16
#             nn.Conv2d(32, 32, 4),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, stride=2, padding=1),  # B, 32, 8, 8
#             nn.Conv2d(32, 32, 4),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, stride=2, padding=1),  # B, 32, 4, 4
#             nn.Conv2d(32, 32, 4),
            nn.ReLU()
        )

        self.enc_linLayer = nn.Sequential(
            nn.Linear(512, 256),                 # B, 256
#             nn.Linear(18496, 256),                 # B, 256
            nn.ReLU(),
            nn.Linear(256, 256),                        # B, 256
            nn.ReLU(),
            nn.Linear(256, h_dim * 2)                          # B, 20
        )

        # Decoder
        self.dec_linLayer = nn.Sequential(
            nn.Linear(h_dim, 256),  # B, 256
            nn.ReLU(),
            nn.Linear(256, 256),  # B, 256
            nn.ReLU(),
            nn.Linear(256, 512),  # B, 512
#             nn.Linear(256, 18496),                 # B, 256
            
            nn.ReLU()
        )

        self.dec_convLayer = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),  # B, 32, 8, 8
#             nn.ConvTranspose2d(32, 32, 4),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, stride=2,
                               padding=1),                      # B, 32, 16, 16
#             nn.ConvTranspose2d(32, 32, 4),
            nn.ReLU(),
            # B, 32, 32, 32
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),
#             nn.ConvTranspose2d(32, 32, 4),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, padding=3)
#             nn.ConvTranspose2d(32, 1, 4)
        )

    def encode(self, x):
        x = self.enc_convLayer(x)                       # Encode to B, 32, 4, 4
        print(x.size())
        x = x.view(-1, 512)                      # B, 512
#         x = x.view(-1, 18496)
        x = self.enc_linLayer(x)
        return x

    def decode(self, sample):
        print(sample.size())
        x = self.dec_linLayer(sample)  # B, 512
        x = x.view(-1, 32, 4, 4)  # B, 32, 4, 4
#         x = x.view(-1, 32, 17, 17)
        x = self.dec_convLayer(x)
        return x

    def reparameterize(self, x):
        mu, logvar = x[:, :self.h_dim], x[:, self.h_dim:]
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        sample = mu + std * eps
        return sample, mu, logvar

    def forward(self, x):
        x_size = x.size()
        # for layer in self.enc_convLayer:
        #    x = layer(x)
        #    print(x.size())

        x = self.encode(x)
        # split latent layer in half
        # mu, logvar = x[:, :self.h_dim], x[:, self.h_dim:]
        sample, mu, logvar = self.reparameterize(x)
        x = self.decode(sample)
        x = x.view(x_size)
        return x, mu, logvar

    def final_loss(self, reconstruction, x, mu, logvar, gamma, C):
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        bce_loss = criterion(reconstruction, x)
        kld = gamma * \
            torch.abs((-0.5 * torch.mean(1 + logvar -
                                         mu.pow(2) - logvar.exp(), dim=1) - C).mean(dim=0))
        return bce_loss, kld, bce_loss + kld

v = VAEFC(80)
print(summary(v, (1, 29, 29)))