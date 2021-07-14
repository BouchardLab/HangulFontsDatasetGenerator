import torch
import unittest
from vae import VAE
from torchsummary import summary


class TestVAE(unittest.TestCase):

    def setUp(self) -> None:
        self.model = VAE()

    def test_summary(self):
        print(summary(self.model, (1, 29, 29), device='cpu'))

    def test_encoder(self):
        x = torch.randn(64, 1, 29, 29)
        y = self.model.encode(x)
        print("Encoder output size: ", y.size())

    def test_decoder(self):
        x = torch.randn(10)
        y = self.model.decode(x)
        print("Decoder output size: ", y[0].size())

    def test_forward(self):
        x = torch.randn(64, 1, 29, 29)
        z = x.detach().clone()
        for layer in self.model.enc_convLayer:
            # print(type(layer))
            z = layer(z)
            # print(z.size())
        z = z.view(-1, 4 * 4 * 32)
        for layer in self.model.enc_linLayer:
            #print(type(layer), 'hi')
            z = layer(z)
            # print(z.size())
        mu, logvar = z[:, :10], z[:, 10:]
        #print("bye", mu.shape)
        sample = self.model.reparameterize(mu, logvar)
        i = 0
        for layer in self.model.dec_linLayer:
            if i == 0:
                z = layer(sample)
                i += 1
            else:
                z = layer(z)
        z = z.view(-1, 32, 4, 4)
        for layer in self.model.dec_convLayer:
            z = layer(z)
        z = x.view(x.size())
        y = self.model(x)
        #print("Model Output size:", y[0].size())

    def test_loss(self):
        x = torch.randn(64, 1, 29, 29)

        recon, mu, logvar = self.model(x)
        loss = self.model.final_loss(recon, x, mu, logvar, 1000, 25)
        print(str(loss) + 'hi')


if __name__ == '__main__':
    unittest.main()
