import numpy as np
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img


class ConvGenerator(nn.Module):
    def __init__(self, no_of_channels=1, noise_dim=100, gen_dim=32):
      super(ConvGenerator, self).__init__()

      self.network = nn.Sequential(
          nn.ConvTranspose2d(noise_dim, gen_dim*4, 4, 1, 0, bias=False),
          nn.BatchNorm2d(gen_dim*4),
          nn.ReLU(True),
  
          nn.ConvTranspose2d(gen_dim*4, gen_dim*2, 3, 2, 1, bias=False),
          nn.BatchNorm2d(gen_dim*2),
          nn.ReLU(True),
  
          nn.ConvTranspose2d(gen_dim*2, gen_dim, 4, 2, 1, bias=False),
          nn.BatchNorm2d(gen_dim),
          nn.ReLU(True),
  
          nn.ConvTranspose2d(gen_dim, no_of_channels, 4, 2, 1, bias=False),
          nn.Tanh()
      )
  
    def forward(self, input):
      output = self.network(input)
      return output


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


class ConvDiscriminator(nn.Module):
    def __init__(self, no_of_channels=1, disc_dim=32):
        super(ConvDiscriminator, self).__init__()
        self.network = nn.Sequential(
                
                nn.Conv2d(no_of_channels, disc_dim, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(disc_dim, disc_dim * 2, 4, 2, 1, bias=False),
                nn.InstanceNorm2d(disc_dim * 2, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(disc_dim * 2, disc_dim * 4, 3, 2, 1, bias=False),
                nn.InstanceNorm2d(disc_dim * 4, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(disc_dim * 4, 1, 4, 1, 0, bias=False),
                
            )
    def forward(self, input):
        output = self.network(input)
        #return output.view(-1, 1).squeeze(1)
        return output