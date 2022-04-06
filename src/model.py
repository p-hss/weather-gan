import numpy as np
import torch.nn as nn

class MLPGenerator(nn.Module):
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


class ResBlock(nn.Module):

    def __init__(self, in_channels: int, apply_dp: bool = True):

        """
                            Defines a ResBlock
        X ------------------------identity------------------------
        |-- Convolution -- Norm -- ReLU -- Convolution -- Norm --|
        """

        """
        Parameters:
            in_channels:  Number of input channels
            apply_dp:     If apply_dp is set to True, then activations are 0'ed out with prob 0.5
        """

        super().__init__()

        conv = nn.Conv2d(in_channels = in_channels, out_channels = in_channels, kernel_size = 3, stride = 1)
        layers =  [nn.ReflectionPad2d(1), conv, nn.InstanceNorm2d(in_channels), nn.ReLU(True)]

        if apply_dp:
            layers += [nn.Dropout(0.5)]

        conv = nn.Conv2d(in_channels = in_channels, out_channels = in_channels, kernel_size = 3, stride = 1)
        layers += [nn.ReflectionPad2d(1), conv, nn.InstanceNorm2d(in_channels)]

        self.net = nn.Sequential(*layers)


    def forward(self, x): return x + self.net(x)


class Generator(nn.Module):

    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 64,
                 apply_dp: bool = True,
                 num_resblocks = 3,
                 num_downsampling = 2
                 ):

        """
                                Generator Architecture (Image Size: 256)
        c7s1-64, d128, d256, R256, R256, R256, R256, R256, R256, R256, R256, R256, u128, u64, c7s1-3,
        where c7s1-k denote a 7 × 7 Conv-InstanceNorm-ReLU layer with k filters and stride 1, dk denotes a 3 × 3
        Conv-InstanceNorm-ReLU layer with k filters and stride 2, Rk denotes a residual block that contains two
        3 × 3 Conv layers with the same number of filters on both layer. uk denotes a 3 × 3 DeConv-InstanceNorm-
        ReLU layer with k filters and stride 1.
        """

        """
        Parameters:
            in_channels:  Number of input channels
            out_channels: Number of output channels
            apply_dp:     If apply_dp is set to True, then activations are 0'ed out with prob 0.5
        """

        super().__init__()

        f = 1

        conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 7, stride = 1)
        self.layers = [nn.ReflectionPad2d(3), conv, nn.InstanceNorm2d(out_channels), nn.ReLU(True)]

        for i in range(num_downsampling):
            conv = nn.Conv2d(out_channels * f, out_channels * 2 * f, kernel_size = 3, stride = 2, padding = 1)
            self.layers += [conv, nn.InstanceNorm2d(out_channels * 2 * f), nn.ReLU(True)]
            f *= 2

        for i in range(num_resblocks):
            res_blk = ResBlock(in_channels = out_channels * f, apply_dp = apply_dp)
            self.layers += [res_blk]

        for i in range(num_downsampling):
            conv = nn.ConvTranspose2d(out_channels * f, out_channels * (f//2), 3, 2, padding = 1, output_padding = 1)
            self.layers += [conv, nn.InstanceNorm2d(out_channels * (f//2)), nn.ReLU(True)]
            f = f // 2

        conv = nn.Conv2d(in_channels = out_channels, out_channels = in_channels, kernel_size = 7, stride = 1)
        self.layers += [nn.ReflectionPad2d(3), conv, nn.Tanh()]

        self.net = nn.Sequential(*self.layers)


    def forward(self, x): 
       
        x = self.net(x)

        return x 

class ConvGenerator(nn.Module):
    def __init__(self, no_of_channels=1, input_dim=100, gen_dim=32):
      super(ConvGenerator, self).__init__()

      self.network = nn.Sequential(
          nn.ConvTranspose2d(input_dim, gen_dim*4, 4, 1, 0, bias=False),
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

    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 64,
                 num_layers: int = 3):

        """
                                    Discriminator Architecture!
        C64 - C128 - C256 - C512, where Ck denote a Convolution-InstanceNorm-LeakyReLU layer with k filters
        """

        """
        Parameters:
            in_channels:    Number of input channels
            out_channels:   Number of output channels
            num_layers:     Number of layers in the 70*70 Patch Discriminator
        """

        super().__init__()
        in_f  = 1
        out_f = 2

        conv = nn.Conv2d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1)
        self.layers = [conv, nn.LeakyReLU(0.2, True)]

        for idx in range(1, num_layers):
            conv = nn.Conv2d(out_channels * in_f, out_channels * out_f, kernel_size = 4, stride = 2, padding = 1)
            self.layers += [conv, nn.InstanceNorm2d(out_channels * out_f), nn.LeakyReLU(0.2, True)]
            in_f   = out_f
            out_f *= 2

        out_f = min(2 ** num_layers, 8)
        conv = nn.Conv2d(out_channels * in_f,  out_channels * out_f, kernel_size = 4, stride = 1, padding = 1)
        self.layers += [conv, nn.InstanceNorm2d(out_channels * out_f), nn.LeakyReLU(0.2, True)]

        conv = nn.Conv2d(out_channels * out_f, out_channels = 1, kernel_size = 4, stride = 1, padding = 1)
        self.layers += [conv]

        self.net = nn.Sequential(*self.layers)


    def forward(self, x):
        out = self.net(x)
        out = out.view(out.size(0), -1)
        return out 


class MLPDiscriminator(nn.Module):
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
