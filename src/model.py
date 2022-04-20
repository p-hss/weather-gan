import torch.nn as nn
from collections import OrderedDict
import torch
import torchvision
from pytorch_lightning.core import LightningModule
from torch.autograd import grad as torch_grad
from torch.autograd import Variable


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
                 latent_dim: int = 0,
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

        conv = nn.Conv2d(in_channels = in_channels+latent_dim, out_channels = out_channels, kernel_size = 7, stride = 1)
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


class WeatherGenerator(LightningModule):

    def __init__(self,
                 config):

        super().__init__()

        self.save_hyperparameters()

        self.latent_dim = config.latent_dim # noise dim
        self.lr = config.lr
        self.b1 = config.beta1 # beta for optimizer
        self.b2 = config.beta2
        self.n_critic = config.n_critic

        # networks
        self.generator = Generator(in_channels=config.num_variables,
                                   out_channels=config.generator_channels, 
                                   latent_dim=config.latent_dim,
                                   apply_dp=config.apply_dropout,
                                   num_resblocks=config.generator_num_resblocks,
                                   num_downsampling=config.generator_num_downsampling)

        self.discriminator = Discriminator(in_channels=config.num_variables,
                                           out_channels=config.discriminator_channels,
                                           num_layers=config.discriminator_num_layers)


    def forward(self, z):
        return self.generator(z)

    def gradient_penalty(self, real_data, generated_data):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1).to(self.device)
        alpha = alpha.expand_as(real_data)

        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True)


        # Calculate probability of interpolated examples
        prob_interpolated = self.discriminator(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).to(self.device),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)
        #self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().data[0])

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return ((gradients_norm - 1) ** 2).mean()    


    def training_step(self, batch, batch_idx, optimizer_idx):
        #input = batch[0]['input']
        #target = batch[0]['target']

        input = batch[0]
        target = batch[1]

        # sample noise
        z = torch.randn(input.shape[0], self.latent_dim, input.shape[2],  input.shape[3])
        z = z.type_as(input)
        input = torch.cat([input, z], dim=1)

        lambda_gp = 10

        # train generator
        if optimizer_idx == 0:

            # generate images
            self.generated_imgs = self(input)

            # log sampled images
            grid = torchvision.utils.make_grid(self.generated_imgs[0,0].unsqueeze(0), nrow=1)
            self.logger.experiment.add_image('generated_precipitation', grid, self.current_epoch,
                                             dataformats="CHW")
            grid = torchvision.utils.make_grid(target[1,0], nrow=1)
            self.logger.experiment.add_image('target_precipitation', grid, self.current_epoch,
                                             dataformats="CHW")
            grid = torchvision.utils.make_grid(self.generated_imgs[0,1], nrow=1)
            self.logger.experiment.add_image('generated_temperature', grid, self.current_epoch,
                                             dataformats="CHW")
            grid = torchvision.utils.make_grid(target[0,1], nrow=1)
            self.logger.experiment.add_image('target_temperature', grid, self.current_epoch,
                                             dataformats="CHW")

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(input.size(0), 1)
            valid = valid.type_as(input)

            generated_fields = self(input)

            g_loss = -torch.mean(self.discriminator(generated_fields))
            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            self.log("g_loss", g_loss.detach(),
                     on_step = True,
                     on_epoch = True,
                     prog_bar = True,
                     logger = True)
            return output

        # train discriminator
        # Measure discriminator's ability to classify real from generated samples
        elif optimizer_idx == 1:
            fake_imgs = self(input)

            # Real images
            real_validity = self.discriminator(target)
            # Fake images
            fake_validity = self.discriminator(fake_imgs)
            # Gradient penalty
            gradient_penalty = self.gradient_penalty(target.data, fake_imgs.data)
            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })

            self.log("d_loss", d_loss.detach(),
                     on_step = True,
                     on_epoch = True,
                     prog_bar = True,
                     logger = True)
            return output


    def configure_optimizers(self):

        lr = self.lr
        b1 = self.b1
        b2 = self.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return (
            {'optimizer': opt_g, 'frequency': 1},
            {'optimizer': opt_d, 'frequency': self.n_critic}
        )





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
