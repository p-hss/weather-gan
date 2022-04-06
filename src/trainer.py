import os
from collections import OrderedDict
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from pytorch_lightning.core import LightningModule
from torch.autograd import grad as torch_grad
from torch.autograd import Variable

from src.model import Generator, Discriminator

class WeatherGenerator(LightningModule):

    def __init__(self,
                 latent_dim: int = 100,
                 lr: float = 0.0002,
                 b1: float = 0.5,
                 b2: float = 0.999,
                 batch_size: int = 64, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.latent_dim = latent_dim # noise dim
        self.lr = lr
        self.b1 = b1 # beta for optimizer
        self.b2 = b2
        self.batch_size = batch_size
        self.n_critic = 5

        # networks
        self.generator = Generator(in_channels=2,
                                   out_channels=64, #hidden state
                                   latent_dim = self.latent_dim,
                                   apply_dp=True,
                                   num_resblocks=3,
                                   num_downsampling=2)

        self.discriminator = Discriminator(in_channels=2,
                                           out_channels=32,
                                           num_layers=2)

        #self.validation_z = torch.randn(8, self.latent_dim)


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

    #def compute_gradient_penalty(self, real_samples, fake_samples):
    #    """Calculates the gradient penalty loss for WGAN GP"""
    #    print(real_samples.shape, fake_samples.shape)

    #    # Random weight term for interpolation between real and fake samples
    #    alpha = torch.Tensor(np.random.random(real_samples.shape)).to(self.device)
    #    # Get random interpolation between real and fake samples
    #    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    #    interpolates = interpolates.to(self.device)
    #    d_interpolates = self.discriminator(interpolates)
    #    fake = torch.Tensor(real_samples.shape).fill_(1.0).to(self.device)
    #    # Get gradient w.r.t. interpolates
    #    gradients = torch.autograd.grad(
    #        outputs=d_interpolates,
    #        inputs=interpolates,
    #        grad_outputs=fake,
    #        create_graph=True,
    #        retain_graph=True,
    #        only_inputs=True,
    #    )[0]
    #    gradients = gradients.view(gradients.size(0), -1).to(self.device)
    #    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    #    return gradient_penalty


    def training_step(self, batch, batch_idx, optimizer_idx):
        input = batch[0]['input']
        target = batch[0]['target']

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
            sample_imgs = self.generated_imgs[0,0]
            grid = torchvision.utils.make_grid(sample_imgs,  nrow=1)
            self.logger.experiment.add_image('generated_images', grid, self.current_epoch, dataformats = "CHW")

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


    #def on_epoch_end(self):

    #    z = torch.randn(imgs.shape[0], imgs.shape[1],imgs.shape[2],  self.latent_dim)

    #    # log sampled images
    #    sample_imgs = self(z)
    #    grid = torchvision.utils.make_grid(sample_imgs)
    #    self.logger.experiment.add_image('generated_images', grid, self.current_epoch)