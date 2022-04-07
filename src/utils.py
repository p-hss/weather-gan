from uuid import uuid1
from datetime import datetime
import os
import time

def get_version():
    model_id = str(uuid1())
    #date = datetime.now().date().strftime("%Y_%m_%d")
    date = datetime.now().time().strftime("%Hh_%Mm_%Ss")
    version = f'{date}/{model_id}'
    return version


def show_config(config):
    print(f'################ configuration ################')
    print(f'-------------------- paths --------------------')
    print(f'tensorbard: {config.tensorboard_path}')
    print(f'checkpoint: {config.checkpoint_path}')
    print(f'config: {config.config_path}')
    print(f'results: {config.results_path}')
    print(f'input: {config.input_fname}')
    print(f'target: {config.target_fname}')
    print(f'-------------------- splits  -------------------')
    print(f'training:   {config.train_start} - {config.train_end}')
    print(f'validation: {config.valid_start} - {config.valid_end}')
    print(f'test:       {config.test_start} - {config.test_end}')
    print(f'------------------ hyperparams -----------------')
    print(f'epochs: {config.epochs}') 
    print(f'train batch size: {config.train_batch_size}') 
    print(f'test batch size: {config.test_batch_size}') 
    print(f'lr: {config.lr}') 
    print(f'latent dim: {config.latent_dim}') 
    print(f'generator channels: {config.generator_channels}') 
    print(f'generator num. residual blocks: {config.generator_num_resblocks}') 
    print(f'generator num. downsampling: {config.generator_num_downsampling}') 
    print(f'apply dropout: {config.apply_dropout}') 
    print(f'discriminator channels: {config.discriminator_channels}') 
    print(f'discriminator num layers: {config.discriminator_num_layers}') 
    print(f'num discriminator steps: {config.n_critic}') 