from typing import List
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class Config:

    home = str(Path.home())
    if home == '/home/hess': 
        scratch_path: str = '/p/tmp/hess/scratch/weather-gan'

    if home == '/home/ftei-dsw': 
        scratch_path: str = '/home/ftei-dsw/data/weather-gan'
    
    tensorboard_path: str = f'{scratch_path}/tensorboard/'
    checkpoint_path: str = f'{scratch_path}/checkpoints/'
    config_path: str = f'{scratch_path}/config-files/'
    results_path: str = f'{scratch_path}/results/'
    input_fname: str = f'{scratch_path}/datasets/monthly_gfdl_historical.nc'
    target_fname: str = f'{scratch_path}/datasets/daily_gfdl_historical.nc'

    #train_start: int = 1900
    train_start: int = 1950
    train_end: int = 2000
    valid_start: int = 2001
    #valid_end: int = 2014
    valid_end: int = 2001
    test_start: int = 2001
    #test_end: int = 2014
    test_end: int = 2001

    num_workers = 8
    prefetch_factor = 2
    
    model_name: str = 'weather-gan'
    
    num_variables: int = 2
    #epochs: int = 250
    epochs: int = 1
    train_batch_size: int = 32
    test_batch_size: int = 64
    lr: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999
    
    latent_dim: int = 8
    apply_dropout: bool = False
    generator_channels: int = 64
    generator_num_resblocks: int = 5
    generator_num_downsampling: int = 2

    discriminator_channels: int = 32
    discriminator_num_layers = 3
    n_critic = 5