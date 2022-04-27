from uuid import uuid1
from datetime import datetime
from pathlib import Path


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
    print(f'num workers: {config.num_workers}') 
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


def get_version():
    model_id = str(uuid1())
    #date = datetime.now().date().strftime("%Y_%m_%d")
    date = datetime.now().time().strftime("%Hh_%Mm_%Ss")
    version = f'{date}/{model_id}'
    return version


def get_checkpoint_path(config, version):

    model_name = config.model_name    
    checkpoint_path = config.checkpoint_path
    uuid_legth = 36
    date_legth = 10

    path = f'{checkpoint_path[:-1]}/{model_name}/{version[:date_legth]}/{version[len(version)-uuid_legth:]}'
    Path(path).mkdir(parents=True, exist_ok=True)

    return path


def save_config(config, version):
    import json
    uuid_legth = 36
    fname = f'{config.config_path}config_model_{version[len(version)-uuid_legth:]}.json'
    with open(fname, 'w') as file:
        file.write(json.dumps(vars(config))) 


def config_from_file(file_name):
    import json
    with open(file_name) as json_file:
        data = json.load(json_file)
    config = ClassFromDict(data)
    return config


class ClassFromDict:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)
        setattr(self, 'flag', None)