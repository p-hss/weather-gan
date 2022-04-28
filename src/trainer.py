from src.data import DataModule
from src.model import WeatherGenerator
from pytorch_lightning.loggers import TensorBoardLogger

from pytorch_lightning.trainer import Trainer
from src.utils import get_version, save_config, show_config, get_checkpoint_path
from src.configuration import Config
from src.callbacks import get_callbacks
from src.dataloader import PyTorchDataModule

def training(config, enable_profiler=False):
    """ Main training function """

    version = get_version() # generate uuid and date string
    checkpoint_path = get_checkpoint_path(config, version)
    callbacks = get_callbacks(checkpoint_path)

    print(f'Running model: {config.model_name}/{version}')
    print(f'Checkpoint path: {config.checkpoint_path}/{config.model_name}/{version}')

    show_config(config)
    save_config(config, version)

    tb_logger = TensorBoardLogger(config.tensorboard_path,
                                  name=config.model_name,
                                  default_hp_metric=False,
                                  version=version)

    if enable_profiler:
        profiler = 'simple'
    else:
        profiler = None

    trainer = Trainer(gpus=1,
                     max_epochs=config.epochs, 
                     logger=tb_logger,
                     callbacks=callbacks,
                     profiler=profiler)

    model = WeatherGenerator(config)
    data = PyTorchDataModule(config)
    
    trainer.fit(model, data)
