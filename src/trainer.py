from src.data import DataModule
from src.model import WeatherGenerator
from pytorch_lightning.loggers import TensorBoardLogger

from pytorch_lightning.trainer import Trainer
from src.utils import get_version
from src.configuration import Config

def training(config, enable_profiler=False):

    tb_logger = TensorBoardLogger(config.tensorboard_path,
                                  name=config.model_name,
                                  default_hp_metric=False,
                                  version = get_version())
    
    model = WeatherGenerator(config)
                             
    data = DataModule(config)
    data.setup('fit')

    if enable_profiler:
        profiler = 'simple'
    else:
        profiler = None


    trainer = Trainer(gpus=1,
                     logger=tb_logger,
                     profiler=profiler)
    
    trainer.fit(model, data)
