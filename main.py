from argparse import ArgumentParser, Namespace
from src.data import DataModule
from src.trainer import WeatherGenerator
from pytorch_lightning.loggers import TensorBoardLogger

from pytorch_lightning.trainer import Trainer
from src.utils import get_version
from src.configuration import Config


def main(args: Namespace) -> None:
    config = Config()
    training(config)


def training(config):

    tb_logger = TensorBoardLogger(config.tensorboard_path,
                                  name=config.model_name,
                                  default_hp_metric=False,
                                  version = get_version())
    
    model = WeatherGenerator(config)
                             
    data = DataModule(config)
    data.setup('fit')
    
    trainer = Trainer(gpus=1,
                     logger=tb_logger)
    
    trainer.fit(model, data)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int, default=0, help="number of GPUs")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=100,
                        help="dimensionality of the latent space")

    hparams = parser.parse_args()

    main(hparams)