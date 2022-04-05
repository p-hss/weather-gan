from argparse import ArgumentParser, Namespace
from src.trainer import WGANGP, Trainer
from src.data import DataModule
from pytorch_lightning.trainer import Trainer

def main(args: Namespace) -> None:
    model = WGANGP(**vars(args))

    data = DataModule()
    data.setup('fit')

    trainer = Trainer(gpus=args.gpus)

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