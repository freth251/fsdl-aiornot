import argparse

from data.dataset import AIOrNotDataset
from datasets import load_dataset
from lightning.pytorch.profilers import PassThroughProfiler, PyTorchProfiler
from models.resnet import ResnetModel
import pytorch_lightning as pl
import torch
from torchvision import transforms


def orderTensor(x):
    x = x.to(torch.float32)
    return x


def _setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(prog="Train_Model")
    parser.add_argument("-b", "--batch_size", help="batch size", type=int, default=1)
    parser.add_argument(
        "-d", "--log_dir", help="log directory", type=str, default="./logs"
    )
    parser.add_argument("-f", "--log_freq", help="log frequency", type=int, default=100)
    parser.add_argument(
        "-m", "--max_epochs", help="maximum training epochs", type=int, default=1
    )
    parser.add_argument(
        "-n",
        "--num_workers",
        help="number of workers for dataloader",
        type=int,
        default=1,
    )

    parser.add_argument(
        "-o",
        "--overfit_batches",
        help="Used as the flag with the same name in pl.Trainer",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="If passed, uses the PyTorch Profiler to track computation, exported as a Chrome-style trace.",
    )
    return parser


def main():
    parser = _setup_parser()
    args = parser.parse_args()
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(orderTensor),
        ]
    )

    data = load_dataset("competitions/aiornot")
    dataset = AIOrNotDataset(data=data["train"], transform=transform)
    model = ResnetModel()

    tdl = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    logger = pl.loggers.WandbLogger(
        log_model="all", save_dir=str(args.log_dir), job_type="train"
    )
    logger.watch(model, log_freq=args.log_freq)
    experiment_dir = logger.experiment.dir
    trainer = pl.Trainer(
        max_epochs=args.max_epochs, logger=logger, overfit_batches=args.overfit_batches
    )
    if args.profile:
        sched = torch.profiler.schedule(wait=0, warmup=3, active=4, repeat=0)
        profiler = PyTorchProfiler(
            export_to_chrome=True, schedule=sched, dirpath=experiment_dir
        )
        profiler.STEP_FUNCTIONS = {"training_step"}  # only profile training
    else:
        profiler = PassThroughProfiler()

    trainer.profiler = profiler
    trainer.fit(model=model, train_dataloaders=tdl)


if __name__ == "__main__":
    main()
