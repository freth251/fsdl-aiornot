import argparse
import glob

from data.dataset import AIOrNotDataset
from datasets import load_dataset
from lightning.pytorch.profilers import PassThroughProfiler, PyTorchProfiler
from models.resnet import ResnetModel
import pytorch_lightning as pl
import torch
from torchvision import transforms
import wandb


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
    logger = pl.loggers.WandbLogger(
        log_model="all", save_dir=str(args.log_dir), job_type="train"
    )

    experiment_dir = logger.experiment.dir

    goldstar_metric = "validation/loss"
    filename_format = "epoch={epoch:04d}-validation.loss={validation/loss:.3f}"
    if goldstar_metric == "validation/cer":
        filename_format += "-validation.cer={validation/cer:.3f}"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=5,
        filename=filename_format,
        monitor=goldstar_metric,
        mode="min",
        auto_insert_metric_name=False,
        dirpath=experiment_dir,
        every_n_epochs=1,
    )

    summary_callback = pl.callbacks.ModelSummary(max_depth=2)

    callbacks = [summary_callback, checkpoint_callback]

    data = load_dataset("competitions/aiornot")
    train_dataset = AIOrNotDataset(data=data["train"], transform=transform)
    test_dataset = AIOrNotDataset(data=data["test"], transform=transform)
    model = ResnetModel()

    train_dl = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )
    test_dl = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    logger.watch(model, log_freq=args.log_freq)
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        overfit_batches=args.overfit_batches,
        callbacks=callbacks,
    )
    if args.profile:
        sched = torch.profiler.schedule(wait=0, warmup=3, active=4, repeat=0)
        profiler = PyTorchProfiler(
            export_to_chrome=True,
            schedule=sched,
            dirpath=f"{experiment_dir}/tbprofile",
        )
        profiler.STEP_FUNCTIONS = {"training_step"}  # only profile training
    else:
        profiler = PassThroughProfiler()

    trainer.profiler = profiler
    trainer.fit(model=model, train_dataloaders=train_dl)
    if args.profile:
        profile_art = wandb.Artifact(f"trace-{wandb.run.id}", type="profile")
        profile_art.add_file(
            glob.glob("logs/wandb/latest-run/tbprofile/*.pt.trace.json")[0],
            "trace.pt.trace.json",
        )
        wandb.run.log_artifact(profile_art)
    trainer.profiler = PassThroughProfiler()
    trainer.test(model=model, dataloaders=test_dl)

    wandb.finish()


if __name__ == "__main__":
    main()
