from data.dataset import ParquetDataset
from models.resnet import ResnetModel
from torchvision import transforms
import pytorch_lightning as pl
import torch
import argparse
from datasets import load_dataset

def orderTensor(x):
    x = x.to(torch.float32)
    return x



def main():
    parser= argparse.ArgumentParser(prog='Train_Model')
    parser.add_argument('-b', '--batch_size', help="batch size", type=int, default=1)
    parser.add_argument('-d', '--log_dir', help="log directory", type=str, default='./logs')
    parser.add_argument('-f', '--log_freq', help="log frequency", type=int, default=100)
    parser.add_argument('-m', '--max_epochs', help="maximum training epochs", type=int, default=1)
    parser.add_argument('-w', '--num_workers', help="number of workers for dataloader", type=int, default=1)
    args = parser.parse_args()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(orderTensor)
    ])
    

    data = load_dataset("competitions/aiornot")
    dataset = ParquetDataset(data=data['train'], transform=transform)
    model= ResnetModel()
    
    tdl = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    logger = pl.loggers.WandbLogger(log_model="all", save_dir=str(args.log_dir), job_type="train")
    logger.watch(model, log_freq=args.log_freq)
    trainer = pl.Trainer(max_epochs=args.max_epochs, logger=logger)
    trainer.fit(model=model, train_dataloaders=tdl)


if __name__ == "__main__":
    main()