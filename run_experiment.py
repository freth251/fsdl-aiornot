from data.dataset import ParquetDataset
from models.resnet import ResnetModel
from torchvision import transforms
import pytorch_lightning as pl
import torch
from datasets import load_dataset

def orderTensor(x):
    x = x.to(torch.float32)
    return x



def main():
    log_dir='./logs'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(orderTensor)
    ])

    data = load_dataset("competitions/aiornot")
    dataset = ParquetDataset(data=data['train'], transform=transform)
    model= ResnetModel()
    trainer = pl.Trainer(max_epochs=1)
    tdl = torch.utils.data.DataLoader(dataset, batch_size=1)

    logger = pl.loggers.WandbLogger(log_model="all", save_dir=str(log_dir), job_type="train")
    logger.watch(model, log_freq=100)
    trainer.fit(model=model, train_dataloaders=tdl)


if __name__ == "__main__":
    main()