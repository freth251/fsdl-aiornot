from data.dataset import ParquetDataset
from models.resnet import ResnetModel
from torchvision import transforms
import pytorch_lightning as pl
import torch
def orderTensor(x):
    x = x.to(torch.float32)
    return x



def main():
    # Usage example
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(orderTensor)
    ])

    dataset = ParquetDataset(directory='./data/train', transform=transform)
    model= ResnetModel()
    trainer = pl.Trainer(max_epochs=1)
    tdl = torch.utils.data.DataLoader(dataset, batch_size=1)
    trainer.fit(model=model, train_dataloaders=tdl)


if __name__ == "__main__":
    main()