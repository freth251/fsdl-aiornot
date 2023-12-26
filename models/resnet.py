import pytorch_lightning as pl
import torch
from typing import Tuple
from torchmetrics import Accuracy
from torch import nn
from torch.nn import functional as F


class ResnetModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # Block 1
        self.conv1_b1 = nn.Conv2d(3, 64, (7,7), stride= (2,2), padding=(3,3), bias=False)
        self.batchnorm1_b1 = nn.BatchNorm2d(64)

        # Block 2
        self.maxpool1_b2= torch.nn.MaxPool2d(kernel_size=(3,3), padding=(1,1), stride=(2,2))
        self.conv1_b2=nn.Conv2d(64, 64, (1,1), bias=False)
        self.batchnorm1_b2=torch.nn.BatchNorm2d(64)
        self.conv2_b2=nn.Conv2d(64, 64, (3,3), padding=(1,1), bias=False)
        self.batchnorm2_b2=torch.nn.BatchNorm2d(64)
        self.conv3_b2=nn.Conv2d(64, 256, (1,1), bias=False)
        self.convres_b2=nn.Conv2d(64, 256, (1,1), bias=False)
        self.batchnorm3_b2=torch.nn.BatchNorm2d(256)
        self.batchnormres_b2=torch.nn.BatchNorm2d(256)

        # Block 3
        self.conv1_b3=nn.Conv2d(256, 64, (1,1), bias=False)
        self.batchnorm1_b3=torch.nn.BatchNorm2d(64)
        self.conv2_b3=nn.Conv2d(64, 64, (3,3), padding=(1,1), bias=False)
        self.batchnorm2_b3=torch.nn.BatchNorm2d(64)
        self.conv3_b3=nn.Conv2d(64, 256, (1,1), bias=False)
        self.batchnorm3_b3=torch.nn.BatchNorm2d(256)

        # Block 4
        self.conv1_b4=nn.Conv2d(256, 64, (1,1), bias=False)
        self.batchnorm1_b4=torch.nn.BatchNorm2d(64)
        self.conv2_b4=nn.Conv2d(64, 64, (3,3), padding=(1,1), bias=False)
        self.batchnorm2_b4=torch.nn.BatchNorm2d(64)
        self.conv3_b4=nn.Conv2d(64, 256, (1,1), bias=False)
        self.batchnorm3_b4=torch.nn.BatchNorm2d(256)

        # Block 5
        self.conv1_b5=nn.Conv2d(256, 128, (1,1), bias=False)
        self.batchnorm1_b5=torch.nn.BatchNorm2d(128)
        self.conv2_b5=nn.Conv2d(128, 128, (3,3),padding=(1,1), bias=False)
        self.batchnorm2_b5=torch.nn.BatchNorm2d(128)
        self.conv3_b5=nn.Conv2d(128, 512, (1,1),stride=(2,2), bias=False)
        self.batchnorm3_b5=torch.nn.BatchNorm2d(512)
        self.convres_b5=nn.Conv2d(256, 512, (1,1),stride=(2,2), bias=False)
        self.batchnormres_b5=torch.nn.BatchNorm2d(512)

        # Block 6
        self.conv1_b6=nn.Conv2d(512, 128, (1,1), bias=False)
        self.batchnorm1_b6=torch.nn.BatchNorm2d(128)
        self.conv2_b6=nn.Conv2d(128, 128, (3,3),padding=(1,1), bias=False)
        self.batchnorm2_b6=torch.nn.BatchNorm2d(128)
        self.conv3_b6=nn.Conv2d(128, 512, (1,1), bias=False)
        self.batchnorm3_b6=torch.nn.BatchNorm2d(512)


        # Block 7
        self.conv1_b7=nn.Conv2d(512, 128, (1,1), bias=False)
        self.batchnorm1_b7=torch.nn.BatchNorm2d(128)
        self.conv2_b7=nn.Conv2d(128, 128, (3,3),padding=(1,1), bias=False)
        self.batchnorm2_b7=torch.nn.BatchNorm2d(128)
        self.conv3_b7=nn.Conv2d(128, 512, (1,1), bias=False)
        self.batchnorm3_b7=torch.nn.BatchNorm2d(512)

        # Block 8
        self.conv1_b8=nn.Conv2d(512, 128, (1,1), bias=False)
        self.batchnorm1_b8=torch.nn.BatchNorm2d(128)
        self.conv2_b8=nn.Conv2d(128, 128, (3,3),padding=(1,1), bias=False)
        self.batchnorm2_b8=torch.nn.BatchNorm2d(128)
        self.conv3_b8=nn.Conv2d(128, 512, (1,1), bias=False)
        self.batchnorm3_b8=torch.nn.BatchNorm2d(512)

        # Block 9
        self.conv1_b9=nn.Conv2d(512, 256, (1,1), bias=False)
        self.batchnorm1_b9=torch.nn.BatchNorm2d(256)
        self.conv2_b9=nn.Conv2d(256, 256, (3,3),padding=(1,1), bias=False)
        self.batchnorm2_b9=torch.nn.BatchNorm2d(256)
        self.conv3_b9=nn.Conv2d(256, 1024, (1,1), stride=(2,2), bias=False)
        self.convres_b9=nn.Conv2d(512, 1024, (1,1), stride=(2,2), bias=False)
        self.batchnorm3_b9=torch.nn.BatchNorm2d(1024)
        self.batchnormres_b9=torch.nn.BatchNorm2d(1024)

        # Block 10
        self.conv1_b10=nn.Conv2d(1024, 256, (1,1), bias=False)
        self.batchnorm1_b10=torch.nn.BatchNorm2d(256)
        self.conv2_b10=nn.Conv2d(256, 256, (3,3),padding=(1,1), bias=False)
        self.batchnorm2_b10=torch.nn.BatchNorm2d(256)
        self.conv3_b10=nn.Conv2d(256, 1024, (1,1), bias=False)
        self.batchnorm3_b10=torch.nn.BatchNorm2d(1024)


        # Block 11
        self.conv1_b11=nn.Conv2d(1024, 256, (1,1), bias=False)
        self.batchnorm1_b11=torch.nn.BatchNorm2d(256)
        self.conv2_b11=nn.Conv2d(256, 256, (3,3),padding=(1,1), bias=False)
        self.batchnorm2_b11=torch.nn.BatchNorm2d(256)
        self.conv3_b11=nn.Conv2d(256, 1024, (1,1), bias=False)
        self.batchnorm3_b11=torch.nn.BatchNorm2d(1024)

        # Block 12
        self.conv1_b12=nn.Conv2d(1024, 256, (1,1), bias=False)
        self.batchnorm1_b12=torch.nn.BatchNorm2d(256)
        self.conv2_b12=nn.Conv2d(256, 256, (3,3),padding=(1,1), bias=False)
        self.batchnorm2_b12=torch.nn.BatchNorm2d(256)
        self.conv3_b12=nn.Conv2d(256, 1024, (1,1), bias=False)
        self.batchnorm3_b12=torch.nn.BatchNorm2d(1024)

        # Block 13
        self.conv1_b13=nn.Conv2d(1024, 256, (1,1), bias=False)
        self.batchnorm1_b13=torch.nn.BatchNorm2d(256)
        self.conv2_b13=nn.Conv2d(256, 256, (3,3),padding=(1,1), bias=False)
        self.batchnorm2_b13=torch.nn.BatchNorm2d(256)
        self.conv3_b13=nn.Conv2d(256, 1024, (1,1), bias=False)
        self.batchnorm3_b13=torch.nn.BatchNorm2d(1024)

        # Block 14
        self.conv1_b14=nn.Conv2d(1024, 256, (1,1), bias=False)
        self.batchnorm1_b14=torch.nn.BatchNorm2d(256)
        self.conv2_b14=nn.Conv2d(256, 256, (3,3),padding=(1,1), bias=False)
        self.batchnorm2_b14=torch.nn.BatchNorm2d(256)
        self.conv3_b14=nn.Conv2d(256, 1024, (1,1), bias=False)
        self.batchnorm3_b14=torch.nn.BatchNorm2d(1024)

        # Block 15
        self.conv1_b15=nn.Conv2d(1024, 256, (1,1), bias=False)
        self.batchnorm1_b15=torch.nn.BatchNorm2d(256)
        self.conv2_b15=nn.Conv2d(256, 256, (3,3),padding=(1,1), bias=False)
        self.batchnorm2_b15=torch.nn.BatchNorm2d(256)
        self.conv3_b15=nn.Conv2d(256, 1024, (1,1), bias=False)
        self.batchnorm3_b15=torch.nn.BatchNorm2d(1024)

        # Block 16
        self.conv1_b16=nn.Conv2d(1024, 512, (1,1), bias=False)
        self.batchnorm1_b16=torch.nn.BatchNorm2d(512)
        self.conv2_b16=nn.Conv2d(512, 512, (3,3),padding=(1,1), bias=False)
        self.batchnorm2_b16=torch.nn.BatchNorm2d(512)
        self.conv3_b16=nn.Conv2d(512, 2048, (1,1), stride=(2,2),bias=False)
        self.batchnorm3_b16=torch.nn.BatchNorm2d(2048)
        self.convres_b16=nn.Conv2d(1024, 2048, (1,1), stride=(2,2), bias=False)
        self.batchnormres_b16=torch.nn.BatchNorm2d(2048)

        # Block 17
        self.conv1_b17=nn.Conv2d(2048, 512, (1,1), bias=False)
        self.batchnorm1_b17=torch.nn.BatchNorm2d(512)
        self.conv2_b17=nn.Conv2d(512, 512, (3,3),padding=(1,1), bias=False)
        self.batchnorm2_b17=torch.nn.BatchNorm2d(512)
        self.conv3_b17=nn.Conv2d(512, 2048, (1,1), bias=False)
        self.batchnorm3_b17=torch.nn.BatchNorm2d(2048)

        # Block 18
        self.conv1_b18=nn.Conv2d(2048, 512, (1,1), bias=False)
        self.batchnorm1_b18=torch.nn.BatchNorm2d(512)
        self.conv2_b18=nn.Conv2d(512, 512, (3,3),padding=(1,1), bias=False)
        self.batchnorm2_b18=torch.nn.BatchNorm2d(512)
        self.conv3_b18=nn.Conv2d(512, 2048, (1,1), bias=False)
        self.batchnorm3_b18=torch.nn.BatchNorm2d(2048)

        # Block 19
        self.avgpool_b19= torch.nn.AvgPool2d((7,7), stride=(1,1))
        self.linear_layer_b19 = nn.Linear(2048, 1)
        self.sigmoid_b19= nn.Sigmoid()

        # metrics
        self.train_acc = Accuracy(task='binary')

        self.loss= torch.nn.BCELoss()

        

    def block1(self, x):
        x=self.conv1_b1(x)
        x=self.batchnorm1_b1(x)
        return x
    
    def block2(self, x):
        x=F.relu(x)
        x=self.maxpool1_b2(x)

        residual=x #split residual

        x=self.conv1_b2(x)
        x=self.batchnorm1_b2(x)
        x= F.relu(x)
        x=self.conv2_b2(x)
        x=self.batchnorm2_b2(x)
        x= F.relu(x)
        x=self.conv3_b2(x)
        x=self.batchnorm3_b2(x)

        residual=self.convres_b2(residual)
        residual= self.batchnormres_b2(residual)

        x=x+residual #add residual connection
        return x
    
    def block3(self, x):
        x=F.relu(x)
        residual=x #split connection

        x=self.conv1_b3(x)
        x=self.batchnorm1_b3(x)
        x=F.relu(x)
        x=self.conv2_b3(x)
        x=self.batchnorm2_b3(x)
        x=F.relu(x)
        x=self.conv3_b3(x)
        x-self.batchnorm3_b3(x)

        x=x+residual
        return x
    
    def block4(self, x):
        x=F.relu(x)
        residual=x #split connection

        x=self.conv1_b4(x)
        x=self.batchnorm1_b4(x)
        x=F.relu(x)
        x=self.conv2_b4(x)
        x=self.batchnorm2_b4(x)
        x=F.relu(x)
        x=self.conv3_b4(x)
        x=self.batchnorm3_b4(x)

        x=x+residual
        return x
    
    def block5(self, x):
        x= F.relu(x)
        residual=x

        x=self.conv1_b5(x)
        x=self.batchnorm1_b5(x)
        x= F.relu(x)
        x=self.conv2_b5(x)
        x=self.batchnorm2_b5(x)
        x= F.relu(x)
        x=self.conv3_b5(x)
        x=self.batchnorm3_b5(x)

        residual=self.convres_b5(residual)
        residual=self.batchnormres_b5(residual)

        x=x+residual
        return x

    def block6(self, x):
        x= F.relu(x)
        residual=x

        x=self.conv1_b6(x)
        x=self.batchnorm1_b6(x)
        x= F.relu(x)
        x=self.conv2_b6(x)
        x=self.batchnorm2_b6(x)
        x= F.relu(x)
        x=self.conv3_b6(x)
        x=self.batchnorm3_b6(x)


        x=x+residual
        return x
    
    def block7(self, x):
        x= F.relu(x)
        residual=x

        x=self.conv1_b7(x)
        x=self.batchnorm1_b7(x)
        x= F.relu(x)
        x=self.conv2_b7(x)
        x=self.batchnorm2_b7(x)
        x= F.relu(x)
        x=self.conv3_b7(x)
        x=self.batchnorm3_b7(x)


        x=x+residual
        return x
    
    def block8(self, x):
        x= F.relu(x)
        residual=x

        x=self.conv1_b8(x)
        x=self.batchnorm1_b8(x)
        x= F.relu(x)
        x=self.conv2_b8(x)
        x=self.batchnorm2_b8(x)
        x= F.relu(x)
        x=self.conv3_b8(x)
        x=self.batchnorm3_b8(x)

        x=x+residual
        return x
    
    def block9(self, x):
        x= F.relu(x)
        residual=x

        x=self.conv1_b9(x)
        x=self.batchnorm1_b9(x)
        x= F.relu(x)
        x=self.conv2_b9(x)
        x=self.batchnorm2_b9(x)
        x= F.relu(x)
        x=self.conv3_b9(x)
        x=self.batchnorm3_b9(x)

        residual= self.convres_b9(residual)
        residual= self.batchnormres_b9(residual)

        x=x+residual
        return x

    def block10(self, x):
        x= F.relu(x)
        residual=x

        x=self.conv1_b10(x)
        x=self.batchnorm1_b10(x)
        x= F.relu(x)
        x=self.conv2_b10(x)
        x=self.batchnorm2_b10(x)
        x= F.relu(x)
        x=self.conv3_b10(x)
        x=self.batchnorm3_b10(x)


        x=x+residual
        return x

    def block11(self, x):
        x= F.relu(x)
        residual=x

        x=self.conv1_b11(x)
        x=self.batchnorm1_b11(x)
        x= F.relu(x)
        x=self.conv2_b11(x)
        x=self.batchnorm2_b11(x)
        x= F.relu(x)
        x=self.conv3_b11(x)
        x=self.batchnorm3_b11(x)

        x=x+residual
        return x
    
    def block12(self, x):
        x= F.relu(x)
        residual=x

        x=self.conv1_b12(x)
        x=self.batchnorm1_b12(x)
        x= F.relu(x)
        x=self.conv2_b12(x)
        x=self.batchnorm2_b12(x)
        x= F.relu(x)
        x=self.conv3_b12(x)
        x=self.batchnorm3_b12(x)

        x=x+residual
        return x
    
    def block13(self, x):
        x= F.relu(x)
        residual=x

        x=self.conv1_b13(x)
        x=self.batchnorm1_b13(x)
        x= F.relu(x)
        x=self.conv2_b13(x)
        x=self.batchnorm2_b13(x)
        x= F.relu(x)
        x=self.conv3_b13(x)
        x=self.batchnorm3_b13(x)

        x=x+residual
        return x
    
    def block14(self, x):
        x= F.relu(x)
        residual=x

        x=self.conv1_b14(x)
        x=self.batchnorm1_b14(x)
        x= F.relu(x)
        x=self.conv2_b14(x)
        x=self.batchnorm2_b14(x)
        x= F.relu(x)
        x=self.conv3_b14(x)
        x=self.batchnorm3_b14(x)

        x=x+residual
        return x
    
    def block15(self, x):
        x= F.relu(x)
        residual=x

        x=self.conv1_b15(x)
        x=self.batchnorm1_b15(x)
        x= F.relu(x)
        x=self.conv2_b15(x)
        x=self.batchnorm2_b15(x)
        x= F.relu(x)
        x=self.conv3_b15(x)
        x=self.batchnorm3_b15(x)

        x=x+residual
        return x

    def block16(self, x):
        x= F.relu(x)
        residual=x

        x=self.conv1_b16(x)
        x=self.batchnorm1_b16(x)
        x= F.relu(x)
        x=self.conv2_b16(x)
        x=self.batchnorm2_b16(x)
        x= F.relu(x)
        x=self.conv3_b16(x)
        x=self.batchnorm3_b16(x)

        residual=self.convres_b16(residual)
        residual=self.batchnormres_b16(residual)

        x=x+residual
        return x

    def block17(self, x):
        x= F.relu(x)
        residual=x

        x=self.conv1_b17(x)
        x=self.batchnorm1_b17(x)
        x= F.relu(x)
        x=self.conv2_b17(x)
        x=self.batchnorm2_b17(x)
        x= F.relu(x)
        x=self.conv3_b17(x)
        x=self.batchnorm3_b17(x)

        x=x+residual
        return x
    
    def block18(self, x):
        x= F.relu(x)
        residual=x

        x=self.conv1_b18(x)
        x=self.batchnorm1_b18(x)
        x= F.relu(x)
        x=self.conv2_b18(x)
        x=self.batchnorm2_b18(x)
        x= F.relu(x)
        x=self.conv3_b18(x)
        x=self.batchnorm3_b18(x)

        x=x+residual
        return x
    
    def block19(self, x):
        x=F.relu(x)
        x= self.avgpool_b19(x)
        x=x.view(1, 2048)
        x=self.linear_layer_b19(x)
        x=self.sigmoid_b19(x)
        return x

    def forward(self, x):
        x=self.block1(x)
        x=self.block2(x)
        x=self.block3(x)
        x=self.block4(x)
        x=self.block5(x)
        x=self.block6(x)
        x=self.block7(x)
        x=self.block8(x)
        x=self.block9(x)
        x=self.block10(x)
        x=self.block11(x)
        x=self.block12(x)
        x=self.block13(x)
        x=self.block14(x)
        x=self.block15(x)
        x=self.block16(x)
        x=self.block17(x)
        x=self.block18(x)
        x=self.block19(x)
        return x
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        xs, ys = batch  # unpack the batch
        outs = self.forward(xs)  # apply the model
        loss = self.loss(outs[0], ys)
        
        loss= torch.squeeze(loss)
        outputs = {"loss": loss}
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True)
        return outputs
    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)  # https://fsdl.me/ol-reliable-img
        return optimizer 

