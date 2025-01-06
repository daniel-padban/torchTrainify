import torch
from torchTrainify.classTrainer import classificationTrainer
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms.v2 as v2
from torch.optim import AdamW
from torcheval.metrics.functional import multiclass_f1_score

class testModel(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.conv1 = nn.Conv2d(1,128,2,padding='same')
        self.conv2 = nn.Conv2d(128,256,padding='same',kernel_size=2)
        self.fco = nn.Linear(((28**2)*256),10)

    def forward(self,x):
        x = self.conv1(x)
        x = nn.functional.silu(x)
        x = self.conv2(x)
        x = nn.functional.silu(x)
        x = torch.flatten(x,start_dim=1)
        x = self.fco(x)
        return x

if __name__ == '__main__':
    torch.manual_seed(100)
    torch.mps.manual_seed(100)
    torch.cuda.manual_seed_all(100)

    transform = v2.Compose([
        v2.ToTensor(), # first, convert image to PyTorch tensor
        v2.Resize(28),
        v2.Normalize((0.1307,), (0.3081,)) # normalize inputs
    ])

    train_mnist = MNIST('mnist',train=True,transform=transform,download=True)
    test_mnist = MNIST('mnist',train=False,transform=transform,download=True)
    
    train_loader = DataLoader(train_mnist,200,True,)
    test_loader = DataLoader(test_mnist,200,True,)
    model = testModel()

    optimizer = AdamW(model.parameters(),1e-3)
    loss = nn.CrossEntropyLoss()
    f1 = multiclass_f1_score
    trainer = classificationTrainer(model,train_loader,test_loader,optimizer,loss,f1,'mps',vocal=True,dtype=torch.float32)
    trainer.train_test(2,'tests/test_model')