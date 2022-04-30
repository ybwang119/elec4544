from torch import nn
import torch
class Alexnet_miniImage_gr(nn.Module):
    def __init__(self,classes: int):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels=3,out_channels=128,kernel_size=11,stride=2,padding=2)
        self.relu=nn.ReLU(inplace=True)
        self.maxpool1=nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv2=nn.Conv2d(in_channels=128,out_channels=256,stride=1,padding=2,kernel_size=5)
        self.conv3=nn.Conv2d(in_channels=256,out_channels=512,stride=1,padding=1,kernel_size=3)
        self.conv4=nn.Conv2d(in_channels=512,out_channels=512,padding=1,stride=1,kernel_size=3,groups=128)
        self.conv5=nn.Conv2d(in_channels=512,out_channels=256,stride=1,padding=1,kernel_size=3)
        self.fc1=nn.Linear(4096,4096)
        self.fc2=nn.Linear(4096,4096)
        self.fc3=nn.Linear(4096,classes)
        self.features=   nn.Sequential(
                            self.conv1,
                            self.relu,
                            self.maxpool1,
                            self.conv2,
                            self.relu,
                            self.maxpool1,
                            self.conv3,
                            self.relu,
                            self.conv4,
                            self.relu,
                            self.conv5,
                            self.relu,
                            self.maxpool1)
        self.dropout=nn.Dropout()
        self.sfm=nn.Softmax(dim=1)
    def forward(self,x):
        temp_f=self.features(x)
        #print(temp_f.size())
        temp_f=torch.flatten(temp_f,start_dim=1,end_dim=3)
        temp_c=self.fc1(temp_f)
        temp_c=self.dropout(temp_c)
        temp_c=self.fc2(temp_c)
        temp_c=self.dropout(temp_c)
        result=self.fc3(temp_c)
        #result=self.sfm(result)
        return result