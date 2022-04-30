
from torch import  nn,Tensor
import torch
def shuffle(x: Tensor, groups: int) -> Tensor:
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)
    return x
class Alexnet_flower_gr_shuffle(nn.Module):
    

    def __init__(self,classes: int):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels=3,out_channels=128,kernel_size=11,stride=4,padding=2)
        self.relu=nn.ReLU(inplace=True)
        self.maxpool1=nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv2=nn.Conv2d(in_channels=128,out_channels=256,stride=1,padding=2,kernel_size=5)
        self.conv3=nn.Conv2d(in_channels=256,out_channels=512,stride=1,padding=1,kernel_size=3)
        self.conv4=nn.Conv2d(in_channels=512,out_channels=512,padding=1,stride=1,kernel_size=3,groups=128)
        self.conv5=nn.Conv2d(in_channels=512,out_channels=256,stride=1,padding=1,kernel_size=3)
        self.fc1=nn.Linear(9216,4096)
        self.fc2=nn.Linear(4096,4096)
        self.fc3=nn.Linear(4096,classes)
        self.features1=nn.Sequential(
                            self.conv1,
                            self.relu,
                            self.maxpool1,
                            self.conv2,
                            self.relu,
                            self.maxpool1,
                            self.conv3,
                            self.relu,
                            self.conv4,
                            )
        self.features2=nn.Sequential(                    
                            self.relu,
                            self.conv5,
                            self.relu,
                            self.maxpool1
                            )
        self.dropout=nn.Dropout()
        self.sfm=nn.Softmax(dim=1)
    def forward(self,x):
        temp_f=self.features1(x)
        temp_f=shuffle(temp_f,128)
        temp_f=self.features2(temp_f)
        temp_f=torch.flatten(temp_f,start_dim=1,end_dim=3)
        temp_c=self.fc1(temp_f)
        temp_c=self.dropout(temp_c)
        temp_c=self.fc2(temp_c)
        temp_c=self.dropout(temp_c)
        result=self.fc3(temp_c)
        #result=self.sfm(result)
        return result