import torch.nn as nn
VGG_11 = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]
VGG_13 = [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]
VGG_16 = [64, 64, "M", 128, 128, "M", 256, 512, 256, "M1", 512, 1024, 512, "M", 512, 1024, 512, "M"]
VGG_19 = [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"]

class VGG_net(nn.Module):
    def __init__(self, architecture, in_channels=3, num_classes=100):
        super(VGG_net, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(architecture=architecture)

        self.fcs = nn.Sequential(
            nn.Linear(512*5*5, 4096),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        times=0
        in_channels = self.in_channels

        for x in architecture:
            times+=1
            if type(x) == int:
                out_channels = x
                if times==8 or times==12 or times==16:
                    layers += [
                        nn.Conv2d(
                            in_channels=in_channels, 
                            out_channels=out_channels,
                            kernel_size=(3,3),
                            stride=(1,1),
                            padding=(1,1),
                        ),
                        nn.BatchNorm2d(x),
                        nn.ReLU()
                    ]
                else:
                    layers += [
                        nn.Conv2d(
                            in_channels=in_channels, 
                            out_channels=out_channels,
                            kernel_size=(3,3),
                            stride=(1,1),
                            padding=(1,1)
                            ),
                        nn.BatchNorm2d(x),
                        nn.ReLU()
                    ]
                in_channels = x
            elif x == "M1":
                layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(1,1))]
            elif x=='M':
                layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]
        
        return nn.Sequential(*layers)
# class VGG11(VGG_net):
#     def __init__(self):
#         super().__init__(VGG_11)

# class VGG13(VGG_net):
#     def __init__(self):
#         super().__init__(VGG_13)
class vgg16_miniImage_gr_compare(VGG_net):
    def __init__(self):
        super().__init__(VGG_16)
# class VGG19(VGG_net):
#     def __init__(self):
#         super().__init__(VGG_19)
# a=VGG16Next()
# t=torch.randn(1,3,224,224)
# c,b=thop.profile(a,inputs=(t,))
# print(c,b)