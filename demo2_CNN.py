import torch
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv = torch.nn.Sequential(
            # 1. 卷积操作 卷积层 （输入通道 输出通道 卷积核大小 填充） 
            torch.nn.Conv2d(1,32,5,padding=2),
            # 2. 归一化操作 BN层 
            torch.nn.BatchNorm2d(32),
            # 3. 激活函数 激活层
            torch.nn.ReLU(),
            # 4. 池化操作 池化层
            torch.nn.MaxPool2d(2,2)
        )
        # 全连接层 fc层
        self.fc = torch.nn.Linear(in_features=14*14*32,out_features=10)  # 输出这个是算的 14*14*32 是卷积层输出的维度

    def forward(self,x):
            output = self.conv(x)
            output = output.view(output.size(0),-1)
            output = self.fc(output)
            return output    
       
 