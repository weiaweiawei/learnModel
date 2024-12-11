# 导入必要的库
import torch  # PyTorch深度学习框架
import torchvision.datasets as datasets  # PyTorch视觉数据集工具
import torchvision.transforms as transforms  # PyTorch图像预处理工具
import torch.utils.data as data_utils
from demo2_CNN import CNN 

### 数据加载####

# 加载MNIST训练数据集
train_data = datasets.MNIST(
    root='mnist',  # 数据集将被下载到的根目录
    train=True,    # 指定加载训练集
    transform=transforms.ToTensor(),  # 将图像数据转换为PyTorch张量，并将像素值归一化到[0,1]区间
    download=True  # 如果数据集不存在，则自动下载
)

# 加载MNIST测试数据集
test_data = datasets.MNIST(
    root='mnist',  # 使用相同的根目录
    train=False,   # 指定加载测试集
    transform=transforms.ToTensor(),  # 使用相同的数据转换
    download=True  # 如果数据集不存在，则自动下载
)
# 打印数据集信息
# print(train_data)  # 打印训练集的基本信息（数据集大小、类别等）
# print(test_data)   # 打印测试集的基本信息
 
#### 分批加载 ####

train_loader = data_utils.DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_loader = data_utils.DataLoader(dataset=test_data, batch_size=64, shuffle=True)

# print(train_loader) 
# print(test_loader)
cnn = CNN()
# 如果安装了显卡加速，可以放到cuda上运行，不然就在cpu上运行
# cnn = cnn.cuda()


#### 损失函数 ####

loss_function = torch.nn.CrossEntropyLoss()

#### 优化器 ####

optimizer = torch.optim.Adam(cnn.parameters(),lr=0.001) # 学习率0.01不是非常小了

### 训练模型 ####
# epoch 通常指 一次训练数据全部训练一遍
# 大的循环 我要循环10次
for epoch in range(10):
    
    for index,(images,labels) in enumerate(train_loader):
        # print(index)
        # print(images)
        # print(labels)
        # images = images.cuda()
        # labels = labels.cuda()
        ### 前向传播 ###
        output = cnn(images)
        ### 计算损失  传入输出层节点和真实标签来计算损失函数
        loss = loss_function(output,labels)
        ### 反向传播
        # 先清空梯度
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        print("当前为第{}轮，第{}/{}次训练，损失为{}".format(epoch,index,len(train_loader),loss.item()))
        # break

    ### 每次一轮后，我需要在测试集上测试一下模型的效果
    # 测试集上测试
    # 计算总损失
    total_test_loss = 0
    # 计算准确率 
    total_test_accuracy = 0
    # 分批取出在测试集上的数据
    for index,(images,labels) in enumerate(test_loader):
        # images = images.cuda()
        # labels = labels.cuda()
        output = cnn(images)
        loss = loss_function(output,labels)
        total_test_loss += loss.item()
        # print(output)
        # print(output.size())
        # print(labels)
        # print(labels.size())
        _,pred = output.max(1) # 它也是一个张量
        # print(pred) 
        # eq() 把两个张量中的每一个元素进行比较，如果相等返回True，不相等返回False
        rightValue = (pred == labels).sum().item()
        total_test_accuracy += rightValue
       
        # print(pred == labels)
        # print((pred == labels).sum().item())
        print("当前为第{}轮，测试集上的损失为{}，测试集上的准确率为{}".format(epoch,total_test_loss,total_test_accuracy/64))
        # break

    # break

# 这里的保存 权重 还是保存模型 看个人选择吧
torch.save(cnn,"model/mnist_model.pkl")

# 单通道 手写数字识别的数据集有4个维度，分别是： h*w*c*n

### value里面包括 真实标签和图片数据 


