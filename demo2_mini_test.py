import cv2
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data_utils
from demo2_CNN import CNN

# 检查是否有可用的CUDA设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载测试数据集
test_data = datasets.MNIST(
    root='mnist',
    train=False,
    transform=transforms.ToTensor(),
    download=True
)

# 创建数据加载器
test_loader = data_utils.DataLoader(
    dataset=test_data, batch_size=64, shuffle=True)

# 加载预训练模型
cnn = torch.load("model/mnist_model.pkl")
cnn.to(device)  # 将模型移至GPU或CPU
cnn.eval()  # 设置为评估模式，这行很重要！

# 定义损失函数
loss_function = torch.nn.CrossEntropyLoss()

# 初始化统计变量
total_loss = 0
total_correct = 0
total_samples = 0

# 关闭梯度计算，提高推理速度
# 关闭梯度计算，提高推理速度
with torch.no_grad():
    for index, (images, labels) in enumerate(test_loader):
        # 将数据移至GPU或CPU
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        output = cnn(images)
        _, pred = output.max(1)

        # 计算损失
        loss = loss_function(output, labels)
        total_loss += loss.item()

        # 计算正确预测数
        correct = (pred == labels).sum().item()
        total_correct += correct
        total_samples += labels.size(0)  # 此处的 labels 是 PyTorch 张量

        # 将数据移回CPU并转换为NumPy数组用于显示
        images = images.cpu().numpy()
        labels = labels.cpu().numpy()  # 此处 labels 变成了 NumPy 数组
        pred = pred.cpu().numpy()

        # 可视化每个批次中的图像
        for idx in range(images.shape[0]):
            # 转换图像格式 (channel, height, width) -> (height, width, channel)
            im_data = images[idx].transpose(1, 2, 0)
            # 因为MNIST是灰度图，需要适当处理才能正确显示
            im_data = (im_data * 255).astype('uint8').squeeze()  # 转换为0-255范围
            im_label = labels[idx]
            im_pred = pred[idx]

            print(f"预测值：{im_pred}，真实值：{im_label}")
            cv2.imshow("now_img", im_data)
            key = cv2.waitKey(0)
            if key == 27:  # ESC键退出
                cv2.destroyAllWindows()
                break

        # 打印当前批次的准确率
        batch_size = labels.shape[0]  # 修正为 NumPy 数组的 shape
        print(f"当前批次准确率: {correct / batch_size:.4f}")

# 打印整体测试结果
print(f"测试集总损失: {total_loss/len(test_loader):.4f}")
print(f"测试集总准确率: {total_correct/total_samples:.4f}")

    
