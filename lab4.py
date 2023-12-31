"""
任务要求：针对已训练好的卷积神经网络，给定一张输入图片，生成该图片对于特定类别的可解释性分析结果。

实验将提供基于PyTorch的二分类模型，该模型可用于猫和狗的分类。注：PyTorch使用的网络架构是AlexNet。

实验将同时提供三张输入图片，对于每张图片，分别针对猫和狗的类别，进行Grad-CAM和LayerCAM的可解释性分析。

注意事项：
实验报告需包含每张输入图片在最后一层卷积层输出的的可视化结果（对输出特征图的每一个通道进行可视化），每张图片分别针对猫和狗两个类别的可解释性分析结果（Grad-CAM及LayerCAM），以及对应的实验分析。
AlexNet(
    [3 x 224 x 224]
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    [64 x 55 x 55]
    (1): ReLU(inplace=True)
    [64 x 55 x 55]
    (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    [64 x 27 x 27]
    (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    [192 x 27 x 27]
    (4): ReLU(inplace=True)
    [192 x 27 x 27]
    (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    [192 x 13 x 13]
    (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    [384 x 13 x 13]
    (7): ReLU(inplace=True)
    [384 x 13 x 13]
    (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    [256 x 13 x 13]
    (9): ReLU(inplace=True)
    [256 x 13 x 13]
    (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    [256 x 13 x 13]
    (11): ReLU(inplace=True)
    [256 x 13 x 13]
    (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    [256 x 6 x 6]
  )
  [256 x 6 x 6]
  (avgpool): AdaptiveAvgPool2d(output_size=(6, 6))
  [256 x 6 x 6]
  (classifier): Sequential(
    [9216 x 4096]
    (0): Dropout(p=0.5, inplace=False)
    [4096 x 4096]
    (1): Linear(in_features=9216, out_features=4096, bias=True)
    [4096 x 4096]
    (2): ReLU(inplace=True)
    [4096 x 4096]
    (3): Dropout(p=0.5, inplace=False)
    [4096 x 4096]
    (4): Linear(in_features=4096, out_features=4096, bias=True)
    [4096 x 4096]
    (5): ReLU(inplace=True)
    [4096 x 4096]
    (6): Linear(in_features=4096, out_features=2, bias=True)
    [2 x 4096]
  )
)
"""
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import cv2

device = torch.device('cuda')


def read_image(path):
    img = torchvision.datasets.folder.default_loader(path)
    img = torchvision.transforms.ToTensor()(img)
    img = img.unsqueeze(0).to(device)
    return img


CAT = 0
DOG = 1
# 输入图片 ./data4/cat.jpg ./data4/dog.jpg ./data4/both.jpg
img1 = read_image("./data4/cat.jpg")
img2 = read_image("./data4/dog.jpg")
img3 = read_image("./data4/both.jpg")
# 读取模型
model = torch.load("./data4/torch_alex.pth").to(device)


# print(model)


class CAM:
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.input_img = None
        self.feature = []
        self.gradient = []
        self.model.eval()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, input, output):
            self.feature.append(output.cpu().data.numpy()[0])

        def backward_hook_function(module, grad_in, grad_out):
            self.gradient.insert(0, (grad_out[0].cpu().data.numpy()[0]))

        # Register hook to the target layer
        for target_layer in self.target_layers:
            target_layer.register_forward_hook(hook_function)
            target_layer.register_backward_hook(backward_hook_function)

    def eval_model(self, input_image, target_class=None):
        self.input_img = input_image
        # Forward pass
        model_output = self.model(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.cpu().data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_().to(device)
        one_hot_output[0][target_class] = 1
        one_hot_output = torch.sum(one_hot_output * model_output)
        # Zero grads
        self.model.zero_grad()
        # Backward pass
        one_hot_output.backward(retain_graph=True)

    def cams_on_image(self, cams):
        ret = [self.input_img.cpu().squeeze().permute(1, 2, 0)]
        for cam in cams:
            cam = np.maximum(cam, 0)
            cam = cv2.resize(cam, (self.input_img.shape[3], self.input_img.shape[2]))
            cam = np.uint8(cv2.normalize(cam, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))
            heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            heatmap = np.float32(heatmap) / 255
            cam = heatmap + np.float32(self.input_img.cpu().squeeze().permute(1, 2, 0))
            cam = cam / np.max(cam)
            ret.append(cam)
        return ret


class GradCAM(CAM):
    """
    Grad-CAM：Grad-CAM使用目标类别相对于特征图的梯度来作为权重。
    具体来说，它首先计算目标类别的得分相对于特征图的梯度，然后对梯度在空间维度（宽度和高度）上进行全局平均池化，得到每个通道的权重。
    最后，使用这些权重对特征图进行加权求和，得到类激活映射。
    """
    def generate_cam(self, input_image, target_class=None):
        self.feature = []
        self.gradient = []
        self.eval_model(input_image, target_class)
        # 原图
        cams = []
        for i in range(len(self.gradient)):
            # 层平均梯度(权重) -> 各个通道乘以权重 -> 各个通道求和 -> relu -> cam
            # 层平均梯度(权重)
            weights = np.mean(self.gradient[i], axis=(1, 2)).reshape(-1, 1, 1)
            # 各个通道乘以权重 & 各个通道求和
            cam = np.sum(weights * self.feature[i], axis=0)
            cams.append(cam)
            # relu & resize & 归一化 (0, 255)
        return self.cams_on_image(cams)


class LayerCAM(CAM):
    """
    Layer-CAM：Layer-CAM也使用目标类别的得分相对于特征图的梯度，但是它在计算权重时使用了不同的方法。
    具体来说，Layer-CAM首先计算目标类别的得分相对于特征图的梯度，
    然后将梯度与特征图进行逐元素相乘，得到一个新的特征图。
    最后，对新的特征图在空间维度上进行全局平均池化，得到类激活映射。
    """
    def generate_cam(self, input_image, target_class=None):
        self.feature = []
        self.gradient = []
        self.eval_model(input_image, target_class)
        cams = []
        for i in range(len(self.gradient)):
            # relu -> 逐元素相乘 -> 各个通道求和 -> relu -> cam
            # relu
            self.gradient[i] = np.maximum(self.gradient[i], 0)
            # 逐元素相乘 & 平均
            cam = np.sum(self.gradient[i] * self.feature[i], axis=0)
            cams.append(cam)
            # relu & resize & 归一化 (0, 255)
        return self.cams_on_image(cams)


# 生成Grad-CAM

grad_cam = GradCAM(model, model.features[::2])
grad_cam1 = grad_cam.generate_cam(img1, CAT)
grad_cam2 = grad_cam.generate_cam(img2, DOG)
grad_cam3 = grad_cam.generate_cam(img3, CAT)
grad_cam4 = grad_cam.generate_cam(img3, DOG)

# 生成Layer-CAM
layer_cam = LayerCAM(model, model.features[::2])
layer_cam1 = layer_cam.generate_cam(img1, CAT)
layer_cam2 = layer_cam.generate_cam(img2, DOG)
layer_cam3 = layer_cam.generate_cam(img3, CAT)
layer_cam4 = layer_cam.generate_cam(img3, DOG)


def draw_cams(cams, save_path):
    fig = plt.figure(figsize=(10 * len(cams), 10))
    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    for i in range(len(cams)):
        ax = fig.add_subplot(1, len(cams), i + 1)
        ax.imshow(cams[i])
        ax.axis('off')
    plt.savefig(save_path)


draw_cams(grad_cam1, "./figs4/cat-gradcam.jpg")
draw_cams(grad_cam2, "./figs4/dog-gradcam.jpg")
draw_cams(grad_cam3, "./figs4/cat-both-gradcam.jpg")
draw_cams(grad_cam4, "./figs4/dog-both-gradcam.jpg")

draw_cams(layer_cam1, "./figs4/cat-layercam.jpg")
draw_cams(layer_cam2, "./figs4/dog-layercam.jpg")
draw_cams(layer_cam3, "./figs4/cat-both-layercam.jpg")
draw_cams(layer_cam4, "./figs4/dog-both-layercam.jpg")
