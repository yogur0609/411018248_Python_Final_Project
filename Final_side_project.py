# encoding=utf-8
# Programmer: 411018248,吳典祐
# Date: 2023/06/14
# 程式說明：MobileNet V3 神經網路訓練
# version：Python 3.10
# 安裝指引：無

import torch

device = torch.device('cuda')  # 'cuda'/'cpu'，import torch
num_outputs = 1
train_size = 430
valid_size = 20
batch_size = 32
learning_rate = 1e-04
step_size = 250  # Reriod of learning rate decay
epochs = 1000
path = r'C:\Users\w7010\Downloads\USB\project\software'
model_name = f'Tracking450ROI(MobileNet)_batch{batch_size}_lr{learning_rate:.0e}_step_size{step_size}_epoch{epochs}'
# file_name     = f'Tracking_batch{batch_size}_lr{learning_rate:.0e}_epoch{epochs}_scheduler_patience{scheduler_patience}_cd{cooldown}_target_loss{target_train_loss:.0e}_stop_patience{early_stopping_patience}'
model_path = f'{path}\Model\{model_name}.pth'  # 副檔名通常以.pt或.pth儲存，建議使用.pth
# model_div = f'{file_path.split(".")[0]}.pth'
TrainingImage = f'{path}\TrainingImage450ROI/'
Annotation = f'{path}\Annotation450ROI/'

# 建立dataset
from torch.utils.data import Dataset


class ImageLabel(Dataset):  # from torch.utils.data import Dataset
    def __init__(self, img, xy):
        self.img = img
        self.xy = xy

    def __getitem__(self, idx):
        return self.img[idx], self.xy[idx]

    def __len__(self):
        return len(self.img)


import torch.nn as nn
import torch.nn.functional as F

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class SEBlock(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1),
            h_sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class BottleNeck(nn.Module):
    def __init__(self, in_size, exp_size, out_size, s, is_se_existing, NL, k):
        super(BottleNeck, self).__init__()
        self.is_shortcut = s == 1 and in_size == out_size
        if NL == "RE":
            self.NL_layer = nn.ReLU(inplace=True)
        elif NL == "HS":
            self.NL_layer = h_swish()
        else:
            raise NotImplementedError
        self.is_se_existing = is_se_existing
        if is_se_existing:
            self.se = SEBlock(exp_size)
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(in_size, exp_size, 1, 1, 0, bias=False),
            nn.BatchNorm2d(exp_size),
            self.NL_layer,
            # dw
            nn.Conv2d(exp_size, exp_size, k, s, (k - 1) // 2, groups=exp_size, bias=False),
            nn.BatchNorm2d(exp_size),
            self.NL_layer if is_se_existing else nn.Sequential(),
            # pw-linear
            nn.Conv2d(exp_size, out_size, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_size),
        )

    def forward(self, x):
        if self.is_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV3_Large(nn.Module):
    def __init__(self, num_classes=1):
        super(MobileNetV3_Large, self).__init__()

        layers = [
            nn.Sequential(
                nn.Conv2d(3, 16, 3, 2, 1, bias=False),
                nn.BatchNorm2d(16),
                h_swish()
            ),
            BottleNeck(16, 16, 16, 1, False, "RE", 3),
            BottleNeck(16, 64, 24, 2, False, "RE", 3),
            BottleNeck(24, 72, 24, 1, False, "RE", 3),
            BottleNeck(24, 72, 40, 2, True, "RE", 5),
            BottleNeck(40, 120, 40, 1, True, "RE", 5),
            BottleNeck(40, 120, 40, 1, True, "RE", 5),
            BottleNeck(40, 240, 80, 2, False, "HS", 3),
            BottleNeck(80, 200, 80, 1, False, "HS", 3),
            BottleNeck(80, 184, 80, 1, False, "HS", 3),
            BottleNeck(80, 184, 80, 1, False, "HS", 3),
            BottleNeck(80, 480, 112, 1, True, "HS", 3),
            BottleNeck(112, 672, 112, 1, True, "HS", 3),
            BottleNeck(112, 672, 160, 2, True, "HS", 5),
            BottleNeck(160, 960, 160, 1, True, "HS", 5),
            BottleNeck(160, 960, 160, 1, True, "HS", 5),
        ]
        self.features = nn.Sequential(*layers)

        self.conv = nn.Sequential(
            nn.Conv2d(160, 960, 1, bias=False),
            nn.BatchNorm2d(960),
            h_swish()
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(960, 1280),
            h_swish(),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def main():
    from torchvision import transforms
    transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5,
                                                                            0.5))])  # ToTensor將影像像素歸一化至0~1(直接除以255)，from torchvision import transforms
    import os
    all_image_name = os.listdir(TrainingImage)  # 所有影像檔名(含.jpg)，import os
    img = list()
    xy = list()
    from PIL import Image
    import xml.etree.ElementTree as ET
    for image_name in all_image_name:
        I = Image.open(TrainingImage + image_name, mode='r')  # from PIL import Image
        I = I.crop((0, 255, 1920, 1080 - 255 + 5))  # ROI
        I = transforms(I)  # from torchvision import transforms
        img.append(I)  # 列表長度為影像個數，列表中每個元素為一個[3,300,300]的tensor
        image_name = image_name[:-4]  # 移除4個字元(.jpg)
        root = ET.parse(Annotation + image_name + '.xml').getroot()  # 獲取xml文件物件的根結點，import xml.etree.ElementTree as ET
        size = root.find('size')  # 獲取size子結點
        width = int(size.find('width').text)  # 原始影像的寬(像素)
        height = int(size.find('height').text)  # 原始影像的高(像素)
        width_scale = I.size(1) / width  # 輸入影像與原始影像的寬比
        height_scale = I.size(1) / height  # 輸入影像與原始影像的高比
        for object in root.iter('object'):  # 遞迴查詢所有的object子結點
            bndbox = object.find('bndbox')
            # chi_en=torch.Tensor([int(bndbox.find('xmin').text)*width_scale,int(bndbox.find('ymin').text)*height_scale]) # [num_outputs]，import torch
            chi_en = torch.Tensor([int(bndbox.find('xmin').text) * width_scale])
        xy.append(chi_en)  # 列表長度為影像個數，列表中每個元素為一個[num_outputs]的tensor
    dataset = ImageLabel(img, xy)
    train_data, valid_data = torch.utils.data.random_split(dataset, [train_size, valid_size])  # import torch
    train_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True,
                                               num_workers=1,  # num_workers = （執行緒數）(4,8,16)
                                               pin_memory=True)  # pin_memory = True
    # pin_memory = True 省掉了將資料從CPU傳入到快取RAM裡面，再給傳輸到GPU上；為True時是直接對映到GPU的相關記憶體塊上，省掉了一點資料傳輸時間。
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size, shuffle=True)  # imort torch

    regressor = MobileNetV3_Large().to(device)
    # regressor.load_state_dict(torch.load(model_path))
    criterion = nn.MSELoss()  # 回歸
    optimizer = torch.optim.Adam(regressor.parameters(), lr=learning_rate)
    # optimizer=torch.optim.SGD(regressor.parameters(),lr=learning_rate)
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = learning_rate
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.75, last_epoch=3000)

    train_losses_his, valid_losses_his, learning_rate_his = [], [], []
    import matplotlib.pyplot as plt
    import time
    start_time = time.time()
    for i in range(1, epochs + 1):
        print('Running Epoch:' + str(i))
        train_loss, train_total, valid_loss, valid_total = 0, 0, 0, 0
        regressor.train()
        for img, value in train_loader:  # 一個batch的image、value。image：[batch_size,3,224,224]，value：[batch_size,num_outputs]
            img, value = img.to(device), value.to(device)
            pred = regressor(img)  # pred：[batch_size,num_outputs]
            loss = criterion(pred, value)  # loss.item()：一個batch的平均loss，[1]
            train_loss += loss.item() * img.size(
                0)  # 累加計算每一epoch的loss總和。loss.item()：一個batch的平均loss，[1]。image.size(0)：一個batch的訓練資料總數
            train_total += img.size(0)  # 累加計算訓練資料總數
            optimizer.zero_grad()  # 權重梯度歸零
            loss.backward()  # 計算每個權重的loss梯度
            optimizer.step()  # 權重更新
        scheduler.step()
        learning_rate_his.append(optimizer.param_groups[0]['lr'])  # 保存當前學習率

        regressor.eval()

        # torch.save(regressor.state_dict(), model_path)

        for img, value in valid_loader:  # 一個batch的image、value。image：[batch_size,3,224,224]，value：[batch_size,1,num_outputs]
            img, value = img.to(device), value.to(device)
            pred = regressor(img)  # pred：[batch_size,num_outputs]
            loss = criterion(pred, value)  # loss.item()：一個batch的平均loss，[1]
            valid_loss += loss.item() * img.size(
                0)  # 累加計算每一epoch的loss總和。loss.item()：一個batch的平均loss，[1]。image.size(0)：一個batch的驗證資料總數
            valid_total += img.size(0)  # 累加計算驗證資料總數

        train_loss = train_loss / train_total  # 計算每一個epoch的平均訓練loss
        valid_loss = valid_loss / valid_total  # 計算每一個epoch的平均驗證loss
        train_losses_his.append(train_loss)  # 累積記錄每一個epoch的平均訓練loss，[epochs]
        valid_losses_his.append(valid_loss)  # 累積記錄每一個epoch的平均驗證loss，[epochs]
        print('Training Loss=' + str(train_loss))
        print('Validation Loss=' + str(valid_loss))
        plt.close('all')  # Close all existing figures. This line is added.
        # Plotting
        plt.ion()  # Enable interactive mode
        if i != 1:
            plt.figure(figsize=(15, 10))
            # 我們在這裡使用了semilogy函數，這個函數會在 Y 軸上使用對數尺度。這對於觀察巨大的損失變化非常有用。如果你希望在常規尺度上繪製損失，你可以將semilogy替換為plot。
            plt.semilogy(train_losses_his, 'b', linewidth=0.7, label='training loss')
            plt.semilogy(valid_losses_his, 'r', linewidth=0.7, label='validation loss')
            plt.title('Loss at Epoch: ' + str(i))
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(loc='best')
            plt.grid(True)
            if i == epochs:
                plt.savefig(f'{path}\Model_Image\{model_name}_loss.png', dpi=400)  # Save the figure
            plt.show()
            plt.pause(0.001)  # Add a short pause to update the figure
            # plt.figure(figsize=(15, 10))
            # plt.plot(learning_rate_his, label='Learning rate')
            # plt.legend(loc='best')
            # plt.grid(True)
            # plt.xlabel('Epoch')
            # plt.ylabel('Learning rate')
            if i == epochs:
                plt.savefig(f'{path}\Model_Image\{model_name}_lr.png', dpi=400)  # Save the figure
            # plt.show()
            plt.pause(0.001)  # Add a short pause to update the figure
    end_time = time.time()
    plt.ioff()  # Disable interactive mode

    torch.save(regressor.state_dict(), model_path)

    elapsed_time = end_time - start_time
    print("Total training time: {:.2f} seconds".format(elapsed_time))

if __name__ == '__main__':
    main()