file_name='Tracking_SyntheticImage_202303121113.pth' # 副檔名通常以.pt或.pth儲存，建議使用.pth
import torch
device=torch.device('cuda') # 'cuda'/'cpu'，import torch
num_outputs=1
train_size=148
valid_size=2
batch_size=4
learning_rate=0.0001
step_size=1000 # Reriod of learning rate decay
epochs=10
TrainingImage='H:\\project\\software\\TrainingImage/'
Annotation='H:\\project\\software\\Annotation/'
from torchvision import transforms
transforms=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]) # ToTensor將影像像素歸一化至0~1(直接除以255)，from torchvision import transforms

# 建立dataset
from torch.utils.data import Dataset
class ImageLabel(Dataset): # from torch.utils.data import Dataset
    def __init__(self,img,xy):
        self.img=img
        self.xy=xy
    def __getitem__(self,idx):
        return self.img[idx],self.xy[idx]
    def __len__(self):
        return len(self.img)
import os
all_image_name=os.listdir(TrainingImage) # 所有影像檔名(含.jpg)，import os
img=list()
xy=list()
from PIL import Image
import xml.etree.ElementTree as ET
for image_name in all_image_name:
    I=Image.open(TrainingImage+image_name,mode='r') # from PIL import Image
    I=transforms(I) # from torchvision import transforms
    img.append(I) # 列表長度為影像個數，列表中每個元素為一個[3,300,300]的tensor
    image_name=image_name[:-4] # 移除4個字元(.jpg)
    root=ET.parse(Annotation+image_name+'.xml').getroot() # 獲取xml文件物件的根結點，import xml.etree.ElementTree as ET
    size=root.find('size') # 獲取size子結點
    width=int(size.find('width').text) # 原始影像的寬(像素)
    height=int(size.find('height').text) # 原始影像的高(像素)
    width_scale=I.size(1)/width # 輸入影像與原始影像的寬比
    height_scale=I.size(1)/height # 輸入影像與原始影像的高比
    for object in root.iter('object'): # 遞迴查詢所有的object子結點
        bndbox=object.find('bndbox')
        #chi_en=torch.Tensor([int(bndbox.find('xmin').text)*width_scale,int(bndbox.find('ymin').text)*height_scale]) # [num_outputs]，import torch  
        chi_en=torch.Tensor([int(bndbox.find('xmin').text)*width_scale])   
    xy.append(chi_en) # 列表長度為影像個數，列表中每個元素為一個[num_outputs]的tensor
dataset=ImageLabel(img,xy)
train_data,valid_data=torch.utils.data.random_split(dataset,[train_size,valid_size]) # import torch
train_loader=torch.utils.data.DataLoader(train_data,batch_size,shuffle=True) # imort torch
valid_loader=torch.utils.data.DataLoader(valid_data,batch_size,shuffle=True) # imort torch

from torch import nn
class VGG16_Model(nn.Module):
    def __init__(self):
        super(VGG16_Model,self).__init__()
        self.vgg16=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1), # [batch_size,64,224,224]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1), # [batch_size,64,224,224]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2), # [batch_size,64,112,112]

            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1), # [batch_size,128,112,112]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1), # [batch_size,128,112,112]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2), # [batch_size,128,56,56]

            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1), # [batch_size,256,56,56]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1), # [batch_size,256,56,56]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1), # [batch_size,256,56,56]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2), # [batch_size,256,28,28]

            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1), # [batch_size,512,28,28]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1), # [batch_size,512,28,28] 
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1), # [batch_size,512,28,28]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2), # [batch_size,512,14,14]

            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1), # [batch_size,512,14,14]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1), # [batch_size,512,14,14]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1), # [batch_size,512,14,14]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2), # [batch_size,512,7,7]
        )
        self.fc=nn.Sequential(
            nn.Linear(512*7*7,4096),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(4096,num_outputs),
        )

    def forward(self,x):
        x=self.vgg16(x)
        x=x.view(x.size(0),-1)
        x=self.fc(x)
        return x

regressor=VGG16_Model().to(device)
criterion=nn.MSELoss() # 回歸
optimizer=torch.optim.Adam(regressor.parameters(),lr=learning_rate)
#optimizer=torch.optim.SGD(regressor.parameters(),lr=learning_rate)
scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size,0.1)

train_losses_his,valid_losses_his=[],[]
for i in range(1,epochs+1):
    print('Running Epoch:'+str(i))
    train_loss,train_total,valid_loss,valid_total=0,0,0,0
    regressor.train()
    for img,value in train_loader: # 一個batch的image、value。image：[batch_size,3,224,224]，value：[batch_size,num_outputs]
        img,value=img.to(device),value.to(device)
        pred=regressor(img) # pred：[batch_size,num_outputs]
        loss=criterion(pred,value) # loss.item()：一個batch的平均loss，[1]
        train_loss+=loss.item()*img.size(0) # 累加計算每一epoch的loss總和。loss.item()：一個batch的平均loss，[1]。image.size(0)：一個batch的訓練資料總數
        train_total+=img.size(0) # 累加計算訓練資料總數
        optimizer.zero_grad() # 權重梯度歸零
        loss.backward() # 計算每個權重的loss梯度
        optimizer.step() # 權重更新
    scheduler.step()

    regressor.eval()
    for img,value in valid_loader: # 一個batch的image、value。image：[batch_size,3,224,224]，value：[batch_size,1,num_outputs]
        img,value=img.to(device),value.to(device)
        pred=regressor(img) # pred：[batch_size,num_outputs]
        loss=criterion(pred,value) # loss.item()：一個batch的平均loss，[1]
        valid_loss+=loss.item()*img.size(0) # 累加計算每一epoch的loss總和。loss.item()：一個batch的平均loss，[1]。image.size(0)：一個batch的驗證資料總數
        valid_total+=img.size(0) # 累加計算驗證資料總數

    train_loss=train_loss/train_total # 計算每一個epoch的平均訓練loss
    valid_loss=valid_loss/valid_total # 計算每一個epoch的平均驗證loss
    train_losses_his.append(train_loss) # 累積記錄每一個epoch的平均訓練loss，[epochs]
    valid_losses_his.append(valid_loss) # 累積記錄每一個epoch的平均驗證loss，[epochs]
    print('Training Loss='+str(train_loss))
    print('Validation Loss='+str(valid_loss))

import matplotlib.pyplot as plt
plt.semilogy(train_losses_his,'b',label='training loss')
plt.semilogy(valid_losses_his,'r',label='validation loss')
plt.title('Loss')
plt.legend(loc='best')
plt.show()

torch.save(regressor.state_dict(),file_name) # import torch