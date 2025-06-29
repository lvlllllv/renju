import numpy as np
import torch
import random

import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from torch import nn
import torch.nn.functional as F

from models import renju

import matplotlib.pyplot as plt

#################################################################
####################################################################数据处理

from itertools import accumulate

import re

seed = 783
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


data_path = r'D:\machine_learning\HWF\data\good_ones.txt'

class Datapre():
    def __init__(self, path):
        #文件路径
        self.path = path
        with open(path, 'r') as file:
            #打开文件并移除末尾换行符
            self.lines = [re.sub(r' +', ' ', line.strip()) for line in file.readlines()]

        #棋谱数量
        self.data_len = len(self.lines)

    def get_dataloader(self):
        dataset = Mydataset(self.lines)
        
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size

        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

        return train_loader, test_loader



class Mydataset(Dataset):
    def __init__(self, lines):
        self.lines = lines
        self.len_games = len(self.lines)
        self.datasize = self._get_datasize(self.lines)#数组，每一项为每一盘对局样本数
        self.fix_datasize = list(accumulate(self.datasize))
        

    def __len__(self):
        return sum(self.datasize)

    def _get_xy(self, sentence):
        #返回棋谱落子的坐标[(x,y),(),()]
        coords = []
    # 先分割字符串为坐标列表
        elements = sentence.split()
        for elem in elements:
            # 改用 \d+ 匹配多位数字
            match = re.match(r"([A-Za-z])(\d+)", elem)
            if match:
                letter = match.group(1).lower()
                number = int(match.group(2)) - 1  # 行号减1
                x = ord(letter) - ord('a')        # 字母转0-based
                coords.append((x, number))
        return coords

    def _get_datasize(self, lines):
        #返回每一局的样本数量
        return [s.count(' ') for s in lines]

    def _find_data(self, target):
        left, right = 0, self.len_games - 1
        ans = -1
        while left <= right:
            mid = (left + right) // 2
            if self.fix_datasize[mid] > target:
                ans = mid  # 记录可能的位置
                right = mid - 1  # 继续向左寻找更小的索引
            else:
                left = mid + 1
        return ans  # 找不到时返回-1

    def _get_board(self, game, idx):
        manual = np.zeros([2, 15, 15])
        politic = np.zeros([15, 15])
        if idx == 0:
            politic[game[0][0]][game[0][1]] = 1
            return manual, politic
        else:
            for i in range(idx):
                manual[i % 2][game[i][0]][game[i][1]] = 1#通道0为黑，通道1为白
            if idx % 2 != 0:#此时应该将白棋放在通道0
                manual = manual[::-1, :, :]
                manual = manual.copy()
            
            politic[game[idx][0]][game[idx][1]] = 1

        return manual, politic
    
    def _transform_board(self, board, mode, is_label=False):
        """mode: 0-7，对应8种变换"""
        if not is_label:
            if mode == 0:  # 原图
                return board
            elif mode == 1:
                return np.rot90(board, 1, axes=(1, 2)).copy()
            elif mode == 2:
                return np.rot90(board, 2, axes=(1, 2)).copy()
            elif mode == 3:
                return np.rot90(board, 3, axes=(1, 2)).copy()
            elif mode == 4:
                return np.flip(board, axis=2).copy()
            elif mode == 5:
                return np.flip(board, axis=1).copy()
            elif mode == 6:
                return np.transpose(board, (0, 2, 1)).copy()
            elif mode == 7:
                return np.flip(np.transpose(board, (0, 2, 1)), axis=2).copy()
        else:
            if mode == 0:
                return board
            elif mode == 1:
                return np.rot90(board, 1).copy()
            elif mode == 2:
                return np.rot90(board, 2).copy()
            elif mode == 3:
                return np.rot90(board, 3).copy()
            elif mode == 4:
                return np.fliplr(board).copy()
            elif mode == 5:
                return np.flipud(board).copy()
            elif mode == 6:
                return board.T.copy()
            elif mode == 7:
                return np.fliplr(board.T).copy()
        

    def __getitem__(self, idx):
        ans = self._find_data(idx)#获取对局的序号
        game_str = self.lines[ans]#获取对局的字符串记录
        
        game = self._get_xy(game_str)
        
        if ans == 0:
            idx = idx  # 第一个对局直接使用原始索引
        else:
            idx = idx - self.fix_datasize[ans-1]  # 减去前一个对局的累计样本数

        sample, label = self._get_board(game, idx)

        transform_idx = np.random.randint(8)

        sample = self._transform_board(sample, transform_idx)
        label = self._transform_board(label, transform_idx, is_label=True)

        sample_tensor = torch.tensor(sample, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return sample_tensor, label_tensor
    


data_pre = Datapre(data_path)#实例化数据加载器

train_loader, test_loader = data_pre.get_dataloader()#加载包装好的数据

###################################################################
    
model = renju(board_size = 15)

####################################################################自定义损失函数

def manual_binary_cross_entropy_with_logits(inputs, targets, reduction='mean'):
    # 计算二元交叉熵损失
    loss = -targets * torch.log(inputs + 1e-10) - (1 - targets) * torch.log(1 - inputs + 1e-10)
    
    # 根据指定的 reduction 方式进行聚合
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError("Invalid reduction mode. Use 'mean', 'sum', or 'none'.")

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=1.55, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 计算标准BCE损失
        BCE_loss = manual_binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # 计算焦点损失
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        return F_loss
        
criterion = FocalLoss()


###################################################################
###################################################################预训练神经网络

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
model.load_state_dict(torch.load(r'D:\machine_learning\HWF\model\Miami_o7.pth'))

torch.backends.cudnn.benchmark = True#加速卷积运算
torch.cuda.empty_cache() #初始化前清空缓存


num_epochs = 1

test_loss_min = np.inf
lr = 0#初始化学习率


#记录训练集损失
all_train_loss = []
#记录测试集损失
all_test_loss = []

for epoch in range(num_epochs):

    if epoch <2:
        lr = 0.001
    elif epoch <5:
        lr = 0.0002
    elif epoch <15:
        lr = 0.000005
    elif epoch <25:
        lr = 0.000003
    else:
        lr = 0.000005

    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loss = 0

    model.train()
    for i ,(state, label) in enumerate(train_loader):
            
        # 迁移数据到设备并前向传播
        state = state.to(device)
        label = label.to(device)
            
        optimizer.zero_grad()  # 清零梯度
        output = model(state).squeeze()

        loss = criterion(output, label)
        train_loss +=loss.item()

        # 反向传播与优化
        loss.backward()
        optimizer.step()

        if i % 1000 == 0 and i !=0:
            train_loss = train_loss / 1000
            all_train_loss.append(train_loss)

            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {train_loss:.8f}')
            torch.save(model.state_dict(), r'D:\machine_learning\HWF\model\Miami_r4.pth')
            train_loss = 0


        del state, output, label, loss
        torch.cuda.empty_cache()
        
    #model.eval()

    #test_loss = 0

    #with torch.no_grad():
    #    for i ,(state, label) in enumerate(test_loader):
    #        
    #        state = state.to(device)
    #        label = label.to(device)

    #        output = model(state).squeeze()
    #        loss_t = criterion(output, label)
        
    #        test_loss += loss_t.item()

    #        del state, output, label, loss_t
    #        torch.cuda.empty_cache()

    #test_loss = test_loss / len(test_loader)

    #all_test_loss.append(test_loss)

    #if test_loss < test_loss_min :
    #    torch.save(model.state_dict(), r'D:\machine_learning\HWF\model\Miami_o7.pth')
    #    print(f"已保存模型, test_loss={test_loss}")
    #    test_loss_min = test_loss

plt.figure(figsize=(12, 9))

# 绘制训练损失曲线
plt.plot(all_train_loss, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()


plt.show()