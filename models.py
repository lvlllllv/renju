import torch

from torch import nn
import torch.nn.functional as F

####################################################################
#renju类

class Resnet(nn.Module):
    """使用残差网络连接"""
    def __init__(self, in_channels, k_size=3):
        super(Resnet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=k_size, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=k_size, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + residual  # 残差连接
        return F.relu(x)


class renju(nn.Module):
    
    def __init__(self, board_size):
        super(renju, self).__init__()
        self.board_size = board_size#棋盘尺寸，后续不进行修改

        #公共特征提取
        self.conv_init3 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.conv_init5 = nn.Conv2d(2, 64, kernel_size=5, padding=2)
        self.bn_init = nn.BatchNorm2d(128)

        self.res_blocks = nn.Sequential(
            Resnet(in_channels=128, k_size=3),
            Resnet(in_channels=128, k_size=3),
            Resnet(in_channels=128, k_size=3),
            Resnet(in_channels=128, k_size=3),

            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            Resnet(in_channels=256, k_size=3),
            Resnet(in_channels=256, k_size=3),
            Resnet(in_channels=256, k_size=3),
            Resnet(in_channels=256, k_size=3)
        )
        
        #预测落子层
        self.policy_conv = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        self.policy_fc = nn.Sequential(
            nn.Linear(32 * board_size**2, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            
            nn.Linear(512, board_size**2)
        )

    def create_legal_mask(self, x):
        #生成合法位置掩码（考虑已有棋子）
        mask = torch.ones((x.size(0), 1, self.board_size, self.board_size), device=x.device)
        
        # 检查已有棋子位置
        black = (x[:,0] > 0.5).float().unsqueeze(1)
        white = (x[:,1] > 0.5).float().unsqueeze(1)

        mask = mask * (1 - black) * (1 - white)#没有棋子的地方是1，有棋子是0
    
        return mask  # (b,1,board,board)
        

    def forward(self, x):

        mask = self.create_legal_mask(x)
        
        x1 = self.conv_init3(x)
        x2 = self.conv_init5(x)
        
        x = torch.cat([x1, x2], dim=1)
        
        x = self.bn_init(x)
        
        x = F.relu(x)

        x = self.res_blocks(x)

        p = self.policy_conv(x)
        p = p.view(-1, 32 * self.board_size**2)
        p = self.policy_fc(p)

        p = F.softmax(p, dim=1)
        #使用交叉熵损失不用softmax

        p = p.view(-1, 1, self.board_size, self.board_size)

        p_mask = p * mask

        p_mask = torch.clamp(p_mask, min=1e-10)

        p_mask = p_mask / (p_mask.sum(dim=(1,2,3), keepdim=True) + 1e-10)

        return p_mask#p_out(b,1,15,15)
    
