import numpy as np
import torch
import random

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import pygame

class Game():
    def __init__(self, board_size = 15):
        self.board_size = board_size

        self.black_board = np.zeros([15, 15])
        self.white_board = np.zeros([15, 15])

        self.game_legal = []#记录比赛过程

    def black_play(self, x, y):
        if (x, y) not in self.game_legal:
            self.game_legal.append((x, y))
            self.black_board[x][y] = 1

    def white_play(self, x, y):
        if (x, y) not in self.game_legal:
            self.game_legal.append((x, y))
            self.white_board[x][y] = 1

    def check_win(self, board):
        for i in range(15):
            for j in range(11):
                if np.all(board[i, j:j+5] == 1):
                    return True
    
        # 检查纵向连续五子
        for j in range(15):
            for i in range(11):
                if np.all(board[i:i+5, j] == 1):
                    return True
    
        # 检查左上到右下斜向连续五子
        for i in range(11):
            for j in range(11):
                if np.all(board[i:i+5, j:j+5].diagonal() == 1):
                    return True
    
        # 检查右上到左下斜向连续五子
        for i in range(11):
            for j in range(4, 15):
                submatrix = board[i:i+5, j-4:j+1]
                if np.all(np.fliplr(submatrix).diagonal() == 1):
                    return True
    
        return False


    def draw_board_pygame(self, screen, cell_size, margin, last_ai_move=None):
        screen.fill((240, 200, 120))
        font = pygame.font.SysFont(None, 24)
        board_size = self.board_size
        width = cell_size * board_size + 2 * margin
        height = width

        pygame.draw.rect(screen, (0, 0, 0), 
                         (margin - 2, margin - 2, 
                          cell_size * (board_size - 1) + 4, 
                          cell_size * (board_size - 1) + 4), 2)

        for i in range(board_size):
            # 水平线
            pygame.draw.line(screen, (0, 0, 0),
                             (margin, margin + i * cell_size),
                             (margin + (board_size - 1) * cell_size, margin + i * cell_size), 1)
            # 垂直线
            pygame.draw.line(screen, (0, 0, 0),
                             (margin + i * cell_size, margin),
                             (margin + i * cell_size, margin + (board_size - 1) * cell_size), 1)


        # 星位
        stars = [(3,3), (11,3), (3,11), (11,11), (7,7)]
        for i, j in stars:
            pygame.draw.circle(screen, (0, 0, 0),
                               (margin + i * cell_size, margin + j * cell_size), 5)

        # 坐标标记
        for i in range(board_size):
            label = font.render(str(i), True, (0, 0, 0))
            screen.blit(label, (margin + i * cell_size - 5, height - margin + 5))
            screen.blit(label, (margin - 25, margin + (board_size - 1 - i) * cell_size - 5))

        # 棋子
        for i in range(board_size):
            for j in range(board_size):
                x = margin + i * cell_size
                y = margin + (self.board_size - 1 - j) * cell_size
                if self.black_board[i, j] == 1:
                    pygame.draw.circle(screen, (0, 0, 0), (x, y), 15)
                elif self.white_board[i, j] == 1:
                    pygame.draw.circle(screen, (255, 255, 255), (x, y), 15)
                    pygame.draw.circle(screen, (0, 0, 0), (x, y), 15, 1)

        # 绿色标记落子
        if last_ai_move:
            i, j = last_ai_move
            cx = margin + i * cell_size
            cy = margin + (self.board_size - 1 - j) * cell_size
            pygame.draw.circle(screen, (0, 255, 0), (cx, cy), 5)

        pygame.display.flip()


    def black_ai(self):
        broad = np.stack([self.black_board, self.white_board], axis=0)
        broad = torch.tensor(broad, dtype=torch.float32).unsqueeze(0)
        return broad

    def white_ai(self):
        broad = np.stack([self.white_board, self.black_board], axis=0)
        broad = torch.tensor(broad, dtype=torch.float32).unsqueeze(0)
        return broad
    
    def num_three(self, own_board, other_board):#判断有几个没堵的3
        num = 0
        for i in range(15):
            for j in range(11):
                if (own_board[i, j]==0 
                    and own_board[i, j+1]==1 
                    and own_board[i, j+2]==1 
                    and own_board[i, j+3]==1 
                    and own_board[i, j+4]==0 
                    and other_board[i,j]==0 
                    and other_board[i,j+4]==0):
                    num += 1
    
        # 检查纵向连续五子
        for j in range(15):
            for i in range(11):
                if (own_board[i, j]==0 
                    and own_board[i+1, j]==1 
                    and own_board[i+2, j]==1 
                    and own_board[i+3, j]==1 
                    and own_board[i+4, j]==0 
                    and other_board[i,j]==0 
                    and other_board[i+4,j]==0):
                    num += 1
    
        # 检查左上到右下斜向连续五子
        for i in range(11):
            for j in range(11):
                if (own_board[i, j]==0 
                    and own_board[i+1, j+1]==1 
                    and own_board[i+2, j+2]==1 
                    and own_board[i+3, j+3]==1 
                    and own_board[i+4, j+4]==0
                    and other_board[i,j] == 0
                    and other_board[i+4, j+4]==0):
                    num += 1
    
        # 检查右上到左下斜向连续五子
        for i in range(11):
            for j in range(4, 15):
                if (own_board[i, j]==0 
                    and own_board[i+1, j-1]==1 
                    and own_board[i+2, j-2]==1 
                    and own_board[i+3, j-3]==1 
                    and own_board[i+4, j-4]==0
                    and other_board[i, j]==0
                    and other_board[i+4, j-4]==0):
                    num += 1
    
        return num
    
    def num_four(self, own_board, other_board):#判断有几个没堵的4
        num = 0
        for i in range(15):
            for j in range(11):
                if (((own_board[i, j]==1 and other_board[i,j+4]==1) 
                    or (own_board[i, j+4]==1 and other_board[i,j]==1))
                    and own_board[i, j+1]==1 
                    and own_board[i, j+2]==1 
                    and own_board[i, j+3]==1 ):
                    num += 1
    
        # 检查纵向连续五子
        for j in range(15):
            for i in range(11):
                if (((own_board[i, j]==1 and other_board[i+4,j]==1) 
                    or (own_board[i+4, j]==1 and other_board[i,j]==1))
                    and own_board[i+1, j]==1 
                    and own_board[i+2, j]==1 
                    and own_board[i+3, j]==1 ):
                    num += 1
    
        # 检查左上到右下斜向连续五子
        for i in range(11):
            for j in range(11):
                if (((own_board[i, j]==1 and other_board[i+4,j+4]==1) 
                    or (own_board[i+4, j+4]==1 and other_board[i,j]==1))
                    and own_board[i+1, j+1]==1 
                    and own_board[i+2, j+2]==1 
                    and own_board[i+3, j+3]==1 ):
                    num += 1
    
        # 检查右上到左下斜向连续五子
        for i in range(11):
            for j in range(4, 15):
                if (((own_board[i, j]==1 and other_board[i+4,j-4]==1) 
                    or (own_board[i+4, j-4]==1 and other_board[i,j]==1))
                    and own_board[i+1, j-1]==1 
                    and own_board[i+2, j-2]==1 
                    and own_board[i+3, j-3]==1 ):
                    num += 1
        return num
    
    def stem_three(self, own_board, other_board):#这一手堵住了几个3
        num = 0
        x, y =self.game_legal[-1]
        if x <= 10:
            if (other_board[x+1,y] == 1 
                and other_board[x+2,y]==1 
                and other_board[x+3,y] ==1 
                and own_board[x+4, y] ==0):
                num += 1

        if x >= 4:
            if (other_board[x-1,y] == 1 
                and other_board[x-2,y]==1 
                and other_board[x-3,y] ==1 
                and own_board[x-4, y] ==0):
                num += 1

        if y <= 10:
            if (other_board[x,y+1] == 1 
                and other_board[x,y+2]==1 
                and other_board[x,y+3] ==1 
                and own_board[x, y+4] ==0):
                num += 1

        if y >= 4:
            if (other_board[x,y-1] == 1 
                and other_board[x,y-2]==1 
                and other_board[x,y-3] ==1 
                and own_board[x, y-4] ==0):
                num += 1

        if x <= 10 and y <=10:
            if (other_board[x+1,y+1] == 1 
                and other_board[x+2,y+2]==1 
                and other_board[x+3,y+3] ==1 
                and own_board[x+4, y+4] ==0):
                num += 1

        if x >= 4 and y >=4:
            if (other_board[x-1,y-1] == 1 
                and other_board[x-2,y-2]==1 
                and other_board[x-3,y-3] ==1 
                and own_board[x-4, y-4] ==0):
                num += 1

        if x <= 10 and y >=4:
            if (other_board[x+1,y-1] == 1 
                and other_board[x+2,y-2]==1 
                and other_board[x+3,y-3] ==1 
                and own_board[x+4, y-4] ==0):
                num += 1

        if x >= 4 and y <=10:
            if (other_board[x-1,y+1] == 1 
                and other_board[x-2,y+2]==1 
                and other_board[x-3,y+3] ==1 
                and own_board[x-4, y+4] ==0):
                num += 1
        return num

    def stem_four(self, own_board, other_board):#这一手堵住了几个4
        num = 0
        x, y =self.game_legal[-1]
        if x <= 9:
            if (other_board[x+1,y] == 1 
                and other_board[x+2,y]==1 
                and other_board[x+3,y] ==1 
                and other_board[x+4,y]==1
                and own_board[x+5, y] ==1):
                num += 1
        elif x == 10:
            if (other_board[x+1,y] == 1 
                and other_board[x+2,y]==1 
                and other_board[x+3,y] ==1 
                and other_board[x+4,y]==1):
                num += 1

        if x >= 5:
            if (other_board[x-1,y] == 1 
                and other_board[x-2,y]==1 
                and other_board[x-3,y] ==1 
                and other_board[x-4,y]==1
                and own_board[x-5, y] ==1):
                num += 1
        elif x ==4:
            if (other_board[x-1,y] == 1 
                and other_board[x-2,y]==1 
                and other_board[x-3,y] ==1 
                and other_board[x-4,y]==1):
                num += 1

        if y <= 9:
            if (other_board[x,y+1] == 1 
                and other_board[x,y+2]==1 
                and other_board[x,y+3] ==1 
                and other_board[x,y+4] ==1
                and own_board[x, y+5] ==1):
                num += 1
        elif y ==10:
            if (other_board[x,y+1] == 1 
                and other_board[x,y+2]==1 
                and other_board[x,y+3] ==1 
                and other_board[x,y+4] ==1):
                num += 1

        if y >= 5:
            if (other_board[x,y-1] == 1 
                and other_board[x,y-2]==1 
                and other_board[x,y-3] ==1 
                and other_board[x, y-4] ==1
                and own_board[x, y-5] ==1):
                num += 1
        elif y ==4:
            if (other_board[x,y-1] == 1 
                and other_board[x,y-2]==1 
                and other_board[x,y-3] ==1 
                and other_board[x, y-4] ==1):
                num += 1

        if x <= 9 and y <=9:
            if (other_board[x+1,y+1] == 1 
                and other_board[x+2,y+2]==1 
                and other_board[x+3,y+3] ==1 
                and other_board[x+4,y+4] ==1
                and own_board[x+5, y+5] ==1):
                num += 1
        elif (x==10 and y <=9) or (y ==10 and x <= 9):
            if (other_board[x+1,y+1] == 1 
                and other_board[x+2,y+2]==1 
                and other_board[x+3,y+3] ==1 
                and other_board[x+4,y+4] ==1):
                num += 1

        if x >= 5 and y >=5:
            if (other_board[x-1,y-1] == 1 
                and other_board[x-2,y-2]==1 
                and other_board[x-3,y-3] ==1 
                and other_board[x-4,y-4] ==1
                and own_board[x-5, y-5] ==1):
                num += 1
        elif (x ==4 and y >=5 ) or (y ==4 and x >=5):
            if (other_board[x-1,y-1] == 1 
                and other_board[x-2,y-2]==1 
                and other_board[x-3,y-3] ==1 
                and other_board[x-4,y-4] ==1):
                num += 1

        if x <= 9 and y >=5:
            if (other_board[x+1,y-1] == 1 
                and other_board[x+2,y-2]==1 
                and other_board[x+3,y-3] ==1 
                and other_board[x+4,y-4]==1
                and own_board[x+5, y-5] ==1):
                num += 1
        elif (x==10 and y >=5) or (y == 4 and x <= 9):
            if (other_board[x+1,y-1] == 1 
                and other_board[x+2,y-2]==1 
                and other_board[x+3,y-3] ==1 
                and other_board[x+4,y-4]==1):
                num += 1

        if x >= 5 and y <=9:
            if (other_board[x-1,y+1] == 1 
                and other_board[x-2,y+2]==1 
                and other_board[x-3,y+3] ==1 
                and other_board[x-4,y+4]==1
                and own_board[x-5, y+5] ==1):
                num += 1
        elif (x==4 and y <=9) or (y==10 and x>=5):
            if (other_board[x-1,y+1] == 1 
                and other_board[x-2,y+2]==1 
                and other_board[x-3,y+3] ==1 
                and other_board[x-4,y+4]==1):
                num += 1
        return num
    
    def if_middle(self):
        x, y =self.game_legal[-1]
        if x == 7 and y == 7:
            return 0.1
        return 0
    
    def white_reward(self):
        reward = (self.stem_three(self.white_board, self.black_board) 
                  + self.stem_four(self.white_board, self.black_board) 
                  - 4 * self.num_three(self.black_board, self.white_board)#没有堵的对面的三子
                  -4* self.num_four(self.black_board, self.white_board)
                  + self.num_three(self.white_board, self.black_board)
                  +self.num_four(self.white_board, self.black_board)
                  + self.if_middle())/16
        return reward
    
    def black_reward(self):
        reward = (self.stem_three(self.black_board, self.white_board) 
                  + self.stem_four(self.black_board, self.white_board) 
                  -4* self.num_three(self.white_board, self.black_board)
                  -4 * self.num_four(self.white_board, self.black_board)
                  + self.num_three(self.black_board, self.white_board)
                  +self.num_four(self.black_board, self.white_board)
                  + self.if_middle())/16
        return reward
