import os
import numpy as np
import torch
import random


import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from models import renju
from game import Game

EPISODE = 5000  # 训练轮次
STEP = 60

GAMMA = 0.88  # 衰减系数
LEARNING_RATE = 0.0003  # 学习率

REWARD_WIN = 1  # 获胜奖励
REWARD_FAIL = -1  # 输掉奖励
REWARD_DRAW = -0.2  # 平局奖励
REWARD_NONE = 0  # 平常奖励
REWARD_MOVE = 0

opponent_epsilon = 0.6#对手随机性

model_dir = r'D:\machine_learning\HWF\pre_model'

def load_pre_model(model_dir, pattern='Miami_p'):
    model_files = [f for f in os.listdir(model_dir) if f.startswith(pattern) and f.endswith('.pth')]
    model_paths = [os.path.join(model_dir, f) for f in model_files]
    
    if not model_paths:
        raise ValueError(f"No models found matching pattern '{pattern}' in directory {model_dir}")
    
    models = []
    for path in model_paths:
        model = renju(board_size=15)
        model.load_state_dict(torch.load(path))
        model.eval()
        model = model.to(device)
        models.append(model)
    return models


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

pre_model = load_pre_model(model_dir)

    
model = renju(board_size = 15)

model = model.to(device)
model.load_state_dict(torch.load(r'D:\machine_learning\HWF\model\Miami_r4.pth'))

model.train()

class REINFORCE:
    def __init__(self, epsilon=0.4):
        
        self.time_step = 0
        self.states, self.actions, self.action_probs, self.rewards = [], [], [], []
        self.net = model#加载模型
        self.optim = torch.optim.Adam(self.net.parameters(), lr=LEARNING_RATE)
        self.epsilon = epsilon  # 探索率
        self.ep_reward = []

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=10000, gamma=0.8)

    def predict(self, observation, deterministic=False):#observation应当为(2,15,15)的numpy数组

        observation = observation.to(device)

        action_score = self.net(observation)

        probs = action_score[0].squeeze()

        if not deterministic and np.random.rand() < self.epsilon:
            # 随机探索合法动作
            legal_mask = (probs > 1e-6).flatten()
            legal_indices = torch.nonzero(legal_mask, as_tuple=False)
            if legal_indices.size(0) == 0:
                # 无合法动作时随机选择（理论上不会发生）
                action_idx = torch.randint(0, 225, (1,))
            else:
                action_idx = legal_indices[torch.randint(0, len(legal_indices), (1,))]
            action_idx = action_idx.item()
            x, y = divmod(action_idx, 15)
        else:
            # 贪心选择
            action_idx = torch.argmax(probs.flatten()).item()
            x, y = divmod(action_idx, 15)
        
        return (x, y), probs

    def store_transition(self, s, a, p, r):
        self.states.append(s)
        self.actions.append(a)#a(x,y)的元组，表示落子位置
        self.action_probs.append(p)#p是15*15张量
        self.rewards.append(r)

    def learn(self):
        loss = []
        len_state = len(self.actions)

        #计算G
        G = [self.rewards[-1]]
        g = self.rewards[-1]
        for i in range(len_state - 1):
            g = self.rewards[len_state - 2 - i] + GAMMA * g
            G.append(g)
        
        G.reverse()#反转数组，使与其余数组顺序相同

        with torch.no_grad():
            G = torch.tensor(G, dtype=torch.float32).to(device)
            G = (G - G.mean()) / (G.std() + 1e-8)


        baseline = sum(self.ep_reward[-50:]) / len(self.ep_reward[-50:]) if len(self.ep_reward) >= 50 else 0

        for i in range(len_state):
            x, y =self.actions[i]
            log_prod = torch.log(self.action_probs[i].to(device)[x, y] + 1e-10)

            loss_up = - (G[i] - baseline) * log_prod
            loss.append(loss_up.unsqueeze(0))
        
        self.optim.zero_grad()

        loss = torch.cat(loss).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
        self.optim.step()

        self.ep_reward.append(sum(self.rewards))

        self.states, self.actions, self.action_probs, self.rewards = [], [], [], []

        self.scheduler.step()
        return loss.item()
    
    ###############################################################################


    
agent = REINFORCE(epsilon = 0.8)

for episode in range(EPISODE):
    agent.epsilon = max((0.8 * np.exp(- episode / 900)), 0.1)

    # 在训练循环中添加
    opponent_epsilon = max(0.6 * np.exp(-episode / 1000), 0.1)

    game = Game()

    step = 0

    if episode % 100 == 0:
        torch.save(model.state_dict(), f'D:/machine_learning/HWF/pre_model/Miami_p_ep{episode}.pth')

    if episode % 2 == 0:#后手
        game.black_play(7, 7)
        while True:
            step += 1
            state = game.white_ai()
            action, probs = agent.predict(state)
            game.white_play(action[0], action[1])

            if game.check_win(game.white_board):
                reward = REWARD_WIN
                agent.store_transition(state, action, probs, reward)
                print(f"{episode}/{EPISODE}:白胜(智能体胜利)")
                break

            now_reward = game.white_reward()

            model_test = random.choice(pre_model)

            test_play = game.black_ai()

            test_play = test_play.to(device)

            output = model_test(test_play).squeeze()
            probs = output.view(15, 15)

            if np.random.rand() < opponent_epsilon:
                # 随机从概率较大的位置选一个
                legal_mask = (probs > 1e-6).float().flatten()
                legal_indices = torch.nonzero(legal_mask, as_tuple=False)

                if legal_indices.numel() == 0:
                    idx = random.randint(0, 224)
                else:
                    idx = legal_indices[random.randint(0, legal_indices.size(0)-1)].item()
            else:
                idx = torch.argmax(probs.flatten()).item()

            x, y = divmod(idx, 15)

            game.black_play(x, y)

            if game.check_win(game.black_board):
                reward = REWARD_FAIL
                agent.store_transition(state, action, probs, reward)
                print(f"{episode}/{EPISODE}:黑胜")
                break
            elif step == STEP:
                reward = REWARD_DRAW
                agent.store_transition(state, action, probs, reward)
                print(f"{episode}/{EPISODE}:平局")
                break
            else:
                reward = now_reward + REWARD_MOVE

            agent.store_transition(state, action, probs, reward)

        agent.learn()

    else:#先手
        while True:
            step += 1

            state = game.black_ai()
            action, probs = agent.predict(state)
            game.black_play(action[0], action[1])

            if game.check_win(game.black_board):
                reward = REWARD_WIN
                agent.store_transition(state, action, probs, reward)
                print(f"{episode}/{EPISODE}:黑胜(智能体胜利)")
                break

            now_reward = game.black_reward()

            model_test = random.choice(pre_model)

            test_play = game.white_ai()

            test_play = test_play.to(device)

            output = model_test(test_play).squeeze()
            probs = output.view(15, 15)

            if np.random.rand() < opponent_epsilon:
                # 随机从概率较大的位置选一个
                legal_mask = (probs > 1e-6).float().flatten()
                legal_indices = torch.nonzero(legal_mask, as_tuple=False)

                if legal_indices.numel() == 0:
                    idx = random.randint(0, 224)
                else:
                    idx = legal_indices[random.randint(0, legal_indices.size(0)-1)].item()
            else:
                idx = torch.argmax(probs.flatten()).item()

            x, y = divmod(idx, 15)
            game.white_play(x, y)

            if game.check_win(game.white_board):
                reward = REWARD_FAIL
                agent.store_transition(state, action, probs, reward)
                print(f"{episode}/{EPISODE}:白胜")
                break
            elif step == STEP:
                reward = REWARD_DRAW
                agent.store_transition(state, action, probs, reward)
                print(f"{episode}/{EPISODE}:平局")
                break
            else:
                reward = now_reward + REWARD_MOVE

            agent.store_transition(state, action, probs, reward)

        agent.learn()

    if episode % 1000 ==0:
        torch.save(model.state_dict(), r'D:\machine_learning\HWF\model\Miami_o7.pth')
        print(f"{episode}:已成功保存模型")

    if episode == 800000:
        agent.optim = torch.optim.Adam(agent.net.parameters(), lr=LEARNING_RATE/2)

torch.save(model.state_dict(), r'D:\machine_learning\HWF\model\Miami_o7.pth')
print("已成功保存模型")

plt.figure(figsize=(12, 6))  # 设置画布大小

# 绘制原始奖励曲线
plt.plot(agent.ep_reward, 
         label='Raw Reward', 
         color='blue',
         alpha=0.7,
         linewidth=1)

window_size = 50  # 滑动窗口大小
if len(agent.ep_reward) > window_size:
    moving_avg = [sum(agent.ep_reward[i-window_size:i+1])/window_size for i in range(window_size-1, len(agent.ep_reward))]
    plt.plot(moving_avg, 
             label=f'Moving Average ({window_size} episodes)', 
             color='red',
             linewidth=2)

plt.title('奖励函数曲线', fontsize=14)
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Reward', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.show()