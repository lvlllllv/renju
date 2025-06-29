import pygame
import torch
import time
from models import renju
from game import Game

# 初始化模型
model = renju(board_size=15)
model.load_state_dict(torch.load(r'D:\machine_learning\HWF\model\Miami_r4.pth'))
model.eval()

# 常量
CELL_SIZE = 40
MARGIN = 60
BOARD_SIZE = 15
WIDTH = CELL_SIZE * BOARD_SIZE + 2 * MARGIN
HEIGHT = WIDTH
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BG_COLOR = (200, 180, 120)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("五子棋")
font = pygame.font.SysFont("Microsoft YaHei", 36)  # 黑体

clock = pygame.time.Clock()

def show_result(text):
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 150))
    screen.blit(overlay, (0, 0))

    result_font = pygame.font.SysFont("Microsoft YaHei", 72, bold=True)
    label = result_font.render(text, True, WHITE)
    rect = label.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    screen.blit(label, rect)
    pygame.display.flip()

# 按钮工具函数
def draw_button(text, rect, selected=False):
    pygame.draw.rect(screen, (180, 120, 60), rect)
    if selected:
        pygame.draw.rect(screen, (0, 200, 0), rect, 3)
    label = font.render(text, True, BLACK)
    text_rect = label.get_rect(center=rect.center)
    screen.blit(label, text_rect)

def wait_for_button(buttons):
    while True:
        screen.fill(BG_COLOR)
        for btn in buttons:
            draw_button(btn['text'], btn['rect'])
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                for btn in buttons:
                    if btn['rect'].collidepoint(mx, my):
                        return btn['id']
        clock.tick(30)

# 主逻辑入口
def main():
    game = Game()
    last_ai_move = None

    # 模式选择
    mode = wait_for_button([
        {'id': 'pvp', 'text': '真人对战', 'rect': pygame.Rect(WIDTH//2-100, HEIGHT//2-80, 200, 50)},
        {'id': 'pve', 'text': '人机对战', 'rect': pygame.Rect(WIDTH//2-100, HEIGHT//2+20, 200, 50)},
    ])

    if mode == 'pvp':
        player_vs_player(game)

    elif mode == 'pve':
        first = wait_for_button([
            {'id': 'player', 'text': '玩家先手', 'rect': pygame.Rect(WIDTH//2-100, HEIGHT//2-80, 200, 50)},
            {'id': 'ai', 'text': 'AI先手', 'rect': pygame.Rect(WIDTH//2-100, HEIGHT//2+20, 200, 50)},
        ])
        player_color = 'black' if first == 'player' else 'white'
        player_vs_ai(game, player_color)

def player_vs_player(game):
    current_player = 'black'
    running = True

    while running:
        game.draw_board_pygame(screen, CELL_SIZE, MARGIN)
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                x = round((mx - MARGIN) / CELL_SIZE)
                y = 14 - round((my - MARGIN) / CELL_SIZE)
                if 0 <= x < 15 and 0 <= y < 15 and (x, y) not in game.game_legal:
                    if current_player == 'black':
                        game.black_play(x, y)
                        if game.check_win(game.black_board):
                            game.draw_board_pygame(screen, CELL_SIZE, MARGIN)
                            pygame.display.flip()
                            print("黑方胜利！")
                            show_result("黑方胜利！")
                            time.sleep(3)
                            return
                        current_player = 'white'
                    else:
                        game.white_play(x, y)
                        if game.check_win(game.white_board):
                            game.draw_board_pygame(screen, CELL_SIZE, MARGIN)
                            pygame.display.flip()
                            print("白方胜利！")
                            show_result("白方胜利！")
                            time.sleep(3)
                            return
                        current_player = 'black'
        clock.tick(30)

def player_vs_ai(game, player_color):
    last_ai_move = None
    running = True
    ai_pending = False
    ai_start_time = 0

    # 如果 AI 先手
    if player_color == 'white':
        with torch.no_grad():
            output = model(game.black_ai()).squeeze().view(15, 15)
            for i, j in game.game_legal:
                output[i, j] = -1e9
            idx = torch.argmax(output)
            x, y = divmod(idx.item(), 15)
            game.black_play(x, y)
            last_ai_move = (x, y)

    while running:
        game.draw_board_pygame(screen, CELL_SIZE, MARGIN, last_ai_move)

        pygame.display.flip()

        current_time = time.time()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); exit()
            elif event.type == pygame.MOUSEBUTTONDOWN and not ai_pending:
                mx, my = event.pos
                x = round((mx - MARGIN) / CELL_SIZE)
                y = 14 - round((my - MARGIN) / CELL_SIZE)  # y轴反向

                if 0 <= x < 15 and 0 <= y < 15 and (x, y) not in game.game_legal:
                    if player_color == 'black':
                        game.black_play(x, y)
                        if game.check_win(game.black_board):
                            game.draw_board_pygame(screen, CELL_SIZE, MARGIN, last_ai_move)
                            pygame.display.flip()
                            print("玩家胜利！")
                            show_result("玩家胜利！")
                            time.sleep(3)
                            return
                    else:
                        game.white_play(x, y)
                        if game.check_win(game.white_board):
                            game.draw_board_pygame(screen, CELL_SIZE, MARGIN, last_ai_move)
                            pygame.display.flip()
                            print("玩家胜利！")
                            show_result("玩家胜利！")
                            time.sleep(3)
                            return

                    # 设置 AI 延迟响应
                    ai_pending = True
                    ai_start_time = current_time

        # AI 延迟响应逻辑
        if ai_pending and (current_time - ai_start_time >= 3):
            with torch.no_grad():
                ai_input = game.white_ai() if player_color == 'black' else game.black_ai()
                output = model(ai_input).squeeze().view(15, 15)
                for i, j in game.game_legal:
                    output[i, j] = -1e9
                idx = torch.argmax(output)
                ai_x, ai_y = divmod(idx.item(), 15)
                last_ai_move = (ai_x, ai_y)

                if player_color == 'black':
                    game.white_play(ai_x, ai_y)
                    if game.check_win(game.white_board):
                        game.draw_board_pygame(screen, CELL_SIZE, MARGIN, last_ai_move)
                        pygame.display.flip()
                        print("AI 胜利！")
                        show_result("AI 胜利！")
                        time.sleep(3)
                        return
                else:
                    game.black_play(ai_x, ai_y)
                    if game.check_win(game.black_board):
                        game.draw_board_pygame(screen, CELL_SIZE, MARGIN, last_ai_move)
                        pygame.display.flip()
                        print("AI 胜利！")
                        show_result("AI 胜利！")
                        time.sleep(3)
                        return

            ai_pending = False

        clock.tick(30)


if __name__ == "__main__":
    main()
