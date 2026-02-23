import numpy as np
from block_puzzle_env.environment import BlockPuzzleEnv
from block_puzzle_env.pieces import PIECE_POOL

def play_game():
    env = BlockPuzzleEnv(render_mode="human")
    obs, info = env.reset()
    
    done = False
    total_reward = 0
    
    while not done:
        env.render()
        
        print("Доступные фигуры (индексы 0, 1, 2):")
        for i, idx in enumerate(env.current_pieces):
            print(f"{i}: \n{PIECE_POOL[idx]}")
            
        try:
            user_input = input("Введите: <номер_фигуры> <x> <y> (или 'q' для выхода): ")
            if user_input == 'q':
                break
                
            parts = user_input.split()
            if len(parts) != 3:
                print("Ошибка ввода! Пример: 0 3 3")
                continue
                
            p_idx, x, y = map(int, parts)
            
            # Кодируем действие в формат среды
            action = p_idx * 64 + y * 8 + x
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                print("GAME OVER!")
                print(f"Total Reward: {total_reward}")
                env.render()
                break
                
        except ValueError:
            print("Некорректный ввод.")
        except Exception as e:
            print(f"Ошибка: {e}")

if __name__ == "__main__":
    play_game()