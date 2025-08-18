import numpy as np
from .minesweeper import Minesweeper, GameState, CellState
from .print_board import print_board
from .model import load_latest_model
from .play_game import decide_next_move_with_model

# Overriding the built-in print function to capture output
# so that we can only print losing games.
import builtins

print_buffer = []
print_cached = builtins.print

def print_replacement(*args, **kwargs):
    global print_buffer
    print_buffer.append((args, kwargs))

builtins.print = print_replacement

def clear_print_buffer():
    global print_buffer
    print_buffer.clear()

def flush_print_buffer():
    global print_buffer
    if print_buffer:
        for line in print_buffer:
            print_cached(*line[0], **line[1])
        clear_print_buffer()


def main():
    go = True
    attempt = 0
    while go:
        clear_print_buffer()
        attempt += 1
        print_cached(f"Attempt {attempt} to play a game...")

        game_seed = np.random.randint(2**32 - 1)
        print(f"Game Seed: {game_seed}")

        game = Minesweeper(rows=9, cols=9, mines=10, seed=game_seed)
        model, model_name = load_latest_model(offset=0)
        print(f"Loaded model: {model_name}")

        print("Initial Board:")
        print_board(game)

        turn = 0
        while game.get_game_state() == GameState.PLAYING:
            turn += 1
            print(f"\n\nTurn {turn}:")

            row, col, state = decide_next_move_with_model(game, model)
            if state == CellState.REVEALED:
                game.reveal(row, col)
            else:
                game.flag(row, col)

            print(f"Move: ({row}, {col}), State: {state.name}")
            print_board(game)
        go = (game.get_game_state() == GameState.LOST)
    flush_print_buffer()




if __name__ == "__main__":
    main()