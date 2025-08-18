import numpy as np
from .minesweeper import Minesweeper, GameState, CellState
from .print_board import print_board
from .model import load_latest_model
from .play_game import decide_next_move_with_model

def main():
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
        print(f"\nTurn {turn}:")

        row, col, state = decide_next_move_with_model(game, model)
        if state == CellState.REVEALED:
            game.reveal(row, col)
        else:
            game.flag(row, col)

        print_board(game)




if __name__ == "__main__":
    main()