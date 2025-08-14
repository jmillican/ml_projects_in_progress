from .minesweeper import Minesweeper

def print_board(game: Minesweeper):
    """Print the current board state."""
    visible = game.get_visible_board()
    
    print("\n  ", end="")
    for col in range(game.cols):
        print(f"{col:2}", end="")
    print()
    
    for row in range(game.rows):
        print(f"{row:2}", end=" ")
        for col in range(game.cols):
            cell = visible[row, col]
            if cell == -2:
                print("🚩", end="")
            elif cell == -1:
                print("□ ", end="")
            elif cell == 0:
                print("  ", end="")
            elif cell == 9:
                print("💣", end="")
            else:
                print(f"{cell} ", end="")
        print()
    print(f"\nMines remaining: {game.get_remaining_mines()}")
    print(f"Game state: {game.get_game_state().name}")