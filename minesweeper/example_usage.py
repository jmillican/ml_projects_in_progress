from .minesweeper import Minesweeper, GameState

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


def main():
    # Example 1: Create a deterministic game with seed
    print("=== Example 1: Deterministic game with seed 42 ===")
    game1 = Minesweeper(rows=9, cols=9, mines=10, seed=42)
    
    # Make some moves
    game1.reveal(5, 4)  # Safe first move
    game1.reveal(2, 2)
    game1.flag(0, 0)

    print_board(game1)

    # Example 2: Same seed produces same board
    print("\n=== Example 2: Same seed (42) produces identical board ===")
    game2 = Minesweeper(rows=9, cols=9, mines=10, seed=42)
    game2.reveal(4, 4)  # Same first move
    game2.reveal(0, 0)
    game2.flag(1, 1)
    
    print_board(game2)
    
    # Example 3: Different seed produces different board
    print("\n=== Example 3: Different seed (123) produces different board ===")
    game3 = Minesweeper(rows=9, cols=9, mines=10, seed=123)
    game3.reveal(4, 4)
    
    print_board(game3)
    
    # Example 4: Playing until game over
    print("\n=== Example 4: Playing a small game ===")
    game4 = Minesweeper(rows=5, cols=5, mines=5, seed=999)
    
    # Make moves until we win or lose
    moves = [(2, 2), (0, 0), (4, 4), (0, 4), (4, 0)]
    
    for row, col in moves:
        if game4.get_game_state() == GameState.PLAYING:
            success = game4.reveal(row, col)
            print(f"\nRevealed ({row}, {col}): {'Success' if success else 'Hit mine!'}")
            print_board(game4)
            
            if not success:
                break


if __name__ == "__main__":
    main()