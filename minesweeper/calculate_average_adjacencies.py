from .minesweeper import Minesweeper, GameState, CellState, BOARD_SIZE

adjacencies_sum = 0
cells_checked = 0

for i in range(5000):
    game = Minesweeper(rows=BOARD_SIZE, cols=BOARD_SIZE, mines=10, seed=i)
    game.reveal(0, 0)

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if game.mine_board[row, col]:
                continue
            if game.adjacent_mines[row, col] == 0:
                continue
            adjacencies_sum += game.adjacent_mines[row, col]
            cells_checked += 1

if cells_checked > 0:
    average_adjacency = adjacencies_sum / cells_checked
    print(f"Average number of adjacent mines: {average_adjacency}")
else:
    print("No cells were checked.")