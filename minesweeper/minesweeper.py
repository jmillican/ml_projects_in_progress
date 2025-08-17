import numpy as np
from typing import Tuple, List, Optional
from enum import Enum

class CellState(Enum):
    HIDDEN = 0
    REVEALED = 1
    FLAGGED = 2


class GameState(Enum):
    PLAYING = 0
    WON = 1
    LOST = 2

BOARD_SIZE = 9

class Minesweeper:
    def __init__(self, rows: int = BOARD_SIZE, cols: int = BOARD_SIZE, mines: int = 10, seed: Optional[int] = None):
        """
        Initialize a new Minesweeper game.
        
        Args:
            rows: Number of rows in the board
            cols: Number of columns in the board
            mines: Number of mines to place
            seed: Random seed for deterministic board generation
        """
        self.rows = rows
        self.cols = cols
        self.num_mines = mines
        self.seed = seed
        self.game_state = GameState.PLAYING
        self.first_move = True
        self.num_moves = 0
        self.valid_moves_mask = np.full((rows, cols, 2), 1, dtype=np.float16)
        self.memoized_input_board = None

        # Initialize the random number generator with the seed
        self.rng = np.random.RandomState(seed)
        
        # Initialize boards
        self.mine_board = np.zeros((rows, cols), dtype=bool)
        self.cell_states = np.full((rows, cols), CellState.HIDDEN.value)
        self.adjacent_mines = np.zeros((rows, cols), dtype=int)
        
        # Track revealed cells count for win condition
        self.revealed_count = 0
        self.cells_to_win = rows * cols - mines
        
    def _place_mines(self, avoid_row: int, avoid_col: int):
        """
        Place mines randomly on the board, avoiding the first clicked cell.
        
        Args:
            avoid_row: Row to avoid placing mine
            avoid_col: Column to avoid placing mine
        """
        # Create list of all possible mine positions
        positions = [(r, c) for r in range(self.rows) for c in range(self.cols)]
        
        # Remove the position to avoid
        positions.remove((avoid_row, avoid_col))
        
        # Randomly select mine positions
        mine_positions = self.rng.choice(len(positions), self.num_mines, replace=False)
        placed_positions = [positions[idx] for idx in mine_positions]
        
        # Place mines
        for idx in mine_positions:
            row, col = positions[idx]
            self.mine_board[row, col] = True

        # Calculate adjacent mine counts
        self._calculate_adjacent_mines(placed_positions)
        
    def _calculate_adjacent_mines(self, mine_positions: list[tuple[int, int]]):
        for row, col in mine_positions:
            for r, c in self._get_neighbors(row, col):
                if not self.mine_board[r, c]:
                    self.adjacent_mines[r, c] += 1
                    
    def _get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get all valid neighboring cells."""
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                r, c = row + dr, col + dc
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    neighbors.append((r, c))
        return neighbors
        
    def reveal(self, row: int, col: int) -> bool:
        """
        Reveal a cell. Returns True if move was successful, False if game over.
        
        Args:
            row: Row index
            col: Column index
            
        Returns:
            True if move was successful, False if hit a mine
        """
        if self.game_state != GameState.PLAYING:
            return False
            
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(f"Invalid position: ({row}, {col})")
            
        if self.cell_states[row, col] != CellState.HIDDEN.value:
            raise ValueError(f"Cell ({row}, {col}) is already revealed or flagged, with value {self.cell_states[row, col]}.")

        self.num_moves += 1
        self.clear_memoized_input_board()

        # Place mines on first move
        if self.first_move:
            self._place_mines(row, col)
            self.first_move = False

        # Check if hit a mine
        if self.mine_board[row, col]:
            self.cell_states[row, col] = CellState.REVEALED.value
            self.game_state = GameState.LOST
            return False

        # Reveal the cell
        self._reveal_cell(row, col)

        # Check win condition
        if self.revealed_count == self.cells_to_win:
            self.game_state = GameState.WON

        return True
        
    def _reveal_cell(self, row: int, col: int):
        """Reveal a cell and potentially its neighbors (flood fill)."""
        if self.cell_states[row, col] != CellState.HIDDEN.value:
            return
            
        self.cell_states[row, col] = CellState.REVEALED.value
        self.valid_moves_mask[row, col] = [0, 0]
        self.revealed_count += 1
        
        # If cell has no adjacent mines, reveal all neighbors
        if self.adjacent_mines[row, col] == 0:
            for r, c in self._get_neighbors(row, col):
                self._reveal_cell(r, c)
                self.valid_moves_mask[r, c] = [0, 0]
                
    def flag(self, row: int, col: int):
        """
        Toggle flag on a cell.
        
        Args:
            row: Row index
            col: Column index
        """
        if self.game_state != GameState.PLAYING:
            raise ValueError("Cannot flag cells when game is not in progress.")
            
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(f"Invalid position: ({row}, {col})")
        
        if self.cell_states[row, col] == CellState.REVEALED.value:
            raise ValueError(f"Cannot flag revealed cell ({row}, {col}).")
        

        if self.cell_states[row, col] == CellState.FLAGGED.value:
            raise ValueError(f"Cell ({row}, {col}) is already flagged.")

        # Place mines on first move
        if self.first_move:
            self._place_mines(row, col)
            self.first_move = False
        
        # Flagging a safe square loses the game.
        if not self.mine_board[row, col]:
            self.game_state = GameState.LOST
            return

        self.cell_states[row, col] = CellState.FLAGGED.value
        self.valid_moves_mask[row, col] = [0, 0]

        self.num_moves += 1
        self.clear_memoized_input_board()
            
    def get_visible_board(self) -> np.ndarray:
        """
        Get the current visible state of the board.
        
        Returns:
            2D array where:
            - -2: flagged cell
            - -1: hidden cell
            - 0-8: revealed cell with number of adjacent mines
            - 9: revealed mine (game over)
        """
        visible = np.full((self.rows, self.cols), -1)
        
        for row in range(self.rows):
            for col in range(self.cols):
                if self.cell_states[row, col] == CellState.FLAGGED.value:
                    visible[row, col] = -2
                elif self.cell_states[row, col] == CellState.REVEALED.value:
                    if self.mine_board[row, col]:
                        visible[row, col] = 9
                    else:
                        visible[row, col] = self.adjacent_mines[row, col]
                        
        return visible

    def get_input_board(self) -> np.ndarray:
        """
        Get the input board for the model.
        
        Returns:
            3D array. Treated as a 2D board with the first three parameters of each row, column cell represented as:
            - [0, 0, 0, ]: hidden cell.
            - [1, 0-8, 0, ]: revealed cell with number of adjacent mines.
            - [0, 0, 1, ]: flagged cell.
            - [1, 0, 2, ]: revealed mine - although this should never actually happen in the model.

            The remaining two parameters of each cell represent the global state of the game, with the first
            representing the number of remaining mines, and the second representing the number of un-revealed cells.
        """
        if self.memoized_input_board is not None:
            return self.memoized_input_board
        
        input_board = np.full((self.rows, self.cols, 5), 0, dtype=np.float16)

        input_board[:, :, 0] = np.float16(self.cell_states == CellState.REVEALED.value)
        input_board[:, :, 1] = np.float16(np.where(self.cell_states == CellState.REVEALED.value, self.adjacent_mines / 1.5, 0))

        input_board[:, :, 2] = np.float16(self.cell_states == CellState.FLAGGED.value)
        input_board[:, :, 2] = np.where(self.cell_states == CellState.REVEALED.value, np.where(self.mine_board, 2, input_board[:, :, 2]), input_board[:, :, 2])

        remaining_mines = self.get_remaining_mines()
        remaining_cells = np.sum(self.cell_states == CellState.HIDDEN.value)
        input_board[:, :, 3] = np.float16(remaining_mines) / 10.0
        input_board[:, :, 4] = np.float16(remaining_cells) / 81.0

        self.memoized_input_board = input_board
        return input_board

    def get_game_state(self) -> GameState:
        """Get the current game state."""
        return self.game_state
        
    def get_flag_count(self) -> int:
        """Get the number of flags placed."""
        return np.sum(self.cell_states == CellState.FLAGGED.value)
        
    def get_remaining_mines(self) -> int:
        """Get the estimated number of remaining mines (total mines - flags)."""
        return self.num_mines - self.get_flag_count()
    
    def clear_memoized_input_board(self):
        """Clear the memoized input board to force recalculation."""
        self.memoized_input_board = None