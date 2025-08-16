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
        
        # Place mines
        for idx in mine_positions:
            row, col = positions[idx]
            self.mine_board[row, col] = True
            
        # Calculate adjacent mine counts
        self._calculate_adjacent_mines()
        
    def _calculate_adjacent_mines(self):
        """Calculate the number of adjacent mines for each cell."""
        for row in range(self.rows):
            for col in range(self.cols):
                if not self.mine_board[row, col]:
                    count = 0
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            r, c = row + dr, col + dc
                            if 0 <= r < self.rows and 0 <= c < self.cols and self.mine_board[r, c]:
                                count += 1
                    self.adjacent_mines[row, col] = count
                    
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
            raise ValueError(f"Cell ({row}, {col}) is already revealed or flagged.")

        self.num_moves += 1
            
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
        self.revealed_count += 1
        
        # If cell has no adjacent mines, reveal all neighbors
        if self.adjacent_mines[row, col] == 0:
            for r, c in self._get_neighbors(row, col):
                self._reveal_cell(r, c)
                
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
        
        # Flagging a safe square loses the game.
        if not self.mine_board[row, col]:
            self.game_state = GameState.LOST
            return

        self.cell_states[row, col] = CellState.FLAGGED.value
        self.num_moves += 1
            
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
        
    def get_game_state(self) -> GameState:
        """Get the current game state."""
        return self.game_state
        
    def get_flag_count(self) -> int:
        """Get the number of flags placed."""
        return np.sum(self.cell_states == CellState.FLAGGED.value)
        
    def get_remaining_mines(self) -> int:
        """Get the estimated number of remaining mines (total mines - flags)."""
        return self.num_mines - self.get_flag_count()