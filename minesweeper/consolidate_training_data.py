import os
import numpy as np
from tqdm import tqdm
import time

training_data_dir = os.path.join(os.path.dirname(__file__), 'training_data')

def consolidate_training_data(output_file='consolidated_training_data.npz', max_games=None):
    """Consolidate all training data into a single efficient file."""
    
    # List all game directories
    game_dirs = [d for d in os.listdir(training_data_dir) 
                 if d.startswith('game_') and os.path.isdir(os.path.join(training_data_dir, d))]
    game_dirs.sort(key=lambda x: int(x.split('_')[1]))
    
    if max_games:
        game_dirs = game_dirs[:max_games]
    
    print(f"Found {len(game_dirs)} game directories to process")
    
    # Pre-allocate lists for better performance
    all_boards = []
    all_safe_moves = []
    
    start_time = time.time()
    
    # Process each game directory
    for game_dir in tqdm(game_dirs, desc="Processing games"):
        game_path = os.path.join(training_data_dir, game_dir)
        
        # Get all .npz files in the directory
        npz_files = [f for f in os.listdir(game_path) if f.endswith('.npz')]
        
        # Load each move
        for npz_file in npz_files:
            file_path = os.path.join(game_path, npz_file)
            data = np.load(file_path)
            all_boards.append(data['board'].flatten())
            all_safe_moves.append(data['safe_moves'].flatten())
    
    # Convert to numpy arrays
    boards_array = np.array(all_boards, dtype=np.float32)
    safe_moves_array = np.array(all_safe_moves, dtype=np.float32)
    
    print(f"\nLoaded {len(all_boards)} training examples in {time.time() - start_time:.1f} seconds")
    print(f"Boards shape: {boards_array.shape}")
    print(f"Safe moves shape: {safe_moves_array.shape}")
    print(f"Memory usage: {(boards_array.nbytes + safe_moves_array.nbytes) / 1024**2:.1f} MB")
    
    # Save consolidated data
    output_path = os.path.join(os.path.dirname(__file__), output_file)
    np.savez_compressed(output_path, boards=boards_array, safe_moves=safe_moves_array)
    print(f"\nSaved consolidated data to {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024**2:.1f} MB")

if __name__ == "__main__":
    # Consolidate first 100,000 games (or all available)
    consolidate_training_data(max_games=100000)