# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is an experimental machine learning repository focused on applying reinforcement learning to classic games. The projects use TensorFlow 2.15.0 with Metal acceleration on macOS.

## Key Commands

### Running Projects

**Minesweeper:**
```bash
# Play a trained model
python -m minesweeper.play_game

# Train a new model
python -m minesweeper.rl

# Watch the model play games
python -m minesweeper.watch_game

# Generate training data
python -m minesweeper.generate_training_data --num_batches 10

# Profile performance
python -m minesweeper.profile
```

**Breakout:**
```bash
# Train the model
python -m breakout.rl

# Test/visualize the model
python -m breakout.test
```

### Environment Setup
```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies (see INSTALL.md for TensorFlow Metal setup)
pip install tensorflow-macos==2.15.0
pip install tensorflow-metal
pip install ale-py pygame
```

## Architecture Overview

### Project Structure
- `/minesweeper/` - Reinforcement learning for Minesweeper game
  - `minesweeper.py` - Core game logic and board representation
  - `model.py` - CNN architecture definitions
  - `rl.py` - Reinforcement learning training loop
  - `/models/` - Saved model checkpoints
  - `/training_data/` - Pre-generated training datasets

- `/breakout/` - Reinforcement learning for Atari Breakout
  - `game.py` - ALE wrapper for Breakout game
  - `model.py` - CNN model for Breakout
  - `rl.py` - RL training implementation
  - `test.py` - Pygame-based visualization

### Key Design Patterns

1. **Model Architecture**: Both projects use convolutional neural networks with varying depths. Models typically follow a pattern of Conv2D layers with batch normalization and dropout for regularization.

2. **Training Pipeline**: 
   - Pre-training phase using generated/collected data
   - Reinforcement learning phase with exploration/exploitation balance
   - Baseline comparison for performance evaluation

3. **Game State Representation**:
   - Minesweeper: 3D tensor (9x9xC) where C channels encode cell states
   - Breakout: Direct frame data from ALE with preprocessing

4. **RL Approach**: Custom Q-learning-like implementations with:
   - Exploration using epsilon-greedy or top-k sampling
   - Experience replay for training stability
   - Win/loss tracking against baseline models

### Important Notes

- No formal testing framework or linting setup exists
- Models are saved incrementally during training (e.g., `model_v1.h5`, `model_v2.h5`)
- The repository follows an experimental approach with detailed progress tracking in ONGOING_NOTES.md
- TensorFlow Metal acceleration is used for GPU support on macOS