from ale_py import ALEInterface, roms
import numpy as np

class Breakout:
    def __init__(self, seed: int):
        ale = ALEInterface()
        rom_path = str(roms.get_rom_path("breakout"))
        ale.loadROM(rom_path)
        ale.reset_game()

        self.ale = ale

        self.rng = np.random.RandomState(seed)  # Fixed seed for reproducibility

        self.current_screen = None
        self.current_model_input = None
        self.prev_screen = np.zeros((210, 160, 3), dtype=np.uint8)  # Initial previous screen state

    def decide_next_move_from_prediction(self, actions: np.ndarray) -> int:
        # For Breakout, we can directly use the action predictions
        # Assuming actions is a list of actions corresponding to the minimal action set
        action = int(np.argmax(actions))
        return action

    def decide_next_move_with_rng(self, rng) -> int:
        # For Breakout, we can randomly select an action from the minimal action set
        actions = self.ale.getMinimalActionSet()
        action = self.rng.randint(0, len(actions))
        return action

    def take_action(self, action: int) -> float:
        reward = float(self.ale.act(action))
        self.prev_screen = self.get_current_screen()  # Update previous screen state
        self.current_screen = None  # Reset current screen to force a refresh next time
        self.current_model_input = None  # Reset model input to force a refresh next time
        return reward
    
    def isGameOver(self) -> bool:
        return self.ale.game_over()
    
    def get_current_screen(self) -> np.ndarray:
        if self.current_screen is None:
            self.current_screen = self.ale.getScreenRGB()
        return self.current_screen

    def getModelInput(self) -> np.ndarray:
        if self.current_model_input is None:
            screen = self.get_current_screen()
            # Use the current screen as the first 3 channels, and the previous screen as the last 3 channels
            self.current_model_input = np.concatenate([screen, self.prev_screen], axis=-1)
        return self.current_model_input

def produce_model_predictions_batch(games: list[Breakout], model) -> np.ndarray:
    inputs = np.array([game.getModelInput() for game in games])
    return model.predict(inputs, verbose=0)