from ale_py import ALEInterface, roms
import pygame
import random
pygame.init()

ale = ALEInterface()
rom_path = str(roms.get_rom_path("breakout"))
ale.loadROM(rom_path)
ale.reset_game()

dims = ale.getScreenDims()

print(ale.getMinimalActionSet())

# Print available methods
print("Available ALE methods:")
print([method for method in dir(ale) if not method.startswith('_')])

# Run the game, rendering the screen after each action using pygame, at a 25 fps rate.
total_reward = 0
while not ale.game_over():
    action = random.choice(ale.getMinimalActionSet())  # Choose the first action (e.g., NOOP)
    reward = ale.act(action)
    print(reward)
    total_reward += reward

    screen = ale.getScreenRGB()  # Get the current screen state
    print(screen.shape)  # Print the shape of the screen

    # Render the screen using pygame
    screen_size = (dims[1], dims[0])
    pygame_screen = pygame.display.set_mode(screen_size)
    
    # Convert from (height, width, 3) to (width, height, 3) and then to pygame surface
    screen_transposed = screen.transpose((1, 0, 2))
    pygame.surfarray.blit_array(pygame_screen, screen_transposed)
    pygame.display.flip()
    pygame.time.delay(40)  # Delay to achieve approximately 25 fps (1000/25 = 40)

    # Check what the score of the game is
    if ale.game_over():
        print(f"Game Over! Final Score: {total_reward}")
    elif reward > 0:
        print(f"Score! Current Total: {total_reward}")