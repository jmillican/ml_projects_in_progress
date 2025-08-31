import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

models_dir = os.path.join(os.path.dirname(__file__), 'models')

def loss_function(y_true, y_pred):
    """
    Custom loss function that masks the loss for cells that are not visible.
    """
    mask = (y_true != 0).float()  # Mask where y_true is not zero
    masked_loss = torch.mean(torch.square(y_true - y_pred) * mask)
    return masked_loss

class MinesweeperModel(nn.Module):
    def __init__(self):
        super(MinesweeperModel, self).__init__()
        self.conv0 = nn.Conv2d(6, 32, kernel_size=3, padding=1)  # 6 input channels
        self.conv1 = nn.Conv2d(32, 128, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.output_layer = nn.Conv2d(32, 2, kernel_size=1, padding=0)
        
        # # Initialize weights using Xavier/Glorot initialization (like TensorFlow's default)
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.xavier_uniform_(m.weight)
        #         if m.bias is not None:
        #             nn.init.zeros_(m.bias)
        
    def forward(self, x):
        x = torch.relu(self.conv0(x))
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.output_layer(x)  # No activation on output (linear)
        return x

def create_model(input_shape: tuple[int, ...], output_shape: tuple[int, ...]) -> nn.Module:
    torch.manual_seed(12345)
    model = MinesweeperModel()
    return model

def getModelPath(model_name: str) -> str:
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    return os.path.join(models_dir, f"{model_name}.h5")

def save_model(model: nn.Module, model_name: str, optimizer=None, epoch=None):
    model_path = getModelPath(model_name)
    # Change extension from .h5 to .pt
    model_path = model_path.replace('.h5', '.pt')
    
    save_dict = {'model_state_dict': model.state_dict()}
    if optimizer is not None:
        save_dict['optimizer_state_dict'] = optimizer.state_dict()
    if epoch is not None:
        save_dict['epoch'] = epoch
        
    torch.save(save_dict, model_path)
    print(f"Model saved to {model_path}")

def load_model(model_name: str) -> nn.Module:
    model_path = getModelPath(model_name)
    # Try both .pt and .h5 extensions for backward compatibility
    if not os.path.exists(model_path):
        model_path = model_path.replace('.h5', '.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model {model_name} does not exist at {model_path}.")
    
    model = MinesweeperModel()
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def load_latest_model(offset: int = 0, verbose: bool = True) -> tuple[nn.Module, str]:
    # List all model files in the models directory
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5') or f.endswith('.pt')]
    if not model_files:
        raise FileNotFoundError("No model files found in 'models' directory.")
    
    # Sort files by modification time
    model_files.sort(key=lambda f: os.path.getmtime(os.path.join(models_dir, f)))
    latest_model_file = model_files[-(1 + offset)]  # Get the most recent model file, offset by the parameter
    model_name = os.path.splitext(latest_model_file)[0]  # Remove the extension
    if verbose:
        print(f"Loading model: {model_name}")
    # Load the model
    model = load_model(model_name)
    return model, model_name