import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path


def load_model(model_path="best_model.pth"):
    """
    Load the trained EfficientNet B0 model.
    
    Args:
        model_path (str): Path to the saved model checkpoint
        
    Returns:
        torch.nn.Module: Loaded model in evaluation mode
    """
    # Initialize model architecture
    model = models.efficientnet_b0(pretrained=False)
    
    # Replace classifier for regression task
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(1280, 1)
    )
    
    # Load checkpoint
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location="cpu")
    
    # Load state dict (handle both direct state dict and wrapped checkpoint)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
    
    # Set to evaluation mode
    model.eval()
    
    return model
