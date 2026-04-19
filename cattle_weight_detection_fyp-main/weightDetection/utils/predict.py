import torch
from config import MEAN_WEIGHT, STD_WEIGHT


def predict_weight(model, image_tensor):
    """
    Predict cattle weight from image tensor.
    
    The model outputs normalized weight, which is then denormalized
    using the training statistics (mean and std).
    
    Args:
        model (torch.nn.Module): Loaded prediction model
        image_tensor (torch.Tensor): Preprocessed image tensor (batch size must be 1)
        
    Returns:
        float: Predicted weight in kg (denormalized)
    """
    with torch.no_grad():
        # Get normalized prediction
        normalized_pred = model(image_tensor).item()
    
    # Denormalize: real_weight = normalized_pred * STD + MEAN
    predicted_weight = normalized_pred * STD_WEIGHT + MEAN_WEIGHT
    
    return predicted_weight


def batch_predict(model, image_tensors):
    """
    Predict weights for a batch of images.
    
    Args:
        model (torch.nn.Module): Loaded prediction model
        image_tensors (torch.Tensor): Batch of preprocessed image tensors
        
    Returns:
        list: List of predicted weights in kg (denormalized)
    """
    with torch.no_grad():
        normalized_preds = model(image_tensors).squeeze().cpu().numpy()
    
    # Denormalize all predictions
    predicted_weights = normalized_preds * STD_WEIGHT + MEAN_WEIGHT
    
    return predicted_weights.tolist() if isinstance(predicted_weights, list) else [predicted_weights]
