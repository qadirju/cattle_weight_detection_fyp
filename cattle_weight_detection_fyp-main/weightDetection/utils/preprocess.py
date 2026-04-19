from torchvision import transforms
from config import IMAGE_SIZE


# Define image transformation pipeline
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet normalization
        std=[0.229, 0.224, 0.225]
    )
])


def get_transform():
    """
    Get the preprocessing transform pipeline.
    
    Returns:
        transforms.Compose: Composed image transformations
    """
    return transform
