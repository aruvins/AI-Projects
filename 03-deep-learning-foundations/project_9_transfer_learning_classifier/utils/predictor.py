import torch
from PIL import Image
import torchvision.transforms as transforms


def predict_image(model, image_path, classes, device):
    """
    Predict a single image.
    """

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(image_tensor)
        prediction = outputs.argmax(dim=1).item()
        
    return classes[prediction]