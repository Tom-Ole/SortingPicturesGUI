from typing import List
from PIL import ImageFile
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

class ImageClassifierAi:
    def __init__(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.model.eval()

        self.model.to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.486, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.labels = models.EfficientNet_B0_Weights.DEFAULT.meta["categories"]


    ''' 
    Classifies an image and returns a list of labels.
    Args:
        img (Image.Image): The image to classify.
    Returns:
        List[str]: A list of tags predicted by the model.
    '''
    def classify_image(self, path: str) -> List[str]:
        
        image = Image.open(path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)

        top_probs, top_idxs = torch.topk(probs, 3)

        tags = [self.labels[idx] for idx in top_idxs]

        return tags
    