"""Detects if an input image is AI generated.

Example usage as a script:

  python -m models.ai_recognizer  "/Users/amenabshir/Downloads/vision_board/GCfSGP_aoAA10p6.jpeg"

  python -m models.ai_recognizer  "/Users/amenabshir/Downloads/vision_board/GCfSGP_aoAA10p6.jpeg"
"""
import argparse
from pathlib import Path
from typing import Union

from models import util
from PIL import Image
import torch
from torchvision import transforms

STAGED_MODEL_DIRNAME = Path(__file__).resolve().parent / "artifacts" / "ResNet"
MODEL_FILE = "model.pt"


class AiImageRecognizer:
    """Recognizes if an image is AI generated or not."""

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = STAGED_MODEL_DIRNAME / MODEL_FILE
        self.model = torch.jit.load(model_path)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.ToTensor(),
                transforms.Lambda(AiImageRecognizer._orderTensor),
            ]
        )

    @torch.no_grad()
    def predict(self, image: Union[str, Path, Image.Image]) -> dict:
        image_pil = image
        if not isinstance(image, Image.Image):
            image_pil = util.read_image_pil(image)
        transformed_image = self.transform(image_pil)
        transformed_image = transformed_image.unsqueeze(0)
        pred = self.model(transformed_image).item()

        return {"AI 🤖": pred, "Human 👤": 1 - pred}

    @staticmethod
    def _orderTensor(x):
        x = x.to(torch.float32)
        return x


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "filename",
        type=str,
        help="Name for an image file. This can be a local path, a URL, a URI from AWS/GCP/Azure storage, an HDFS path, or any other resource locator supported by the smart_open library.",
    )
    args = parser.parse_args()

    ai_recognizer = AiImageRecognizer()
    pred_str = ai_recognizer.predict(args.filename)
    print(pred_str)


if __name__ == "__main__":
    main()
