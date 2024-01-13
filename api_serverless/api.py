"""AWS Lambda function serving ai_recognizer predictions."""
import json

from models.ai_recognizer import AiImageRecognizer
from models.ai_recognizer import util

model = AiImageRecognizer()


def handler(event, _context):
    """Provide main prediction API."""
    print("INFO loading image")
    image = _load_image(event)
    if image is None:
        return {"statusCode": 400, "message": "neither image_url nor image found in event"}
    print("INFO image loaded")
    print("INFO starting inference")
    pred = model.predict(image)
    print("INFO inference complete")
    
    return pred


def _load_image(event):
    event = _from_string(event)
    event = _from_string(event.get("body", event))
    image_url = event.get("image_url")
    if image_url is not None:
        print("INFO url {}".format(image_url))
        return util.read_image_pil(image_url, grayscale=True)
    else:
        image = event.get("image")
        if image is not None:
            print("INFO reading image from event")
            return util.read_b64_image(image, grayscale=True)
        else:
            return None


def _from_string(event):
    if isinstance(event, str):
        return json.loads(event)
    else:
        return event
