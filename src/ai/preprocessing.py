from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
PREPROCESS = "retina-border-crop-rgb-imagenet-v1"

def crop_retina(image):
    image = image.convert('RGB')
    a = np.asarray(image)
    mask = a.max(axis=2) > 12
    ys, xs = np.where(mask)
    if len(xs) and xs.max()-xs.min() > image.width//3 and ys.max()-ys.min() > image.height//3:
        image = image.crop((int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)))
    # Pad rather than stretch the retina.
    side = max(image.size)
    square = Image.new('RGB', (side, side))
    square.paste(image, ((side-image.width)//2, (side-image.height)//2))
    return square

