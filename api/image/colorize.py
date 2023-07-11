import base64
import cv2
import matplotlib.pyplot as plt
import numpy as np
from flask import Blueprint, request, send_file
from PIL import Image
from io import BytesIO

from models.scratch_detection import ImageProcessor

colorize_bp = Blueprint('colorize', __name__)
image_processor = ImageProcessor()

@colorize_bp.route('/colorize', methods=['POST'])
def colorize():
    base64_string = request.json['image']  # Assuming the base64 string is sent as JSON in the request body
    img_b64dec = base64.b64decode(base64_string)
    img_byteIO = BytesIO(img_b64dec)
    image = Image.open(img_byteIO).convert('RGB')
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    result = image_processor.colorize(img)
    out = cv2.cvtColor((result * 255).astype(np.float32), cv2.COLOR_BGR2RGB)

    # Convert the result image to bytes
    result_bytes = cv2.imencode('.png', out)[1].tobytes()
    return send_file(BytesIO(result_bytes), mimetype='image/png')
