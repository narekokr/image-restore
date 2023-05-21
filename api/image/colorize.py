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
    image = request.files['image']

    img_array = np.fromstring(image.read(), np.uint8)

    # Decode the image array using OpenCV
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    result = image_processor.colorize(img)

    # Convert the result image to bytes
    result_bytes = cv2.imencode('.png', result)[1].tobytes()
    return send_file(BytesIO(result_bytes), mimetype='image/png')
