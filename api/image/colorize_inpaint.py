from io import BytesIO

import numpy as np
from flask import Blueprint, request, send_file
import cv2

from models.scratch_detection import ImageProcessor

colorize_inpaint_bp = Blueprint('colorize_inpaint', __name__)
image_processor = ImageProcessor()

@colorize_inpaint_bp.route('/colorize-and-inpaint', methods=['POST'])
def colorize_and_inpaint():
    image = request.files['image']
    try:
        mask = request.files['mask']
    except:
        mask = False
    detect_automatically = request.args.get('detect_automatically', default='true')
    detect_automatically = not mask or detect_automatically.lower() == 'true'

    result = image_processor.get_inpainted_image(image, mask, detect_automatically)
    result = image_processor.colorize(result)
    out = cv2.cvtColor((result * 255).astype(np.float32), cv2.COLOR_BGR2RGB)
    result_bytes = cv2.imencode('.png', out)[1].tobytes()

    return send_file(BytesIO(result_bytes), mimetype='image/png')
