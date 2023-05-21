from io import BytesIO
from flask import Blueprint, request, send_file
import cv2

from models.scratch_detection import ImageProcessor

colorize_inpaint_bp = Blueprint('colorize_inpaint', __name__)
image_processor = ImageProcessor()

@colorize_inpaint_bp.route('/colorize-and-inpaint', methods=['POST'])
def colorize_and_inpaint():
    image = request.files['image']
    mask = request.files['mask']
    detect_automatically = request.args.get('detect_automatically', default='true')
    detect_automatically = not mask or detect_automatically.lower() == 'true'
    inpainted = image_processor.get_inpainted_image(image, mask, detect_automatically)
    result_bytes = cv2.imencode('.png', inpainted)[1].tobytes()

    return send_file(BytesIO(result_bytes), mimetype='image/png')
