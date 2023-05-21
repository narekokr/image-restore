from flask import Blueprint, request, send_file
import cv2
from io import BytesIO
from models.scratch_detection import ImageProcessor

inpaint_bp = Blueprint('inpaint', __name__)
image_processor = ImageProcessor()

@inpaint_bp.route('/inpaint', methods=['POST'])
def inpaint():
    image = request.files['image']
    mask = request.files['mask']
    detect_automatically = request.args.get('detect_automatically', default='true')
    detect_automatically = not mask or detect_automatically.lower() == 'true'

    result = image_processor.get_inpainted_image(image, mask, detect_automatically)
    result_bytes = cv2.imencode('.png', result)[1].tobytes()

    return send_file(BytesIO(result_bytes), mimetype='image/png')

