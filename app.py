import os

from flask import Flask
from dotenv import load_dotenv
from api.image.colorize import colorize_bp
from api.image.inpaint import inpaint_bp
from api.image.colorize_inpaint import colorize_inpaint_bp
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

app.register_blueprint(colorize_bp, url_prefix='/image')
app.register_blueprint(inpaint_bp, url_prefix='/image')
app.register_blueprint(colorize_inpaint_bp, url_prefix='/image')

load_dotenv()
is_dev_mode = os.environ['is_dev_mode']
debug = is_dev_mode == 'True'

if __name__ == '__main__':
    app.run(debug=debug)
