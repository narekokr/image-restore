# Image Restoration and Colorization Backend

This repository contains the backend implementation for image restoration and colorization networks using Flask. The backend provides endpoints to send images for restoration and colorization and returns the processed images.

## Getting Started

To set up and run the backend locally, follow these steps:

### Prerequisites

- Python 3.x

### Installation

1. Clone the repository:
```
git clone https://github.com/narekokr/image-restore-back.git
```

2. Navigate to the project directory:
```shell
cd image-restore-back
```

3. Install the required dependencies:
```shell
pip install -r requirements.txt
```

4. Download the networks
```shell
python3 setup.py
```

5. Copy the `.env.example` file to `.env`
```shell
cp .env.example .env
```

### Usage

1. Start the Flask server:
```shell
python3 app.py
```


2. The server should be up and running on `http://localhost:5000`.

### Endpoints

- **POST** `/image/inpaint`: Endpoint to send an image for inpainting. Requires a multipart/form-data POST request with the `image` parameter containing the image file, and `mask` parameter containing the maks for inpainting. Can also set `detect_automatically=true` in params to detect the scratched regions automatically, in which case the `mask` parameter is not required

- **POST** `/image/colorize`: Endpoint to send a black and white image for colorization. Requires a multipart/form-data POST request with the `image` parameter containing the image file.

- **POST** `/image/colorize-and-inpaint` : Endpoint that will combine above two endpoints in one, to inpaint and colorize the image 

The processed images will be returned as a response in the appropriate format.

### Examples

#### Restore Image

```bash
curl --location 'localhost:5000/image/inpaint?detect_automatically=true' \
--form 'image=@"path/to/image"'