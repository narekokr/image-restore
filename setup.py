import os
import gdown


def download_file_from_google_drive(file_id, destination):
    url = 'https://drive.google.com/uc?id=' + file_id
    gdown.download(id=file_id, output=destination, quiet=False)


network = "1I8IQLN87ehFTXExXPNI9tywgX5LWD5Ck"
network_destination = "checkpoints/network.pt"
if not os.path.isfile(network_destination):
    download_file_from_google_drive(network, network_destination)

network = "1y4uKqFdxirgKRlc8BTPBBP6-lriu9gqj"
network_destination = "checkpoints/checkpoint.pt"
if not os.path.isfile(network_destination):
    download_file_from_google_drive(network, network_destination)

network = "1f7RMW1awbwgUjkM0tJ_Q8zV3z5lgfPQU"
network_destination = "checkpoints/siggraph.pt"
if not os.path.isfile(network_destination):
    download_file_from_google_drive(network, network_destination)
