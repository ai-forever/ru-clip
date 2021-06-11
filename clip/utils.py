import wget
import os
import shutil
import torch


MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/"
            "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/"
             "8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/"
              "7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/"
                "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
}


def download_image_encoder(save_dir, name):
    bn = os.path.basename(MODELS[name])
    path = os.path.join(save_dir, bn)
    if not os.path.exists(path):
        if os.path.exists(bn):
            os.remove(bn)
        print("Start downloading of origin pretrained model...")
        wget.download(MODELS[name])
        print("Model has been downloaded.")
        shutil.move(path, bn)
    else:
        print("Model has been already downloaded.")
    return path


def get_origin_image_encoder(save_dir, name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    return torch.load(download_image_encoder(save_dir, name)).eval().visual
