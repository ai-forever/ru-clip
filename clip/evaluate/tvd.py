from torchvision.datasets import CIFAR100
from torchvision.datasets import CIFAR10
from .utils import prepare_classes, get_tokenizer, get_text_batch, get_image_batch
import os
import json
from tqdm import tqdm
import torch


DATASETS = {
    "CIFAR100": CIFAR100,
    "CIFAR10": CIFAR10
}


def get_text_probs_from_dataset(
        model, args,
        tokenizer=None,
        name=None,
        ds=None,
        train=False,
        classes_path=None,
        text_descriptions=None
):
    if ds is None:
        cls = DATASETS[name]
        ds = cls(os.path.expanduser("~/.cache"), download=True, train=train)
        if classes_path is None:
            classes_path = f"../clip/evaluate/{name.lower()}/{name.lower()}classes.json"
        with open(classes_path, "r") as file:
            ds.classes = json.load(file)
    if text_descriptions is None:
        text_descriptions = prepare_classes(ds.classes)
    if tokenizer is None:
        tokenizer = get_tokenizer()
    input_ids, attention_mask = get_text_batch(text_descriptions, tokenizer, args)
    probs = []
    labels = []
    for x in tqdm(ds, total=len(ds)):
        with torch.no_grad():
            logits_per_image, logits_per_text = model(
                img_input={"x": get_image_batch([x[0]], args.img_transform, args)},
                text_input={"x": input_ids, "attention_mask": attention_mask}
            )
        text_probs = logits_per_image.softmax(dim=-1).cpu()
        labels.append(x[1])
        probs.append(text_probs.tolist()[0])
    return probs, labels
