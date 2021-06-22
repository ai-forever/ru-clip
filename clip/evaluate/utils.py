import torch
from transformers import GPT2Tokenizer
import json
from pymorphy2 import MorphAnalyzer
from clip.model import get_image_batch, GPT2Model, load, VisualEncoder, TextEncoder, CLIP
import os
import skimage
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from collections import Counter
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
from clip.download_utils import download_file_from_hf


MODELS = {
    "ViT-B/32-small": {
        "visual_encoder_name": "ViT-B/32",
        "load": "ViT-B/32",
        "load_huggingface": "sberbank-ai/rugpt3small_based_on_gpt2",
        "visual_encoder_dim": 512,
        "clip_projection_dim": 1024,
        "eos_token_id": 2,
        "hidden_size": 768,
        "cpt_name": "ViT-B32-small.pt"
    }
}


def prc_text(text):
    text = text.replace("\n", " ").replace("\t", " ")
    return " ".join(text.split())


def get_text_batch(lines, tokenizer, args):
    texts = []
    for text in lines:
        text = tokenizer(
            prc_text(text),
            padding="max_length",
            max_length=args.seq_length,
            truncation=True
        )
        text["input_ids"] = [tokenizer.bos_token_id] + text["input_ids"]
        text["attention_mask"] = [1] + text["attention_mask"]
        text["attention_mask"] = text["attention_mask"][:args.seq_length]
        text["input_ids"] = text["input_ids"][:args.seq_length]
        try:
            pos = text["input_ids"].index(tokenizer.pad_token_id)
        except ValueError:
            pos = -1
        text["input_ids"][pos] = tokenizer.eos_token_id
        text["attention_mask"][pos] = 1
        texts.append(text)
    input_ids = torch.LongTensor([x["input_ids"] for x in texts]).long()
    input_ids = input_ids.to(input_ids.device if args.cpu else torch.cuda.current_device())
    attention_mask = torch.LongTensor([x["attention_mask"] for x in texts]).long().to(
        input_ids.device if args.cpu else torch.cuda.current_device())
    return input_ids, attention_mask


def get_tokenizer(pretrained_model_name="sberbank-ai/rugpt3small_based_on_gpt2"):
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name)
    add_tokens = tokenizer.add_special_tokens({"bos_token": "<s>"})

    assert add_tokens == 0
    # add_tokens = tokenizer.add_special_tokens({"cls_token": "<case>"})
    add_tokens = tokenizer.add_special_tokens({"eos_token": "</s>"})
    assert add_tokens == 0
    add_tokens = tokenizer.add_special_tokens({"pad_token": "<pad>"})
    assert add_tokens == 0
    return tokenizer


def get_classes(path="../../clip/evaluate/classes.json"):
    with open(path, "r") as file:
        return json.load(file)


def prepare_classes(classes, ma=MorphAnalyzer()):
    res = []
    for cls in classes:
        cls = ma.parse(cls)[0].inflect({"gent"}).word
        res.append(cls)
    return [f"Изображение {label.lower()}" for label in res]


def call_model(model, tokenizer, args, texts, images):
    input_ids, attention_mask = get_text_batch(texts, tokenizer, args)
    with torch.no_grad():
        logits_per_image, logits_per_text = model(
            img_input={"x": get_image_batch(images, args.img_transform, args)},
            text_input={"x": input_ids, "attention_mask": attention_mask}
        )
    return logits_per_image, logits_per_text


def load_weights_only(
        pretrained_model_name_or_path="ViT-B/32-small",
        cpu=False,
        seq_length=128
):
    vals = locals()
    vals.update(MODELS[pretrained_model_name_or_path])

    class Args(object):
        def __init__(self, args):
            for k, v in args.items():
                setattr(self, k, v)
            self.img_transform = None

    return _load_weights_only(Args(vals))


def _load_weights_only(args):
    visual_model, img_transform = load(args.visual_encoder_name, jit=False)
    text_model = GPT2Model.from_pretrained(args.load_huggingface)
    visual_encoder = VisualEncoder(
        model=visual_model.visual,
        d_in=args.visual_encoder_dim,
        d_out=args.clip_projection_dim
    )
    text_encoder = TextEncoder(
        model=text_model,
        eos_token_id=args.eos_token_id,
        d_in=args.hidden_size,
        d_out=args.clip_projection_dim
    )
    model = CLIP(
        visual_encoder=visual_encoder,
        text_encoder=text_encoder,
        img_transform=img_transform
    )

    checkpoint_name = download_file_from_hf(args.cpt_name)

    sd = torch.load(checkpoint_name, map_location='cpu')
    model.load_state_dict(sd)
    args.img_transform = img_transform
    return model, args


def show_test_images(args):
    input_resolution = args.img_transform.transforms[0].size
    preprocess = Compose([
        Resize(input_resolution, interpolation=Image.BICUBIC),
        CenterCrop(input_resolution),
        ToTensor()
    ])
    descriptions = {
        "page": "страница текста о сегментации",
        "chelsea": "фото морды полосатого кота",
        "astronaut": "портрет космонавта с американским флагом",
        "rocket": "ракета стоит на стартовой площадке",
        "motorcycle_right": "красный мотоцикл стоит в гараже",
        "camera": "человек смотрит в камеру на штативе",
        "horse": "черно-белый силуэт лошади",
        "coffee": "чашка кофе на блюдце"
    }
    images = []
    texts = []
    plt.figure(figsize=(16, 5))
    img_paths = []

    for filename in [filename for filename in os.listdir(skimage.data_dir) if
                     filename.endswith(".png") or filename.endswith(".jpg")]:
        name = os.path.splitext(filename)[0]
        if name not in descriptions:
            continue
        fn = os.path.join(skimage.data_dir, filename)
        img_paths.append(fn)
        image = preprocess(Image.open(fn).convert("RGB"))
        images.append(fn)
        texts.append(descriptions[name])

        plt.subplot(2, 4, len(images))
        plt.imshow(image.permute(1, 2, 0))
        plt.title(f"{filename}\n{descriptions[name]}")
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    return images, texts


def show_similarity(images, texts, similarity, args):
    input_resolution = args.img_transform.transforms[0].size
    preprocess = Compose([
        Resize(input_resolution, interpolation=Image.BICUBIC),
        CenterCrop(input_resolution),
        ToTensor()
    ])
    count = len(texts)

    plt.figure(figsize=(20, 14))
    plt.imshow(similarity, vmin=0.1, vmax=0.3)
    plt.yticks(range(count), texts, fontsize=18)
    plt.xticks([])
    for i, image in enumerate(images):
        if isinstance(image, str):
            image = Image.open(image)
            image = preprocess(image.convert("RGB"))
        plt.imshow(image.permute(1, 2, 0), extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
    for x in range(similarity.shape[1]):
        for y in range(similarity.shape[0]):
            plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)

    for side in ["left", "top", "right", "bottom"]:
        plt.gca().spines[side].set_visible(False)

    plt.xlim([-0.5, count - 0.5])
    plt.ylim([count + 0.5, -2])

    plt.title("Cosine similarity between text and image features", size=20)


def show_topk_probs(images, labels, logits, args, k=5, ma=MorphAnalyzer()):
    input_resolution = args.img_transform.transforms[0].size
    preprocess = Compose([
        Resize(input_resolution, interpolation=Image.BICUBIC),
        CenterCrop(input_resolution),
        ToTensor()
    ])
    probs = logits.softmax(dim=-1)
    top_probs, top_labels = probs.cpu().topk(k, dim=-1)
    plt.figure(figsize=(16, 16))

    for i, image in enumerate(images):
        plt.subplot(4, 4, 2 * i + 1)
        if isinstance(image, str):
            image = Image.open(image)
            image = preprocess(image.convert("RGB"))
        plt.imshow(image.permute(1, 2, 0))
        plt.axis("off")

        plt.subplot(4, 4, 2 * i + 2)
        y = np.arange(top_probs.shape[-1])
        plt.grid()
        plt.barh(y, top_probs[i])
        plt.gca().invert_yaxis()
        plt.gca().set_axisbelow(True)
        plt.yticks(y, [labels[index] for index in top_labels[i].numpy()])
        plt.xlabel("probability")

    plt.subplots_adjust(wspace=0.5)
    plt.show()


def show_topk_accuracy(labels, probs, classes, k=5):
    successes_preds = []
    for lbl, p in zip(labels, np.array(probs)):
        _, top_labels = torch.tensor(p).topk(k, dim=-1)
        if lbl in top_labels:
            successes_preds.append(classes[lbl])
    successes_preds = Counter(successes_preds)
    lp = [(k, v) for k, v in successes_preds.items()]
    lp = sorted(lp, key=lambda x_: x_[0])
    x = [x[0] for x in lp]
    y = [x[1] for x in lp]
    plt.figure(figsize=(20, 8))
    plt.bar(x, y)
    plt.grid(axis='y')
    plt.ylabel('successes')
    plt.xticks(
        labels=x,
        ticks=np.arange(len(successes_preds)),
        rotation=90
    )
    _ = plt.title("Correct predictions by class")


def get_topk_accuracy(labels, probs, k=5):
    successes = 0
    for lbl, p in zip(labels, np.array(probs)):
        _, top_labels = torch.tensor(p).topk(k, dim=-1)
        if lbl in top_labels:
            successes += 1
    return successes / len(labels)
