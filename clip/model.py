import torch
from torch import nn
import numpy as np
from .origin.clip import load
from transformers import GPT2Model
from PIL import Image


def gelu(x):
    return x * torch.sigmoid(1.702 * x)


class Projection(nn.Module):
    def __init__(self, d_in: int, d_out: int, p=0.5) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds


class VisualEncoder(nn.Module):
    def __init__(self, model, d_in, d_out):
        super().__init__()
        self.model = model
        self.projection = Projection(d_in, d_out)

    def forward(self, x):
        x = self.model(x)
        # print(x.shape)
        x = self.projection(x)
        projection_len = torch.norm(x, dim=-1, keepdim=True)
        return x / projection_len


class TextEncoder(nn.Module):
    def __init__(self, model, eos_token_id, d_in, d_out):
        super().__init__()
        self.model = model
        self.eos_token_id = eos_token_id
        self.projection = Projection(d_in, d_out)

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.model(x, **kwargs)[0][(x == self.eos_token_id).nonzero(as_tuple=True)]
        # print(x.shape)
        x = self.projection(x)
        projection_len = torch.norm(x, dim=-1, keepdim=True)
        return x / projection_len


class CLIP(nn.Module):
    def __init__(self, visual_encoder, text_encoder, img_transform):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.text_encoder = text_encoder
        self.img_transform = img_transform
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, img_input, text_input):
        image_features = self.visual_encoder(**img_input)
        text_features = self.text_encoder(**text_input)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()
        return logits_per_image, logits_per_text


def get_model(args):
    visual_model, args.img_transform = load(args.visual_encoder_name, jit=False)
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
        img_transform=args.img_transform
    )
    if args.freeze_visual_encoder:
        for p in model.visual_encoder.model.parameters():
            p.requires_grad = False

    if args.freeze_text_encoder:
        for p in model.text_encoder.model.parameters():
            p.requires_grad = False

    # To prevent OOM for model sizes that cannot fit in GPU memory in full precision
    if args.deepspeed and args.fp16:
        model.half()

    if not args.cpu:
        model.cuda(torch.cuda.current_device())

    return model


def get_image_batch(img_paths, img_transform, args):
    images = []
    for path in img_paths:
        if isinstance(path, Image.Image):
            image = path
        else:
            image = Image.open(path)
        image = image.convert("RGB")
        image = img_transform(image)
        images.append(image)
    images = torch.tensor(np.stack(images))
    images = images.to(images.device if args.cpu else torch.cuda.current_device())
    return images
