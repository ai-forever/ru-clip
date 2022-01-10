# -*- coding: utf-8 -*-
import torch
import more_itertools
from tqdm import tqdm


class Predictor:
    def __init__(self, clip_model, clip_processor, device, templates=None, bs=8):
        self.device = device
        self.clip_model = clip_model.to(self.device)
        self.clip_model.eval()
        self.clip_processor = clip_processor
        self.bs = bs
        self.templates = templates or [
            '{}',
            'фото, на котором изображено {}',
            'изображение с {}',
            'картинка с {}',
            'фото с {}',
            'на фото видно {}',
        ]

    def get_text_latents(self, class_labels):
        text_latents = []
        for template in self.templates:
            _text_latents = []
            for chunk in more_itertools.chunked(class_labels, self.bs):
                texts = [template.format(class_label.lower().strip()) for class_label in chunk]
                inputs = self.clip_processor(text=texts, return_tensors='pt', padding=True)
                _text_latents.append(self.clip_model.encode_text(inputs['input_ids'].to(self.device)))
            text_latents.append(torch.cat(_text_latents, dim=0))
        text_latents = torch.stack(text_latents).mean(0)
        text_latents = text_latents / text_latents.norm(dim=-1, keepdim=True)
        return text_latents

    def run(self, images, text_latents):
        labels = []
        pbar = tqdm()
        logit_scale = self.clip_model.logit_scale.exp()
        for pil_images in more_itertools.chunked(images, self.bs):
            inputs = self.clip_processor(text='', images=list(pil_images), return_tensors='pt', padding=True)
            image_latents = self.clip_model.encode_image(inputs['pixel_values'].to(self.device))
            image_latents = image_latents / image_latents.norm(dim=-1, keepdim=True)
            logits_per_text = torch.matmul(text_latents.to(self.device), image_latents.t()) * logit_scale
            _labels = logits_per_text.argmax(0).cpu().numpy().tolist()
            pbar.update(len(_labels))
            labels.extend(_labels)
        pbar.close()
        return labels

    def get_image_latents(self, images):
        pbar = tqdm()
        image_latents = []
        for pil_images in more_itertools.chunked(images, self.bs):
            inputs = self.clip_processor(text='', images=list(pil_images), return_tensors='pt', padding=True)
            image_latents.append(self.clip_model.encode_image(inputs['pixel_values'].to(self.device)))
            pbar.update(len(pil_images))
        image_latents = torch.cat(image_latents)
        image_latents = image_latents / image_latents.norm(dim=-1, keepdim=True)
        return image_latents
