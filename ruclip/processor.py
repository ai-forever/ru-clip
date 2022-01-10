# -*- coding: utf-8 -*-
import os
import json

import torch
import numpy as np
import youtokentome as yttm
import torchvision.transforms as T
from torch.nn.utils.rnn import pad_sequence


class RuCLIPProcessor:
    eos_id = 3
    bos_id = 2
    unk_id = 1
    pad_id = 0

    def __init__(self, tokenizer_path, image_size=224, text_seq_length=77, mean=None, std=None):
        self.tokenizer = yttm.BPE(tokenizer_path)
        self.mean = mean or [0.48145466, 0.4578275, 0.40821073]
        self.std = std or [0.26862954, 0.26130258, 0.27577711]
        self.image_transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomResizedCrop(image_size, scale=(1., 1.), ratio=(1., 1.)),
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std)
        ])
        self.text_seq_length = text_seq_length
        self.image_size = image_size

    def encode_text(self, text):
        text = text.lower()
        tokens = self.tokenizer.encode([text], output_type=yttm.OutputType.ID, dropout_prob=0.0)[0]
        tokens = [self.bos_id] + tokens + [self.eos_id]
        tokens = tokens[:self.text_seq_length]
        return self.prepare_tokens(tokens)

    def prepare_tokens(self, tokens):
        empty_positions = self.text_seq_length - len(tokens)
        if empty_positions > 0:
            tokens = np.hstack((tokens, np.zeros(empty_positions)))  # position tokens after text
        if len(tokens) > self.text_seq_length:
            tokens = tokens[:self.text_seq_length-1] + tokens[-1:]
        return torch.tensor(tokens).long()

    def decode_text(self, encoded):
        return self.tokenizer.decode(encoded.cpu().numpy().tolist(), ignore_ids=[
            self.eos_id, self.bos_id, self.unk_id, self.pad_id
        ])[0]

    def __call__(self, text=None, images=None, **kwargs):
        inputs = {}
        if text is not None:
            input_ids = []
            texts = [text] if isinstance(text, str) else text
            for text in texts:
                tokens = self.encode_text(text)
                input_ids.append(tokens)
            inputs['input_ids'] = pad_sequence(input_ids, batch_first=True)
        if images is not None:
            pixel_values = []
            for i, image in enumerate(images):
                pixel_values.append(self.image_transform(image))
            inputs['pixel_values'] = pad_sequence(pixel_values, batch_first=True)
        return inputs

    @classmethod
    def from_pretrained(cls, folder):
        tokenizer_path = os.path.join(folder, 'bpe.model')
        config = json.load(open(os.path.join(folder, 'config.json')))
        image_size = config['image_resolution']
        text_seq_length = config['context_length']
        mean, std = config.get('mean'), config.get('std')
        return cls(tokenizer_path, image_size=image_size, text_seq_length=text_seq_length, mean=mean, std=std)
