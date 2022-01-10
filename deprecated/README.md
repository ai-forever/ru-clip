# ru-clip
### First multimodal model for Russian language

**ru-clip** is a multimodal model for obtaining images and text similarities + rearranging captions and images.

**ru-clip** (Russian Contrastive Language–Image Pre-training) builds on a large body of work on zero-shot transfer, natural language supervision, and multimodal learning. The idea of zero-data learning dates back over a decade but until recently was mostly studied in computer vision as a way of generalizing to unseen object categories. 

We show that the continuation of work on the pre-trained language models [ru-gpts](https://github.com/sberbank-ai/ru-gpts) with the addition of a new modality - images - is able to make the system stable and generalize complex categories beyond standard samples.
![](https://habrastorage.org/webt/b4/fu/94/b4fu94nng6kzzedavmkawz3hasu.png)

**Note! This is the prototype model of OpenAI CLIP's Russian version following this [paper](https://arxiv.org/abs/2103.00020).**

## Model description
We use ViT-B/32 Image Encoder and RuGPT3Small Text Encoder.

🤗 See HF model cards:
 - [ru-clip](https://huggingface.co/sberbank-ai/ru-clip)
 - [ruGPT-3 small](https://huggingface.co/sberbank-ai/rugpt3small_based_on_gpt2)

## Usage
See [here](examples/Interacting_with_CLIP_ViT_B_32.ipynb), [![here](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sberbank-ai/ru-clip/blob/main/examples/Interacting_with_CLIP_ViT_B_32.ipynb).

## How it works
Habr post coming soon 

![](https://habrastorage.org/webt/et/20/vc/et20vcw-ikbfu_1tfyltdnvxsxk.png) 
