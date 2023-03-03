import gc
import os
import pathlib

import numpy as np
import staintools
import torch
from PIL import Image
from torch import nn

from transformers import BertConfig, BertModel, AutoTokenizer, CLIPFeatureExtractor
from transformers import CLIPVisionConfig, CLIPModel, CLIPConfig, CLIPVisionModel

from .models import Task


# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')
BASE_PATH = pathlib.Path(__file__).parent.resolve()
TARGET_IMAGE_PATH = os.path.join(BASE_PATH, 'static/colorstandard_brca.png')


def baseline_pipe(task: Task):

    task.status = Task.IN_PROCESS
    task.save()
    original_image = np.array(Image.open(task.image.path))
    normalized_image = normalization(original_image)
    probs = predict(task, normalized_image)
    for p, q in zip(probs.tolist()[0], task.queries.all()):
        q.probability = p
        q.save()
    task.status = Task.FINISHED
    task.save()


def normalization(image):
    # fit normalizer to sample picture
    normalizer = staintools.StainNormalizer(method='vahadane')
    target_image = staintools.read_image(TARGET_IMAGE_PATH)
    normalizer.fit(target_image)
    # normalize input picture
    normalized_image = normalizer.transform(image)
    normalized_image = Image.fromarray(normalized_image)
    # normalized_image.save(f"{normalized_base_path}/{f}/{image_size}/{image_ids[i]}.png", "PNG")
    return normalized_image


def predict(task, normalized_image):
    # load pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # build and load model
    clip = baseline_model()
    # Open the image file and convert it to RGB format
    image = normalized_image.convert("RGB")
    # image = Image.open('E:/NLP/content/normalized/TCGA-EW-A1J2-01Z-00-DX1/3072/15860_43508.png').convert("RGB")

    # Create a CLIPFeatureExtractor object and use it to pre-process the image
    vision_preprocessor = CLIPFeatureExtractor.from_pretrained('openai/clip-vit-base-patch32')
    edit_image = vision_preprocessor(image)

    # Transpose the tensor to the correct shape
    process_image = edit_image['pixel_values'][0].transpose(0, 1, 2)

    # Convert the image to a tensor and move it to the GPU
    input_image = torch.tensor(np.stack([process_image])).to(DEVICE)
    text_descriptions = []
    for q in task.queries.all():
        text_descriptions.append(q.text)

    tokens = tokenizer(text_descriptions, padding=True, truncation=True)

    with torch.no_grad():
        image_features = clip.vision_model(input_image).pooler_output.float()
        text_features = clip.text_model(torch.tensor(tokens.input_ids).to(DEVICE),
                                        torch.tensor(tokens.attention_mask).to(DEVICE)).pooler_output.float()

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (1.0 * image_features @ text_features.T).softmax(dim=-1)
    # top_probs, top_labels = text_probs.cpu().topk(len(text_descriptions), dim=-1)
    if 'cuda' in DEVICE.type:
        clear_gpu()

    return text_probs.cpu()


def baseline_model():
    txt_configuration = BertConfig(num_hidden_layers=6, num_attention_heads=6)
    # Initializing a model (with random weights) from the bert-base-uncased style configuration
    text_encoder = BertModel(txt_configuration)

    vision_configuration = CLIPVisionConfig()
    vision_encoder = CLIPVisionModel(vision_configuration)

    clip = clip_wraper_creator()
    clip.text_model = text_encoder
    clip.vision_model = vision_encoder

    ##### load model
    clip.load_state_dict(torch.load(os.path.join(BASE_PATH, 'data/pytorch_model_baseline.bin'), map_location=torch.device('cpu')))
    clip.eval()
    return clip.to(DEVICE)


def clip_wraper_creator():
    """create a dummy CLIPModel to wrap text and vision encoders in order to use CLIPTrainer"""
    config = {'num_hidden_layers': 0,
              'max_position_embeddings': 0,
              'vocab_size': 0,
              'hidden_size': 1,
              'patch_size': 1,
              }
    DUMMY_CONFIG = CLIPConfig(text_config_dict=config,
                              vision_config_dict=config)
    clip = CLIPModel(config=DUMMY_CONFIG)
    # convert projectors to Identity
    clip.text_projection = nn.Identity()
    clip.visual_projection = nn.Identity()
    return clip


def clear_gpu():
    torch.clear_autocast_cache()
    torch.cuda.ipc_collect()
    torch.cuda.empty_cache()
    gc.collect()
