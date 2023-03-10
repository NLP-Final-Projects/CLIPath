import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import gc
import multiprocessing
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import torch
from torch.cuda.amp import autocast
from torch import nn
from transformers import CLIPModel, CLIPConfig, CLIPVisionModel
from transformers import AutoModel, AutoTokenizer
from transformers import TrainingArguments, Trainer
from transformers import default_data_collator


df=pd.read_csv('E:/NLP/base_line_data.csv')

train,test = train_test_split(df,test_size=0.2, random_state=42)

DATA_FILE = 'dataset.csv'
TEST_SIZE = 0.05
TEXT_MODEL = 'm3hrdadfi/roberta-zwnj-wnli-mean-tokens'
IMAGE_MODEL = 'openai/clip-vit-base-patch32'
BATCH_SIZE = 64
IMAGE_SIZE = 224
MAX_LEN = 10
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])

if __name__ == '__main__':
    from transformers import TrainingArguments, AutoTokenizer, CLIPFeatureExtractor
    vision_preprocessor = CLIPFeatureExtractor.from_pretrained(IMAGE_MODEL) 
    args = TrainingArguments(
        "clip-fa",
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=5000,
        save_steps=5000,
        logging_steps=5000,
        learning_rate=3e-6,
        weight_decay=0.003,
        warmup_steps=100,
        fp16=False,
        prediction_loss_only=True,
        gradient_accumulation_steps=1,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=300,
        report_to='tensorboard'
    )



tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
def clear_gpu():
    torch.clear_autocast_cache()
    torch.cuda.ipc_collect()
    torch.cuda.empty_cache()
    gc.collect()


def optimal_workers():
    num_cpus = multiprocessing.cpu_count()
    num_gpus = torch.cuda.device_count()
    optimal_value = min(num_cpus, num_gpus*4) if num_gpus else num_cpus - 1
    return optimal_value


def showshow_data_data(batch, idx=0):
    # show image
    img = batch['pixel_values'][idx].permute(1, 2, 0)
    img = STD * img + MEAN
    print('Image shape: ', img.shape)
    plt.imshow(img)
    # show text
    text = tokenizer.decode(batch['input_ids'][idx],  skip_special_tokens=True)
    print('Text: ', text)


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


class CLIPDataset(Dataset):
    def __init__(self, image_paths: list, text: list, mode: str = 'train'):
        self.image_paths = image_paths
        self.tokens = tokenizer(text, padding='max_length',
                                max_length=MAX_LEN, truncation=True)

        if mode == 'train':
            self.augment = transforms.Compose([
                transforms.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD)
            ])
        elif mode == 'test':
            self.augment = transforms.Compose([
                transforms.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD)
            ])

    def __getitem__(self, idx):
        token = self.tokens[idx]
        return {'input_ids': token.ids, 'attention_mask': token.attention_mask,
                'pixel_values': self.augment(Image.open(self.image_paths[idx]).convert('RGB'))}

    def __len__(self):
        return len(self.image_paths)



train_ds = CLIPDataset(image_paths=train.images.tolist(),
                        text=train.captions.tolist(), mode='train')
test_ds = CLIPDataset(image_paths=test.images.tolist(),
                        text=test.captions.tolist(), mode='test')

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                        collate_fn=default_data_collator)

from transformers import BertConfig, BertModel
configuration = BertConfig(num_hidden_layers = 6,num_attention_heads = 6)

# Initializing a model (with random weights) from the bert-base-uncased style configuration
model = BertModel(configuration)


from transformers import CLIPVisionConfig, CLIPVisionModel
configuration = CLIPVisionConfig()
vision_encoder = CLIPVisionModel(configuration)

text_encoder= model


assert text_encoder.config.hidden_size == vision_encoder.config.hidden_size

clip = clip_wraper_creator()
clip.text_model = text_encoder
clip.vision_model = vision_encoder

clip.cuda()

class CLIPTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs, return_loss=True)
        return outputs["loss"]

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            if 1:
                with autocast():
                    loss = self.compute_loss(model, inputs)
            else:
                loss = self.compute_loss(model, inputs)
        return (loss, None, None)


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # clear_gpu()
    args.dataloader_num_workers = optimal_workers()
    trainer = CLIPTrainer(clip, args,
                          train_dataset=train_ds,
                          eval_dataset=test_ds)

    trainer.train('E:/NLP/clip-fa/checkpoint-25000')
