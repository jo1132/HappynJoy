import argparse
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

from models.multimodal import TextEncoder, SpeechEncoder
from merdataset import *
from config import *
from utils import *

from transformers import Wav2Vec2Processor, Wav2Vec2Model
import soundfile as sf

import time
def parse_args():
    parser = argparse.ArgumentParser(description='get arguments')
    parser.add_argument(
        '--batch',
        default=test_config['batch_size'],
        type=int,
        required=False,
        help='batch size'
    )

    parser.add_argument(
        '--cuda',
        default=test_config['cuda'],
        help='cuda'
    )

    parser.add_argument(
        '--model_name',
        type=str,
        help='checkpoint name to load'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='test all model ckpt in dir'
    )

    args = parser.parse_args()
    return args
args = parse_args()

def main():
    # model on mem
    model_name = './ckpt/best_multimodal_student.pt'
    model = torch.load(model_name, map_location=torch.device('cpu'))

    # encoder on mem
    encoder = Wav2Vec2Model.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")
    return_hidden_state = False
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

    text_conf = pd.Series(text_config)
    emo_map = {0: 'neutral',
                1: 'happy',
                2: 'surprise',
                3: 'angry',
                4: 'sad',
                5: 'disgust',
                6: 'fear'}

    PATH = './data/test_preprocessed_data.json'

    with open(PATH,'r') as file:
        base_json = json.load(file)

    item = base_json['data'][0]
    text = item['utterance']
    audio = item['wav']

    # total time
    total_time = time.time()

    path = './TOTAL/' + audio
    wav, _ = sf.read(path)
    if len(wav.shape) > 1:
        wav = wav.reshape((1,-1)).squeeze()

    inputs = processor(wav,
                        sampling_rate=16000,
                        return_attention_mask=True,
                        return_tensors="pt")
    audio = encoder(output_hidden_states=return_hidden_state, **inputs)
    audio = audio.last_hidden_state


    item['dialogue'] = text

    data = {
        'dialogue' : text,
        'wav' : audio,
    }
    #test_data = MERGEDataset(data_option='test', path='./data/')
    #test_data.prepare_text_data(text_conf)

    # predict time
    pred_time = time.time()
    pred = model([data])

    print('predict:',emo_map[pred.argmax().item()])
    print(item['Emotion'])
    print('total time:', time.time() - total_time)
    print('predict time:', time.time() - pred_time)

if __name__ == '__main__':
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
    main()