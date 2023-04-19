import argparse
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

from models.multimodal import TextEncoder, SpeechEncoder
from my_merdataset import *
from config import *
from utils import *

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
    parser.add_argument(
        '--do_clf',
        action='store_true',
    )
    args = parser.parse_args()
    return args


args = parse_args()
if args.cuda != 'cuda:0':
    text_config['cuda'] = args.cuda
    test_config['cuda'] = args.cuda


def test(model, test_dataset):
    start_time = time.time()
    
    print("Test start")
    model.eval()

    with torch.no_grad():
        dataloader = DataLoader(test_dataset, args.batch,
                                collate_fn=lambda x: (x, torch.FloatTensor([i['label'] for i in x])))
        pred = []

        tq_test = tqdm(total=len(dataloader), desc="testing", position=2)

        for batch in dataloader:
            batch_x, batch_y = batch[0], batch[1]
            batch_y = batch_y.to(args.cuda)

            if isinstance(model,SpeechEncoder) or isinstance(model,TextEncoder):
                outputs = model(batch_x,do_clf=args.do_clf)

            else:
                outputs = model(batch_x)

            outputs = outputs.to(torch.float16)
            outputs = outputs.tolist()
            pred.extend(outputs)

            tq_test.update()

    end_time = time.time()
    
    return pred


def main():
    text_conf = pd.Series(text_config)

    if args.model_name:
        test_data = MERGEDataset(data_option='test', path='./data/')
        test_data.prepare_text_data(text_conf)

        model = torch.load('./ckpt/{}.pt'.format(args.model_name))
        print(model)
        output = test(model, test_data)
        
        output_path = 'distilled_knowledge.csv'
        
        df = pd.DataFrame(output)
        df.to_csv(output_path, index=False)
        #print(type(output))
        #print(output)
        #if os.path.isdir(os.path.join('result', args.model_name)) == False:
         #   os.system('mkdir -p ' + os.path.join('result', args.model_name))
        #save_dict_to_json(result, test_result_path)
        #print("Finish testing")

    else:
        print("You need to define specific model name to test")


if __name__ == '__main__':
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
    main()
