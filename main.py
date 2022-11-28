from argparse import ArgumentParser
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig

from src.config import Config
from src.data import get_user_sequences, TrainDataset, ValidDataset
from src.evaluate import predict
from src.models import AktEncoderDecoderModel, SaintEncoderDecoderModel
from src.optim import NoamLR
from src.utils import set_seed, timer
from src.validation import virtual_time_split

from captum.attr import IntegratedGradients
from captum.attr import LayerConductance
from captum.attr import NeuronConductance

def main():
    parser = ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True)
    args = parser.parse_args()

    # settings
    config_path = Path(args.config_file)
    config = Config.load(config_path)

    warnings.filterwarnings('ignore')
    set_seed(config.seed)
    start_time = time.time()
    
    with timer('load data'):
        DATA_DIR = './input/riiid-test-answer-prediction/'
        usecols = [
            'row_id',
            'timestamp',
            'user_id',
            'content_id',
            'content_type_id',
            'task_container_id',
            'user_answer',
            'answered_correctly',
            'prior_question_elapsed_time', 
            'prior_question_had_explanation',
        ]#
        dtype = {
            'row_id': 'int64',
            'timestamp': 'int64',
            'user_id': 'int32',
            'content_id': 'int16',
            'content_type_id': 'int8',
            'task_container_id': 'int16',
            'user_answer': 'int8',
            'answered_correctly':'int8',
            'prior_question_elapsed_time': 'float32',
            'prior_question_had_explanation': 'boolean'
        }

        train_df = pd.read_csv(DATA_DIR + 'train.csv', usecols=usecols, dtype=dtype)
        question_df = pd.read_csv(DATA_DIR + 'questions.csv', usecols=['question_id', 'part'])
        # print the list using tolist()
        print("The column headers of train.csv are :")
        print(train_df.columns.tolist())
        print("The column headers of questions.csv are :")
        print(question_df.columns.tolist())


    train_df = train_df[train_df['content_type_id'] == 0].reset_index(drop=True)
    print("A number of column headers of train.csv are dropped and the remainig headers are :")
    print(train_df.columns.tolist())
    
    print()
   

            

    question_df['part'] += 1  # 0: padding id, 1: start id
    train_df['content_id'] += 2  # 0: padding id, 1: start id
    question_df['question_id'] += 2
    train_df = train_df.merge(question_df, how='left', left_on='content_id', right_on='question_id')

    print("Basic statistics description of the dataset:")
    for header_title in train_df.columns:
        print(header_title)
        print( train_df[header_title].describe())
        print()

    print('Print unique data to also check that there are no NA')
    print('content_type_id: ',train_df['content_type_id'].unique())    
    print('user_answer: ',train_df['user_answer'].unique())    
    print('answered_correctly: ',train_df['answered_correctly'].unique())    
    print('prior_question_had_explanation: ',train_df['prior_question_had_explanation'].unique()) 


    with timer('validation split'):
        train_idx, valid_idx, epoch_valid_idx = virtual_time_split(train_df,
                                                                   valid_size=config.valid_size,
                                                                   epoch_valid_size=config.epoch_valid_size)
        valid_y = train_df.iloc[valid_idx]['answered_correctly'].values
        epoch_valid_y = train_df.iloc[epoch_valid_idx]['answered_correctly'].values

    print('-' * 20)
    print(f'train size: {len(train_idx)}')
    print(f'valid size: {len(valid_idx)}')

    with timer('prepare data loader'):
        train_user_seqs = get_user_sequences(train_df.iloc[train_idx])
        valid_user_seqs = get_user_sequences(train_df.iloc[valid_idx])
        
        train_dataset = TrainDataset(train_user_seqs, window_size=config.window_size, stride_size=config.stride_size)
        valid_dataset = ValidDataset(train_df, train_user_seqs, valid_user_seqs, valid_idx, window_size=config.window_size)
        
        train_loader = DataLoader(train_dataset, **config.train_loader_params)
        valid_loader = DataLoader(valid_dataset, **config.valid_loader_params)

        # valid loader for epoch validation
        epoch_valid_user_seqs = get_user_sequences(train_df.iloc[epoch_valid_idx])
        epoch_valid_dataset = ValidDataset(train_df, train_user_seqs, epoch_valid_user_seqs, epoch_valid_idx, window_size=config.window_size)
        epoch_valid_loader = DataLoader(epoch_valid_dataset, **config.valid_loader_params)

    with timer('train'):
        if config.model == 'akt':
            content_encoder_config = BertConfig(**config.content_encoder_config)
            knowledge_encoder_config = BertConfig(**config.knowledge_encoder_config)
            decoder_config = BertConfig(**config.decoder_config)

            content_encoder_config.max_position_embeddings = config.window_size + 1
            knowledge_encoder_config.max_position_embeddings = config.window_size
            decoder_config.max_position_embeddings = config.window_size + 1

            model = AktEncoderDecoderModel(content_encoder_config, knowledge_encoder_config, decoder_config)

        elif config.model == 'saint':
            encoder_config = BertConfig(**config.encoder_config)
            decoder_config = BertConfig(**config.decoder_config)

            encoder_config.max_position_embeddings = config.window_size
            decoder_config.max_position_embeddings = config.window_size

            model = SaintEncoderDecoderModel(encoder_config, decoder_config)

        else:
            raise ValueError(f'Unknown model: {config.model}')

        model.to(config.device)
        model.zero_grad()
        
        optimizer = optim.Adam(model.parameters(), **config.optimizer_params)
        scheduler = NoamLR(optimizer, warmup_steps=config.warmup_steps)
        loss_ema = None

        for epoch in range(config.n_epochs):
            epoch_start_time = time.time()
            model.train()

            progress = tqdm(train_loader, desc=f'epoch {epoch + 1}', leave=False)
            for i, (x_batch, w_batch, y_batch) in enumerate(progress):
                y_pred = model(**x_batch.to(config.device).to_dict())
                loss = nn.BCEWithLogitsLoss(weight=w_batch.to(config.device))(y_pred, y_batch.to(config.device))
                loss.backward()

                if (
                    config.gradient_accumulation_steps is None
                    or (i + 1) % config.gradient_accumulation_steps == 0
                ):
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                
                loss_ema = loss_ema * 0.9 + loss.item() * 0.1 if loss_ema is not None else loss.item()
                progress.set_postfix(loss=loss_ema)

            valid_preds = predict(model, epoch_valid_loader, device=config.device)
            valid_score = roc_auc_score(epoch_valid_y, valid_preds)

            elapsed_time = time.time() - epoch_start_time
            print(f'Epoch {epoch + 1}/{config.n_epochs} \t valid score: {valid_score:.5f} \t time: {elapsed_time / 60:.1f} min')
    
    with timer('predict'):
        valid_preds = predict(model, valid_loader, device=config.device)
        valid_score = roc_auc_score(valid_y, valid_preds)

    print(f'valid score: {valid_score:.5f}')

    output_dir = Path(f'./output/{config_path.stem}/')
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), output_dir / 'model.pt')
    torch.save(optimizer.state_dict(), output_dir / 'optimizer.pt')

    elapsed_time = time.time() - start_time
    print(f'all processes done in {elapsed_time / 60:.1f} min.')

    # To apply integrated gradients, we first create an IntegratedGradients 
    # object from the Captum interpretability library, providing the 
    # model object.
    IntegratedGradients(model)

    # To compute the integrated gradients, we use the attribute method of the 
    # IntegratedGradients object. The method takes tensor(s) of input examples 
    # (matching the forward function of the model), and returns the input 
    # attributions for the given examples. For a network with multiple outputs, 
    # a target index must also be provided, defining the index of the output 
    # for which gradients are computed. For this example, we provide target = 1,
    # corresponding to wrong answer.

    # The input tensor provided should require grad, so we call requires\_grad\_ 
    # on the tensor. The attribute method also takes a baseline, which is the 
    # starting point from which gradients are integrated. The default value is 
    # just the 0 tensor, which is a reasonable baseline / default for this task.  

    # The returned values of the attribute method are the attributions, which 
    # match the size of the given inputs, and delta, which approximates the 
    # error between the approximated integral and true integral.

    #test_input_tensor.requires_grad_()
    #attr, delta = ig.attribute(test_input_tensor,target=1, return_convergence_delta=True)
    #attr = attr.detach().numpy()    

    
if __name__ == '__main__':
    main()
