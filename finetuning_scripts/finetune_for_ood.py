import pandas as pd
import re
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from joblib import dump, load
from sklearn import tree
import numpy as np
import itertools
import pickle

from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import load_dataset

from datasets import Dataset, DatasetDict

import argparse
from transformers import enable_full_determinism

from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from transformers import RobertaForSequenceClassification, AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
import numpy as np
import logging
import argparse
import evaluate
import datasets
import random
import os
import shutil
from transformers import AdamW
import tqdm

import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

parser = argparse.ArgumentParser(description='Run adversarial training with pre-generated LLM data.')
parser.add_argument('--no_epochs', type=int, const=10, default=10, nargs='?',
                    help='No. normal epochs before running adversarial')
parser.add_argument('--seed', type=int, const=0, default=0, nargs='?',
                    help='Seed to be used for shuffling.')
parser.add_argument('--batch_size', type=int, const=32, default=32, nargs='?',
                    help='Traing batch size.')
parser.add_argument('--batch_size_eval', type=int, const=256, default=256, nargs='?',
                    help='Eval batch size.')
parser.add_argument('--model_save_dir', type=str,
                    help='Where to save the final model to.')
parser.add_argument('--repeat', type=int, const=10, default=10, nargs='?',
                    help='How many times should the process be repeated.')
parser.add_argument('--base_model_type', type=str,
                    help='Specify what kind of mode to use.')
parser.add_argument(
    '--dataset',
    choices=['ag_news', 'news_topic', 'trec', 'yahoo', 'tweet_eval_sent', 'yelp', 'mnli'],
    required=True,
    help="Choose a dataset from: 'ag_news', 'news_topic', 'trec', 'yahoo', 'tweet_eval_sent', 'yelp', 'mnli'"
)
parser.add_argument(
    '--method',
    choices=['only', 'uniform'],
    required=True,
    help="What method for sample distribution was used."
)
parser.add_argument(
    '--icl_strategy',
    choices=['baseline', 'random', 'synth_dis', 'cos_sim', 'cos_div', 'cartography_easy', 'cartography_hard', 'cartography_easy_ambig', 'forgetting_least', 'forgetting_most'],
    required=True,
    help="What ICL sampling strategy was used."
)
parser.add_argument(
    '--method_gen',
    choices=['para', 'gen'],
    required=True,
    help="What generation method was used."
)

args = parser.parse_args()

dataset = args.dataset
method_to_load = args.method
method_gen = args.method_gen
icl_strategy = args.icl_strategy

default_dir = 'ICL_for_gen'

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(args.base_model_type)
random.seed(args.seed)

def prepare_data(df):
    dataset = Dataset.from_pandas(df)
    tokenized_datasets = tokenizer(dataset["text"], padding=True, return_tensors='pt', truncation=True)
    tokenized_datasets['label'] = dataset['label']
    tokenized_datasets['text'] = dataset['text']
    ds = Dataset.from_dict(tokenized_datasets).with_format("torch")
    return ds

def prepare_data_mnli(df):
    dataset = Dataset.from_pandas(df)
    tokenized_datasets = tokenizer(dataset['premise'], dataset['hypothesis'], padding=True, return_tensors='pt', truncation=True)
    tokenized_datasets['label'] = dataset['label']
    ds = Dataset.from_dict(tokenized_datasets).with_format("torch")
    return ds

def eval_loop(model, ds, eval_batch_size):
    model.eval().to(device)
    test_loader = DataLoader(ds, batch_size=eval_batch_size, shuffle=False)
    all_preds = []
    all_corrs = []
    
    total_batches = len(test_loader)
    #pbar = tqdm.tqdm(total=total_batches)
    with torch.no_grad():
        total_correct = 0
        for idx, batch in enumerate(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            # txts = list(batch['text'])

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            _, predicted = torch.max(outputs[1], 1)
            all_preds.extend(predicted)
            all_corrs.extend(labels)
            
            correct = (predicted == labels).sum().item()
            total_correct+=correct
            #pbar.update(1)
            
    #pbar.close()
    return all_preds, all_corrs

def train_loop(model, ds, BATCH_SIZE=16, NUM_EPOCHS=1):
    train_loader = DataLoader(ds['train'], batch_size=BATCH_SIZE, shuffle=True)
    #test_loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    model.to(device)

    optim = AdamW(model.parameters(), lr=2e-5)
    total_batches = len(train_loader)
    #pbar = tqdm.tqdm(total=total_batches)
    
    for epoch in range(NUM_EPOCHS):
        total_loss = 0

        model.train()
        for batch in train_loader:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optim.step()

            total_loss += loss.item()
            #pbar.update(1)
            
        if epoch % 5 == 0 and epoch != 0:
            logging.info("LOSS: " + str(total_loss/len(train_loader)))
        #pbar.close()        
    return model

dct_model_to_dir = {'FacebookAI/roberta-base': 'roberta', 'google-bert/bert-base-uncased':'bert'}

ood_pairs_dct = {'ag_news': 'news_topic',
 'news_topic': 'ag_news',
'tweet_eval_sent': 'yelp',
'yelp': 'tweet_eval_sent',
'trec': 'yahoo',
'yahoo': 'trec',
'mnli': 'mnli' 
}

random_state = args.seed

path_to_seeds_csv = os.path.join(default_dir, 'datasets', dataset, 'collected_data_llama', str(random_state), 'seeds_para.csv')
df_sub = pd.read_csv(path_to_seeds_csv)

if method_gen == 'para':
    str_gen = '_para'
else:
    str_gen = '_gen'
    
#model used, change if needed
MODEL_USED_DIR = '_llama'

df_sub = pd.read_csv(os.path.join(default_dir, 'datasets', dataset, 'collected_data' + MODEL_USED_DIR, str(random_state), 'seeds.csv'))
df_train_gen = pd.read_csv(os.path.join('ICL_for_gen', 'datasets', dataset, 'collected_data' + MODEL_USED_DIR, str(random_state), icl_strategy, method_to_load+str_gen+'.csv'))

df_train = pd.concat([df_sub, df_train_gen], axis=0).dropna().drop_duplicates()

if dataset == 'mnli':
    ds_train_orig = prepare_data_mnli(df_train)
else:
    ds_train_orig = prepare_data(df_train)

#logging.info(set(df_train['label']))

df_test_orig = pd.read_csv(os.path.join(default_dir, 'datasets', dataset, 'preprocess', 'test.csv')).drop_duplicates().dropna()
if dataset == 'mnli':
    ds_test_orig = prepare_data_mnli(df_test_orig)
else:
    ds_test_orig = prepare_data(df_test_orig)

#logging.info(set(df_test_orig['label']))

df_test_orig_ood = pd.read_csv(os.path.join(default_dir, 'datasets', ood_pairs_dct[dataset],  'preprocess', 'test.csv')).drop_duplicates().dropna()
if dataset == 'mnli':
    df_test_orig_ood = pd.read_csv(os.path.join(default_dir, 'datasets', ood_pairs_dct[dataset],  'preprocess', 'test_ood.csv')).drop_duplicates().dropna()
    ds_test_orig_ood = prepare_data_mnli(df_test_orig_ood)
else:
    ds_test_orig_ood = prepare_data(df_test_orig_ood)
    
ds_dct_orig = datasets.DatasetDict({"train":ds_train_orig,"test":ds_test_orig})

from transformers import AutoModelForSequenceClassification, TextClassificationPipeline
from sklearn import metrics

path_to_save = os.path.join(default_dir, 'datasets', dataset, 'id_res' + MODEL_USED_DIR, str(random_state), icl_strategy)
if 'google-bert' in args.base_model_type:
    path_to_save = os.path.join(default_dir, 'datasets', dataset, 'id_res' + MODEL_USED_DIR, str(random_state), 'bert', icl_strategy)
    
if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)

path_to_save_ood = os.path.join(default_dir, 'datasets', dataset, 'ood_res' + MODEL_USED_DIR, str(random_state), icl_strategy)
if 'google-bert' in args.base_model_type:
    path_to_save_ood = os.path.join(default_dir, 'datasets', dataset, 'ood_res' + MODEL_USED_DIR , str(random_state), 'bert', icl_strategy)
    
if not os.path.exists(path_to_save_ood):
    os.makedirs(path_to_save_ood)

path_for_res = os.path.join(path_to_save, method_to_load+'_'+str_gen+'.csv')
path_for_res_ood = os.path.join(path_to_save_ood, method_to_load+'_'+str_gen+'.csv')

path_for_res_acc = os.path.join(path_to_save, method_to_load+'_'+str_gen+'_acc.csv')
path_for_res_acc_ood = os.path.join(path_to_save_ood, method_to_load+'_'+str_gen+'_acc.csv')

if not (os.path.exists(path_for_res) and os.path.exists(path_for_res_ood) and os.path.exists(path_for_res_acc) and os.path.exists(path_for_res_acc_ood)):
    print(f'Running config: {dataset} {icl_strategy} {str(random_state)} {method_to_load} {str_gen} {args.base_model_type}')
    res_orig = []
    res_ood = []
    
    res_acc = []
    res_acc_ood = []
    
    for repeat_no in range(0, args.repeat):
        logging.info('***************** Now on repeat {} **************.'.format(repeat_no))
        logging.info('Running normal training for {} epochs.'.format(args.no_epochs))
    
        # if args.base_model_type == 'distilbert/distilbert-base-uncased':
        #     model = AutoModelForSequenceClassification.from_pretrained(args.base_model_type, num_labels=len(set(df_sub['label'])), dropout= 0.2)
        model = AutoModelForSequenceClassification.from_pretrained(args.base_model_type, num_labels=len(set(df_sub['label'])), classifier_dropout= 0.2)
    
        model = train_loop(model, ds_dct_orig, BATCH_SIZE=args.batch_size, NUM_EPOCHS=args.no_epochs)
    
        logging.info('Running evaluation on orig data.')
    
        preds, corrs = eval_loop(model, ds_test_orig, args.batch_size_eval)
        preds = [x.item() for x in preds]
        corrs = [x.item() for x in corrs]
        res = metrics.f1_score(corrs, preds, average='macro')
        res_acc.append(metrics.accuracy_score(corrs, preds))
        res_orig.append(res)
        logging.info('F1-macro orig: {}'.format(res))
    
        logging.info('Running evaluation on OOD data.')
        preds, corrs = eval_loop(model, ds_test_orig_ood, args.batch_size_eval)
        preds = [x.item() for x in preds]
        corrs = [x.item() for x in corrs]
        res = metrics.f1_score(corrs, preds, average='macro')
        res_acc_ood.append(metrics.accuracy_score(corrs, preds))
        res_ood.append(res)
        logging.info('F1-macro OOD: {}'.format(res))
        
        torch.cuda.empty_cache()
        del model
        
    dct_df = {'res_orig': res_orig}
    df_res = pd.DataFrame.from_dict(dct_df)
    
    dct_df_ood = {'res_orig': res_ood}
    df_res_ood = pd.DataFrame.from_dict(dct_df_ood)
    
    dct_df_acc = {'res_orig': res_acc}
    df_res_acc = pd.DataFrame.from_dict(dct_df_acc)
    
    dct_df_acc_ood = {'res_orig': res_acc_ood}
    df_res_acc_ood = pd.DataFrame.from_dict(dct_df_acc_ood)
    
    path_to_save = os.path.join(default_dir, 'datasets', dataset, 'id_res_llama', str(random_state), icl_strategy)
    if 'google-bert' in args.base_model_type:
        path_to_save = os.path.join(default_dir, 'datasets', dataset, 'id_res_llama', str(random_state), 'bert', icl_strategy)
        
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    
    path_to_save_ood = os.path.join(default_dir, 'datasets', dataset, 'ood_res_llama', str(random_state), icl_strategy)
    if 'google-bert' in args.base_model_type:
        path_to_save_ood = os.path.join(default_dir, 'datasets', dataset, 'ood_res_llama', str(random_state), 'bert', icl_strategy)
        
    if not os.path.exists(path_to_save_ood):
        os.makedirs(path_to_save_ood)
    
    path_for_res = os.path.join(path_to_save, method_to_load+'_'+str_gen+'.csv')
    path_for_res_ood = os.path.join(path_to_save_ood, method_to_load+'_'+str_gen+'.csv')
    
    path_for_res_acc = os.path.join(path_to_save, method_to_load+'_'+str_gen+'_acc.csv')
    path_for_res_acc_ood = os.path.join(path_to_save_ood, method_to_load+'_'+str_gen+'_acc.csv')
    
    res_file = os.path.join(path_for_res)
    res_file_ood = os.path.join(path_for_res_ood)
    
    res_file_acc = os.path.join(path_for_res_acc)
    res_file_acc_ood = os.path.join(path_for_res_acc_ood)
    
    df_res.to_csv(res_file, index=False)
    df_res_ood.to_csv(res_file_ood, index=False)
    
    df_res_acc.to_csv(res_file_acc, index=False)
    df_res_acc_ood.to_csv(res_file_acc_ood, index=False)
else:
    print(f'Configuration already exists. Skipping! {dataset} {icl_strategy} {str(random_state)} {method_to_load} {str_gen} {args.base_model_type}')