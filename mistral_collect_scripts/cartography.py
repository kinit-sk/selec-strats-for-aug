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
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TextClassificationPipeline
from sklearn import metrics
from collections import defaultdict
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

EPOCHS = 20
BATCH_SIZE = 32
default_dir = 'ICL_for_gen'
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
MODEL = 'FacebookAI/roberta-base'


def prepare_data(df):
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    dataset = Dataset.from_pandas(df)
    tokenized_datasets = tokenizer(dataset["text"], padding=True, return_tensors='pt', truncation=True)
    tokenized_datasets['label'] = dataset['label']
    tokenized_datasets['text'] = dataset['text']
    ds = Dataset.from_dict(tokenized_datasets).with_format("torch")
    return ds

def prepare_data_mnli(df):
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    df['text'] = 'Premise: ' + df['premise'] + ' Hypothesis: ' + df['hypothesis']
    dataset = Dataset.from_pandas(df.dropna())
    tokenized_datasets = tokenizer(dataset['premise'], dataset['hypothesis'], padding=True, return_tensors='pt', truncation=True)
    tokenized_datasets['label'] = dataset['label']
    tokenized_datasets['text'] = dataset['text']
    ds = Dataset.from_dict(tokenized_datasets).with_format("torch")
    return ds

# def prepare_data(df, batch_size=32):
#     dataset = Dataset.from_pandas(df)
#     batched_datasets = []
#     for i in range(0, len(dataset), batch_size):
#         batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
#         tokenized_batch = tokenizer(batch["text"], padding=True, return_tensors='pt', truncation=True)
#         tokenized_batch['label'] = batch['label']
#         tokenized_batch['text'] = batch['text']
#         batched_datasets.append(Dataset.from_dict(tokenized_batch).with_format("torch"))
#     return concatenate_datasets(batched_datasets)

def compute_forgetfulness(correctness_trend):
    if not any(correctness_trend):  # Example is never predicted correctly, or learnt!
        return 1000
    learnt = False  # Predicted correctly in the current epoch.
    times_forgotten = 0
    for is_correct in correctness_trend:
        if (not learnt and not is_correct) or (learnt and is_correct):
            # nothing changed.
            continue
        elif learnt and not is_correct:
            # Forgot after learning at some point!
            learnt = False
            times_forgotten += 1
        elif not learnt and is_correct:
            # Learnt!
            learnt = True
    return times_forgotten


def compute_correctness(trend):
  return sum(trend)



def compute_train_metrics(training_dynamics):
    confidence_ = {}
    variability_ = {}
    threshold_closeness_ = {}
    correctness_ = {}
    forgetfulness_ = {}

    # Functions to be applied to the data.
    # variability_func = lambda conf: np.std(conf)
    variability_func = lambda conf: np.sqrt(np.var(conf) + np.var(conf) * np.var(conf) / (len(conf)-1))
    threshold_closeness_func = lambda conf: conf * (1 - conf)

    loss = torch.nn.CrossEntropyLoss()

    num_tot_epochs = EPOCHS

    logits = {i: [] for i in range(num_tot_epochs)}
    targets = {i: [] for i in range(num_tot_epochs)}
    training_accuracy = defaultdict(float)

    for guid in training_dynamics:
        correctness_trend = []
        true_probs_trend = []

        record = training_dynamics[guid]
        for i, epoch_logits in enumerate(record["logits"]):
            probs = torch.nn.functional.softmax(torch.Tensor(epoch_logits), dim=-1)
            true_class_prob = float(probs[record["gold"]])
            true_probs_trend.append(true_class_prob)

            prediction = np.argmax(epoch_logits)
            is_correct = (prediction == record["gold"]).item()
            correctness_trend.append(is_correct)

            training_accuracy[i] += is_correct
            logits[i].append(epoch_logits)
            targets[i].append(record["gold"])

        correctness_[guid] = compute_correctness(correctness_trend)
        confidence_[guid] = np.mean(true_probs_trend)
        variability_[guid] = variability_func(true_probs_trend)

        forgetfulness_[guid] = compute_forgetfulness(correctness_trend)
        threshold_closeness_[guid] = threshold_closeness_func(confidence_[guid])

    column_names = ['guid',
                    'index',
                    'label',
                    'threshold_closeness',
                    'confidence',
                    'variability',
                    'correctness',
                    'forgetfulness',]
    df = pd.DataFrame([[guid,
                        i,
                        training_dynamics[guid]['gold'],
                        threshold_closeness_[guid],
                        confidence_[guid],
                        variability_[guid],
                        correctness_[guid],
                        forgetfulness_[guid],
                        ] for i, guid in enumerate(correctness_)], columns=column_names)

    return df


def run_cartography(dataset, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print('Loading data')
    path_to_seeds_csv = os.path.join(default_dir, 'datasets', dataset, 'collected_data_llama', str(seed), 'seeds.csv')
    df_train = pd.read_csv(path_to_seeds_csv)
    if dataset == 'mnli':
        ds_train_orig = prepare_data_mnli(df_train)
    else:
        ds_train_orig = prepare_data(df_train)        
    train_dynamics = {}

    logging.info('Running normal training for {} epochs.'.format(EPOCHS))

    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(set(df_train['label'])), classifier_dropout=0.3)
    train_loader = DataLoader(ds_train_orig, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(ds_train_orig, batch_size=256, shuffle=False)
    model.to(device)

    optim = AdamW(model.parameters(), lr=2e-5)
    print('Training model')
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch}')
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
        
        logging.info("LOSS: " + str(total_loss/len(train_loader)))

        print('Evaluating model')
        model.eval().to(device)
        with torch.no_grad():
            for idx, batch in enumerate(test_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                txts = list(batch['text'])

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)['logits'].detach().cpu().numpy().tolist()
                labels = labels.detach().cpu().numpy().tolist()

                for idx, output in enumerate(outputs):
                    guid = txts[idx]
                    if guid not in train_dynamics:
                        train_dynamics[guid] = {'gold': labels[idx], 'logits': []}
                    train_dynamics[guid]['logits'].append(output)
    print('Computing training metrics')
    training_metrics = compute_train_metrics(train_dynamics)

    return training_metrics