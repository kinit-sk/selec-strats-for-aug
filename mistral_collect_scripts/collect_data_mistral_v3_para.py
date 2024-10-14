import json
import os
from pprint import pprint
import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers
import pickle
from datasets import load_dataset
import pandas as pd
from huggingface_hub import notebook_login
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
import numpy as np
import re
import string

from transformers.utils import logging

logging.set_verbosity(40) # only log errors

import logging
import argparse
from sample_selection_strategies import get_random_examples_target_label_only, get_random_examples_uniform, format_examples, get_synth_dis_examples_target_label_only, get_similar_examples_para, get_diverse_examples_para, get_cartography_easy_samples, get_cartography_hard_samples, get_cartography_easy_ambig_samples, get_cartography_forgotten_samples, get_cartography_learned_samples, format_examples_mnli
from sentence_transformers import SentenceTransformer

# Initialize the ArgumentParser
parser = argparse.ArgumentParser(description="Collect data via LLM.")

parser.add_argument(
    '--method',
    choices=['only', 'uniform'],
    required=True,
    help="What method to use for sample distribution in prompt based on labels"
)

parser.add_argument(
    '--icl_strategy',
    choices=['baseline', 'random', 'synth_dis', 'cos_sim', 'cos_div', 'cartography_easy', 'cartography_hard', 'cartography_easy_ambig', 'forgetting_most', 'forgetting_least'],
    required=True,
    help="What ICL sampling strategy to use."
)

parser.add_argument(
    '--dataset',
    choices=['ag_news', 'news_topic', 'trec', 'yahoo', 'tweet_eval_sent', 'yelp', 'mnli'],
    required=True,
    help="Choose a dataset from: 'ag_news', 'news_topic', 'trec', 'yahoo', 'tweet_eval_sent', 'yelp', 'mnli'"
)

parser.add_argument('--seed', type=int, const=0, default=0, nargs='?', help='Seed to be used for sampling.')

args = parser.parse_args()

dataset = args.dataset
method = args.method
icl_strategy = args.icl_strategy

if method == 'only' and icl_strategy == 'baseline':
    exit(0)

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['CURL_CA_BUNDLE'] = ''

model_id = "mistralai/Mistral-7B-Instruct-v0.3"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="cuda",
    tokenizer=tokenizer
)

default_prompt_baseline = """You are given a '{task}' classification dataset. Paraphrase a given text 5 times with the '{label}' category. Output each generated text in the form of a numbered list separated by new lines. The text: '{text}"""

default_qa_prompt_baseline = """You are given a '{task}' classification dataset. Paraphrase a given question 5 times with the '{label}' category. Output each generated question in the form of a numbered list separated by new lines. The question: '{text}"""

default_prompt_icl = """You will be given examples from '{task}' classification dataset, each labeled with a specific category. Based on the examples, paraphrase a given text 5 times with the '{label}' category. Output each paraphrased text in the form of a numbered list separated by new lines. The text: '{text}'
Examples:
{examples}
"""

default_qa_prompt_icl = """You will be given examples of questions from '{task}' classification dataset, each labeled with a specific topic. Based on the examples of questions, paraphrase a given question 5 times with the '{label}' topic. Output each paraphrased question in the form of a numbered list separated by new lines. The question: '{text}'
Examples:
{examples}
"""

default_mnli_prompt_baseline = """You will be given a premise from a Natural Language Inference dataset. Paraphrase 5 times a hypothesis that '{label}' the given premise. The given premise: '{premise}'. Output each paraphrased hypothesis in the form of a numbered list separated by new lines. The hypothesis: '{text}'
"""

default_mnli_prompt_icl = """You will be given a premise and hypothesis pair together with their label from a Natural Language Inference dataset. Based on the examples, paraphrase 5 times a hypothesis that '{label}' the given premise. The given premise: '{premise}'. Output each paraphrased hypothesis in the form of a numbered list separated by new lines. The hypothesis: '{text}'
Examples:
{examples}
"""

dataset_to_labels_dct = {'ag_news': ['World', 'Sports', 'Business', 'Science and Technology'],
 'news_topic': ['World', 'Sports', 'Business', 'Science and Technology'],
'clinc150': ['AddToPlaylist', 'BookRestaurant', 'GetWeather', 'PlayMusic'],
'snips': ['AddToPlaylist', 'BookRestaurant', 'GetWeather', 'PlayMusic'],
'tweet_eval_sent': ['negative', 'neutral', 'positive'],
'yelp': ['negative', 'neutral', 'positive'],
'trec': ['Society & Culture', 'Science & Mathematics', 'Health', 'Education & Reference', 'Sports', 'Business & Finance', 'Entertainment & Music'],
'yahoo': ['Society & Culture', 'Science & Mathematics', 'Health', 'Education & Reference', 'Sports', 'Business & Finance', 'Entertainment & Music'],
 'mnli': ['entail', 'are neutral', 'contradict']
}

dataset_to_task_dct = {'ag_news': 'News Topic',
 'news_topic': 'News Topic',
'clinc150': 'Intent',
'snips': 'Intent',
'tweet_eval_sent': 'Tweet Sentiment',
'yelp': 'Review Sentiment',
'trec': 'Question Topic',
'yahoo': 'Question Topic'
}

format_mnli_text = {}

NO_SHOTS_PER_LABEL = 10 # 5
NO_ALL_SAMPLES_PER_LABEL = 100 # 20

folder_addiction = f'_{NO_ALL_SAMPLES_PER_LABEL}' if NO_ALL_SAMPLES_PER_LABEL != 20 else ''

if icl_strategy in ['synth_dis', 'cos_sim', 'cos_div']:
    sent_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def collect_samples(final_prompts):
    lst_responses = []

    for count, tmp_prompt in enumerate(final_prompts):
        label = tmp_prompt[1]
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": tmp_prompt[0]},
        ]
        
        prompt = pipeline.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
        )
        
        terminators = [
            pipeline.tokenizer.eos_token_id
        ]
        
        outputs = pipeline(
            prompt,
            max_new_tokens=1024,
            eos_token_id=terminators,
            do_sample=True,
            temperature=1.0,
            top_p=1.0
        )
        
        if count > 0 and count % 10 == 0:
            logging.info('Now on count: {}'.format(count))
        #print(outputs[0])
        lst_responses.append((outputs[0], tmp_prompt[1]))
            
    return lst_responses

random_state = args.seed

train_data_path = pd.read_csv(os.path.join('ICL_for_gen', 'datasets', dataset, 'preprocess', 'train.csv')).dropna()
df_sub = train_data_path.groupby('label', group_keys=False).apply(lambda x: x.sample(NO_ALL_SAMPLES_PER_LABEL, random_state=random_state)) # uniform subsampling of the seed data

final_prompts = []

if icl_strategy == 'baseline':
    for index, row in df_sub.iterrows():
        rest_of_df = df_sub.drop(index)
        if dataset != 'mnli':
            text = row['text']
        label = row['label']
        
        if dataset == 'yahoo' or dataset == 'trec':
            final_prompts.append((default_qa_prompt_baseline.format(task=dataset_to_task_dct[dataset], text=text, label=dataset_to_labels_dct[dataset][label]), label))
        elif dataset == 'mnli':
            final_prompts.append((default_mnli_prompt_baseline.format(premise=row['premise'], text=row['hypothesis'], label=dataset_to_labels_dct[dataset][row['label']]), {'label': row['label'], 'premise': row['premise']}))
        else:
            final_prompts.append((default_prompt_baseline.format(task=dataset_to_task_dct[dataset], text=text, label=dataset_to_labels_dct[dataset][label]), label))

elif icl_strategy == 'synth_dis':
    df_0 = pd.read_csv(os.path.join('ICL_for_gen', 'datasets', dataset, f'collected_data{folder_addiction}', str(random_state), 'baseline', 'uniform_para.csv'))
    for index, row in df_sub.iterrows():
        if dataset != 'mnli':
            rest_of_df = df_sub.drop(index)
            text = row['text']
        else:
            rest_of_df = df_sub.drop(index)
            rest_of_df['text'] = 'Premise: ' + rest_of_df['premise'] + ' Hypothesis: ' + rest_of_df['hypothesis']
        label = row['label']     
        examples = get_synth_dis_examples_target_label_only(rest_of_df, label, NO_SHOTS_PER_LABEL, sent_model)
        
        if dataset == 'mnli':
            str_examples = format_examples_mnli(dataset, examples, dataset_to_labels_dct)
        else:        
            str_examples = format_examples(dataset, examples, dataset_to_labels_dct)

        if dataset == 'yahoo' or dataset == 'trec':
            final_prompts.append((default_qa_prompt_baseline.format(task=dataset_to_task_dct[dataset], text=text, label=dataset_to_labels_dct[dataset][label]), label))
        elif dataset == 'mnli':
            final_prompts.append((default_mnli_prompt_icl.format(premise=row['premise'], text=row['hypothesis'], examples=str_examples, label=dataset_to_labels_dct[dataset][row['label']]), {'label': row['label'], 'premise': row['premise']}))
        else:
            final_prompts.append((default_prompt_baseline.format(task=dataset_to_task_dct[dataset], text=text, label=dataset_to_labels_dct[dataset][label]), label))

elif icl_strategy == 'cos_sim':
    for index, row in df_sub.iterrows():
        if dataset != 'mnli':
            text = row['text']
        else:
            df_sub['text'] = 'Premise: ' + df_sub['premise'] + ' Hypothesis: ' + df_sub['hypothesis']
        rest_of_df = df_sub.drop(index)
            
        label = row['label']
        examples = get_similar_examples_para(rest_of_df, label, NO_SHOTS_PER_LABEL, sent_model, df_sub.loc[[index]], method != 'uniform')
            
        str_examples = format_examples(dataset, examples, dataset_to_labels_dct)
        
        if dataset == 'mnli':
            str_examples = format_examples_mnli(dataset, examples, dataset_to_labels_dct)
        else:        
            str_examples = format_examples(dataset, examples, dataset_to_labels_dct)

        if dataset == 'yahoo' or dataset == 'trec':
            final_prompts.append((default_qa_prompt_icl.format(task=dataset_to_task_dct[dataset], examples=str_examples, text=text, label=dataset_to_labels_dct[dataset][label]), label))
        elif dataset == 'mnli':
            final_prompts.append((default_mnli_prompt_icl.format(premise=row['premise'], text=row['hypothesis'], examples=str_examples, label=dataset_to_labels_dct[dataset][row['label']]), {'label': row['label'], 'premise': row['premise']}))
        else:
            final_prompts.append((default_prompt_icl.format(task=dataset_to_task_dct[dataset], examples=str_examples, text=text, label=dataset_to_labels_dct[dataset][label]), label))

elif icl_strategy == 'cos_div':
    for index, row in df_sub.iterrows():
        if dataset != 'mnli':
            text = row['text']
        else:
            df_sub['text'] = 'Premise: ' + df_sub['premise'] + ' Hypothesis: ' + df_sub['hypothesis']
        rest_of_df = df_sub.drop(index)
        
        label = row['label']
        examples = get_diverse_examples_para(rest_of_df, label, NO_SHOTS_PER_LABEL, sent_model, df_sub.loc[[index]], method != 'uniform')
            
        if dataset == 'mnli':
            str_examples = format_examples_mnli(dataset, examples, dataset_to_labels_dct)
        else:        
            str_examples = format_examples(dataset, examples, dataset_to_labels_dct)

        if dataset == 'yahoo' or dataset == 'trec':
            final_prompts.append((default_qa_prompt_icl.format(task=dataset_to_task_dct[dataset], examples=str_examples, text=text, label=dataset_to_labels_dct[dataset][label]), label))
        elif dataset == 'mnli':
            final_prompts.append((default_mnli_prompt_icl.format(premise=row['premise'], text=row['hypothesis'], examples=str_examples, label=dataset_to_labels_dct[dataset][row['label']]), {'label': row['label'], 'premise': row['premise']}))
        else:
            final_prompts.append((default_prompt_icl.format(task=dataset_to_task_dct[dataset], examples=str_examples, text=text, label=dataset_to_labels_dct[dataset][label]), label))

elif icl_strategy == 'cartography_easy':
    for index, row in df_sub.iterrows():
        rest_of_df = df_sub.drop(index)
        if dataset != 'mnli':
            text = row['text']
        else:
            rest_of_df['text'] = 'Premise: ' + rest_of_df['premise'] + ' Hypothesis: ' + rest_of_df['hypothesis']
        label = row['label']
        examples = get_cartography_easy_samples(rest_of_df, label, NO_SHOTS_PER_LABEL, random_state, dataset, method != 'uniform')
            
        if dataset == 'mnli':
            str_examples = format_examples_mnli(dataset, examples, dataset_to_labels_dct)
        else:        
            str_examples = format_examples(dataset, examples, dataset_to_labels_dct)

        if dataset == 'yahoo' or dataset == 'trec':
            final_prompts.append((default_qa_prompt_icl.format(task=dataset_to_task_dct[dataset], examples=str_examples, text=text, label=dataset_to_labels_dct[dataset][label]), label))
        elif dataset == 'mnli':
            final_prompts.append((default_mnli_prompt_icl.format(premise=row['premise'], text=row['hypothesis'], examples=str_examples, label=dataset_to_labels_dct[dataset][row['label']]), {'label': row['label'], 'premise': row['premise']}))
        else:
            final_prompts.append((default_prompt_icl.format(task=dataset_to_task_dct[dataset], examples=str_examples, text=text, label=dataset_to_labels_dct[dataset][label]), label))

elif icl_strategy == 'cartography_hard':
    for index, row in df_sub.iterrows():
        rest_of_df = df_sub.drop(index)
        if dataset != 'mnli':
            text = row['text']
        else:
            rest_of_df['text'] = 'Premise: ' + rest_of_df['premise'] + ' Hypothesis: ' + rest_of_df['hypothesis']
        label = row['label']
        examples = get_cartography_hard_samples(rest_of_df, label, NO_SHOTS_PER_LABEL, random_state, dataset, method != 'uniform')
            
        if dataset == 'mnli':
            str_examples = format_examples_mnli(dataset, examples, dataset_to_labels_dct)
        else:        
            str_examples = format_examples(dataset, examples, dataset_to_labels_dct)

        if dataset == 'yahoo' or dataset == 'trec':
            final_prompts.append((default_qa_prompt_icl.format(task=dataset_to_task_dct[dataset], examples=str_examples, text=text, label=dataset_to_labels_dct[dataset][label]), label))
        elif dataset == 'mnli':
            final_prompts.append((default_mnli_prompt_icl.format(premise=row['premise'], text=row['hypothesis'], examples=str_examples, label=dataset_to_labels_dct[dataset][row['label']]), {'label': row['label'], 'premise': row['premise']}))
        else:
            final_prompts.append((default_prompt_icl.format(task=dataset_to_task_dct[dataset], examples=str_examples, text=text, label=dataset_to_labels_dct[dataset][label]), label))

elif icl_strategy == 'cartography_easy_ambig':
    for index, row in df_sub.iterrows():
        rest_of_df = df_sub.drop(index)
        if dataset != 'mnli':
            text = row['text']
        else:
            rest_of_df['text'] = 'Premise: ' + rest_of_df['premise'] + ' Hypothesis: ' + rest_of_df['hypothesis']
        label = row['label']
        examples = get_cartography_easy_ambig_samples(rest_of_df, label, NO_SHOTS_PER_LABEL, random_state, dataset, method != 'uniform')
            
        if dataset == 'mnli':
            str_examples = format_examples_mnli(dataset, examples, dataset_to_labels_dct)
        else:        
            str_examples = format_examples(dataset, examples, dataset_to_labels_dct)

        if dataset == 'yahoo' or dataset == 'trec':
            final_prompts.append((default_qa_prompt_icl.format(task=dataset_to_task_dct[dataset], examples=str_examples, text=text, label=dataset_to_labels_dct[dataset][label]), label))
        elif dataset == 'mnli':
            final_prompts.append((default_mnli_prompt_icl.format(premise=row['premise'], text=row['hypothesis'], examples=str_examples, label=dataset_to_labels_dct[dataset][row['label']]), {'label': row['label'], 'premise': row['premise']}))
        else:
            final_prompts.append((default_prompt_icl.format(task=dataset_to_task_dct[dataset], examples=str_examples, text=text, label=dataset_to_labels_dct[dataset][label]), label))

elif icl_strategy == 'forgetting_most':
    for index, row in df_sub.iterrows():
        rest_of_df = df_sub.drop(index)
        if dataset != 'mnli':
            text = row['text']
        else:
            rest_of_df['text'] = 'Premise: ' + rest_of_df['premise'] + ' Hypothesis: ' + rest_of_df['hypothesis']
        label = row['label']
        examples = get_cartography_forgotten_samples(rest_of_df, label, NO_SHOTS_PER_LABEL, random_state, dataset, method != 'uniform')
            
        if dataset == 'mnli':
            str_examples = format_examples_mnli(dataset, examples, dataset_to_labels_dct)
        else:        
            str_examples = format_examples(dataset, examples, dataset_to_labels_dct)

        if dataset == 'yahoo' or dataset == 'trec':
            final_prompts.append((default_qa_prompt_icl.format(task=dataset_to_task_dct[dataset], examples=str_examples, text=text, label=dataset_to_labels_dct[dataset][label]), label))
        elif dataset == 'mnli':
            final_prompts.append((default_mnli_prompt_icl.format(premise=row['premise'], text=row['hypothesis'], examples=str_examples, label=dataset_to_labels_dct[dataset][row['label']]), {'label': row['label'], 'premise': row['premise']}))
        else:
            final_prompts.append((default_prompt_icl.format(task=dataset_to_task_dct[dataset], examples=str_examples, text=text, label=dataset_to_labels_dct[dataset][label]), label))

elif icl_strategy == 'forgetting_least':
    for index, row in df_sub.iterrows():
        rest_of_df = df_sub.drop(index)
        if dataset != 'mnli':
            text = row['text']
        else:
            rest_of_df['text'] = 'Premise: ' + rest_of_df['premise'] + ' Hypothesis: ' + rest_of_df['hypothesis']
        label = row['label']
        examples = get_cartography_learned_samples(rest_of_df, label, NO_SHOTS_PER_LABEL, random_state, dataset, method != 'uniform')
            
        if dataset == 'mnli':
            str_examples = format_examples_mnli(dataset, examples, dataset_to_labels_dct)
        else:        
            str_examples = format_examples(dataset, examples, dataset_to_labels_dct)

        if dataset == 'yahoo' or dataset == 'trec':
            final_prompts.append((default_qa_prompt_icl.format(task=dataset_to_task_dct[dataset], examples=str_examples, text=text, label=dataset_to_labels_dct[dataset][label]), label))
        elif dataset == 'mnli':
            final_prompts.append((default_mnli_prompt_icl.format(premise=row['premise'], text=row['hypothesis'], examples=str_examples, label=dataset_to_labels_dct[dataset][row['label']]), {'label': row['label'], 'premise': row['premise']}))
        else:
            final_prompts.append((default_prompt_icl.format(task=dataset_to_task_dct[dataset], examples=str_examples, text=text, label=dataset_to_labels_dct[dataset][label]), label))

else:
    for index, row in df_sub.iterrows():
        rest_of_df = df_sub.drop(index)
        if dataset != 'mnli':
            text = row['text']
        else:
            rest_of_df['text'] = 'Premise: ' + rest_of_df['premise'] + ' Hypothesis: ' + rest_of_df['hypothesis']
        label = row['label']
        if method == 'uniform':
            examples = get_random_examples_uniform(rest_of_df, label, NO_SHOTS_PER_LABEL, index)
        else:
            examples = get_random_examples_target_label_only(rest_of_df, label, NO_SHOTS_PER_LABEL, index)
            
        if dataset == 'mnli':
            str_examples = format_examples_mnli(dataset, examples, dataset_to_labels_dct)
        else:        
            str_examples = format_examples(dataset, examples, dataset_to_labels_dct)

        if dataset == 'yahoo' or dataset == 'trec':
            final_prompts.append((default_qa_prompt_icl.format(task=dataset_to_task_dct[dataset], examples=str_examples, text=text, label=dataset_to_labels_dct[dataset][label]), label))
        elif dataset == 'mnli':
            final_prompts.append((default_mnli_prompt_icl.format(premise=row['premise'], text=row['hypothesis'], examples=str_examples, label=dataset_to_labels_dct[dataset][row['label']]), {'label': row['label'], 'premise': row['premise']}))
        else:
            final_prompts.append((default_prompt_icl.format(task=dataset_to_task_dct[dataset], examples=str_examples, text=text, label=dataset_to_labels_dct[dataset][label]), label))
    
responses = collect_samples(final_prompts)

path_to_save = os.path.join('ICL_for_gen', 'datasets', dataset, f'collected_data{folder_addiction}', str(random_state), icl_strategy)
if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)

path_for_file = os.path.join(path_to_save, method+'_para.pkl')
path_to_seeds_csv = os.path.join('ICL_for_gen', 'datasets', dataset, f'collected_data{folder_addiction}', str(random_state), 'seeds_para.csv')
df_sub.to_csv(path_to_seeds_csv, index=False)
with open(path_for_file, 'wb') as handle:
    pickle.dump(responses, handle)