import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
import os
from cartography import run_cartography
import re

def save_load_selected_samples(dataset, icl_strategy, strategy_type, label, seed, generation):
    path_to_save = os.path.join('ICL_for_gen', 'datasets', dataset, 'collected_data_llama', str(seed), icl_strategy)
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    samples = None
    if os.path.exists(os.path.join(path_to_save, f'selected_seeds_{strategy_type}_{str(label)}_{generation}.csv')):
        samples = pd.read_csv(os.path.join(path_to_save, f'selected_seeds_{strategy_type}_{str(label)}_{generation}.csv'))
    return samples

def get_random_examples_target_label_only(df, label, NO_SHOTS_PER_LABEL, seed):
    examples = df[df['label'] == label].sample(n=NO_SHOTS_PER_LABEL, random_state=seed)
    return examples
    
def get_random_examples_uniform(df, label, NO_SHOTS_PER_LABEL, seed):
    examples = df[df['label'] == label].sample(n=NO_SHOTS_PER_LABEL, random_state=seed)
    examples_rest = df[df['label'] != label].groupby('label', group_keys=False).apply(lambda x: x.sample(NO_SHOTS_PER_LABEL, random_state=seed))
    return pd.concat([examples, examples_rest], axis=0)

def format_examples(dataset, examples, dataset_to_labels_dct) -> str:
    format_template = "{text} => {label}\n"
    final_str = ''
    for index, example in examples.iterrows():
        str_label = dataset_to_labels_dct[dataset][example['label']]
        final_str += format_template.format(text=example['text'], label=str_label)
    return final_str

def format_examples_mnli(dataset, examples, dataset_to_labels_dct) -> str:
    format_template = "Premise: '{premise}' : Hypothesis: '{hypothesis}' => {label}\n"
    final_str = ''
    for index, example in examples.iterrows():
        str_label = dataset_to_labels_dct[dataset][example['label']]
        if 'text' in examples.columns:
            pattern = r'Premise: (.*?) Hypothesis: (.*)'
            match = re.search(pattern, example['text'])
            premise = match.group(1)
            hypothesis = match.group(2)
            final_str += format_template.format(premise=premise, hypothesis=hypothesis, label=str_label)
        else:
            final_str += format_template.format(premise=example['premise'], hypothesis=example['hypothesis'], label=str_label)
    return final_str

def get_embs_for_sents(df_pd, sent_model) -> dict:
    sents_dct = {}
    emb_dct = {}

    for dct in df_pd.to_dict('records'):
        if dct['label'] in sents_dct:
            sents_dct[dct['label']].append(dct['text'])
        else:
            sents_dct[dct['label']] = [dct['text']]
            
    for label in sents_dct.keys():
        emb_dct[label] = {'emb': sent_model.encode(sents_dct[label], show_progress_bar=False), 'sent': sents_dct[label]}
    return emb_dct

def calculate_outliers(df_pd, sent_model) -> dict:
    embs_dct = get_embs_for_sents(df_pd, sent_model)
    mean_dct = {}
    pandas_dct = {'label': [], 'distance': [], 'text': []}
    
    #calculate mean vector per label
    for label in embs_dct:
        mean_dct[label] = embs_dct[label]['emb'].mean(axis=0)
        
    #calculate distance from the mean vector per label
    for label in embs_dct:
        mean_emb = mean_dct[label]
        for (sent_emb, sent) in zip(embs_dct[label]['emb'], embs_dct[label]['sent']):
            dist = np.linalg.norm(mean_emb - sent_emb)
            pandas_dct['label'].append(label)
            pandas_dct['distance'].append(dist)
            pandas_dct['text'].append(sent)                        
    return pd.DataFrame.from_dict(pandas_dct)

def get_synth_dis_examples_target_label_only(df_pd, label, NO_SHOTS_PER_LABEL, sent_model):
    df_outliers = calculate_outliers(df_pd, sent_model)
    examples = df_outliers[df_outliers['label'] == label].sort_values(['distance'], ascending=False).head(NO_SHOTS_PER_LABEL)
    return examples


def get_similar_examples_gen(df, label, NO_SHOTS_PER_LABEL, sent_model, seed, target_label_only=True):
    selected_samples = []
    classes = df['label']
    classes = np.unique(np.array(classes))
    for cls in classes:
        TO_SELECT = NO_SHOTS_PER_LABEL
        if cls != label:
            if target_label_only:
                continue
        examples = df[df['label'] == cls]
        embds_dict = get_embs_for_sents(examples, sent_model)
        indices = np.arange(examples['label'].shape[0])
        np.random.seed(seed)
        np.random.shuffle(indices)
        indices = indices[:1].tolist()
        features = embds_dict[cls]['emb']
        for _ in range(TO_SELECT - 1):
            sim = np.mean(cos_sim(features[indices], features), axis=0).argsort()[::-1]
            for index in sim:
                if index not in indices:
                    indices.append(index)
                    break
        selected_samples.append(examples.iloc[indices])
    selected_seeds = pd.concat(selected_samples)
    return selected_seeds


def get_diverse_examples_gen(df, label, NO_SHOTS_PER_LABEL, sent_model, seed, target_label_only=True):
    selected_samples = []
    classes = df['label']
    classes = np.unique(np.array(classes))
    for cls in classes:
        TO_SELECT = NO_SHOTS_PER_LABEL
        if cls != label:
            if target_label_only:
                continue
        examples = df[df['label'] == cls]
        embds_dict = get_embs_for_sents(examples, sent_model)
        indices = np.arange(examples['label'].shape[0])
        np.random.seed(seed)
        np.random.shuffle(indices)
        indices = indices[:1].tolist()
        features = embds_dict[cls]['emb']
        for _ in range(TO_SELECT - 1):
            sim = np.mean(cos_sim(features[indices], features), axis=0).argsort()
            for index in sim:
                if index not in indices:
                    indices.append(index)
                    break
        selected_samples.append(examples.iloc[indices])
    selected_seeds = pd.concat(selected_samples)
    return selected_seeds


def get_similar_examples_para(df, label, NO_SHOTS_PER_LABEL, sent_model, seed_sentence, target_label_only=True):
    selected_samples = []
    classes = df['label']
    classes = np.unique(np.array(classes))
    seed_sentence_rep = get_embs_for_sents(seed_sentence, sent_model)[label]['emb']
    for cls in classes:
        TO_SELECT = NO_SHOTS_PER_LABEL
        if cls != label:
            if target_label_only:
                continue
        examples = df[df['label'] == cls]
        embds_dict = get_embs_for_sents(examples, sent_model)
        features = embds_dict[cls]['emb']
        sim = np.mean(cos_sim(seed_sentence_rep, features), axis=0).argsort()[::-1]
        indices = sim[:TO_SELECT]
        selected_samples.append(examples.iloc[indices])
    selected_seeds = pd.concat(selected_samples)
    return selected_seeds


def get_diverse_examples_para(df, label, NO_SHOTS_PER_LABEL, sent_model, seed_sentence, target_label_only=True):
    selected_samples = []
    classes = df['label']
    classes = np.unique(np.array(classes))
    seed_sentence_rep = get_embs_for_sents(seed_sentence, sent_model)[label]['emb']
    for cls in classes:
        TO_SELECT = NO_SHOTS_PER_LABEL
        if cls != label:
            if target_label_only:
                continue
        examples = df[df['label'] == cls]
        embds_dict = get_embs_for_sents(examples, sent_model)
        features = embds_dict[cls]['emb']
        sim = np.mean(cos_sim(seed_sentence_rep, features), axis=0).argsort()
        indices = sim[:TO_SELECT]
        selected_samples.append(examples.iloc[indices])
    selected_seeds = pd.concat(selected_samples)
    return selected_seeds

def load_cartography_metrics(dataset, seed):
    training_metrics_path = os.path.join('ICL_for_gen', 'datasets', dataset, 'collected_data_llama', str(seed), 'cartography')
    if not os.path.exists(training_metrics_path):
        os.makedirs(training_metrics_path)
    training_metrics_path = os.path.join('ICL_for_gen', 'datasets', dataset, 'collected_data_llama', str(seed), 'cartography', 'training_metrics.csv')
    if not os.path.exists(training_metrics_path):
        training_metrics = run_cartography(dataset, seed)
        training_metrics.to_csv(training_metrics_path)
    else:
        training_metrics = pd.read_csv(training_metrics_path)
    return training_metrics


def get_cartography_easy_samples(df, label, NO_SHOTS_PER_LABEL, seed, dataset, target_label_only=True):
    training_metrics = load_cartography_metrics(dataset, seed)
    selected_samples = []
    classes = df['label']
    classes = np.unique(np.array(classes))
    for cls in classes:
        TO_SELECT = NO_SHOTS_PER_LABEL
        if cls != label:
            if target_label_only:
                continue
        examples = df[df['label'] == cls]
        merged_df = examples.merge(training_metrics, how='left', left_on='text', right_on='guid')
        merged_dict = merged_df.to_dict('records')
        easy_samples = sorted(merged_dict, key = lambda x: (x['confidence'], -x['variability']), reverse=True)
        selected_samples.append(pd.DataFrame(easy_samples[:TO_SELECT]))
    selected_seeds = pd.concat(selected_samples)
    selected_seeds['label'] = selected_seeds['label_x']
    selected_seeds = selected_seeds[['text', 'label']].reset_index()
    return selected_seeds

def get_cartography_hard_samples(df, label, NO_SHOTS_PER_LABEL, seed, dataset, target_label_only=True):
    training_metrics = load_cartography_metrics(dataset, seed)
    selected_samples = []
    classes = df['label']
    classes = np.unique(np.array(classes))
    for cls in classes:
        TO_SELECT = NO_SHOTS_PER_LABEL
        if cls != label:
            if target_label_only:
                continue
        examples = df[df['label'] == cls]
        merged_df = examples.merge(training_metrics, how='left', left_on='text', right_on='guid')
        merged_dict = merged_df.to_dict('records')
        hard_samples = sorted(merged_dict, key = lambda x: (-x['confidence'], x['variability']), reverse=True)
        selected_samples.append(pd.DataFrame(hard_samples[:TO_SELECT]))
    selected_seeds = pd.concat(selected_samples)
    selected_seeds['label'] = selected_seeds['label_x']
    selected_seeds = selected_seeds[['text', 'label']].reset_index()
    return selected_seeds

def get_cartography_easy_ambig_samples(df, label, NO_SHOTS_PER_LABEL, seed, dataset, target_label_only=True):
    training_metrics = load_cartography_metrics(dataset, seed)
    selected_samples = []
    classes = df['label']
    classes = np.unique(np.array(classes))
    for cls in classes:
        TO_SELECT = NO_SHOTS_PER_LABEL
        if cls != label:
            if target_label_only:
                continue
        examples = df[df['label'] == cls]
        merged_df = examples.merge(training_metrics, how='left', left_on='text', right_on='guid')
        merged_dict = merged_df.to_dict('records')
        confidence = [x['confidence'] for x in merged_dict]
        mean = np.mean(confidence)
        std = np.std(confidence)
        easy_ambigous_samples = [x for x in merged_dict if x['confidence'] >= mean - std]
        np.random.seed(seed)
        np.random.shuffle(easy_ambigous_samples)
        selected_samples.append(pd.DataFrame(easy_ambigous_samples[:TO_SELECT]))
    selected_seeds = pd.concat(selected_samples)
    selected_seeds['label'] = selected_seeds['label_x']
    selected_seeds = selected_seeds[['text', 'label']].reset_index()
    return selected_seeds

def get_cartography_forgotten_samples(df, label, NO_SHOTS_PER_LABEL, seed, dataset, target_label_only=True):
    training_metrics = load_cartography_metrics(dataset, seed)
    selected_samples = []
    classes = df['label']
    classes = np.unique(np.array(classes))
    for cls in classes:
        TO_SELECT = NO_SHOTS_PER_LABEL
        if cls != label:
            if target_label_only:
                continue
        examples = df[df['label'] == cls]
        merged_df = examples.merge(training_metrics, how='left', left_on='text', right_on='guid')
        merged_dict = merged_df.to_dict('records')
        forgotten_samples = sorted(merged_dict, key = lambda x: (x['forgetfulness'], -x['correctness']), reverse=True)
        selected_samples.append(pd.DataFrame(forgotten_samples[:TO_SELECT]))
    selected_seeds = pd.concat(selected_samples)
    selected_seeds['label'] = selected_seeds['label_x']
    selected_seeds = selected_seeds[['text', 'label']].reset_index()
    return selected_seeds

def get_cartography_learned_samples(df, label, NO_SHOTS_PER_LABEL, seed, dataset, target_label_only=True):
    training_metrics = load_cartography_metrics(dataset, seed)
    selected_samples = []
    classes = df['label']
    classes = np.unique(np.array(classes))
    for cls in classes:
        TO_SELECT = NO_SHOTS_PER_LABEL
        if cls != label:
            if target_label_only:
                continue
        examples = df[df['label'] == cls]
        merged_df = examples.merge(training_metrics, how='left', left_on='text', right_on='guid')
        merged_dict = merged_df.to_dict('records')
        learned_samples = sorted(merged_dict, key = lambda x: (x['correctness'], -x['forgetfulness']), reverse=True)
        selected_samples.append(pd.DataFrame(learned_samples[:TO_SELECT]))
    selected_seeds = pd.concat(selected_samples)
    selected_seeds['label'] = selected_seeds['label_x']
    selected_seeds = selected_seeds[['text', 'label']].reset_index()
    return selected_seeds