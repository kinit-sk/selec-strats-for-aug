#!/bin/bash
methods=('only' 'uniform')
seeds=(0 1 2)
strategies=('baseline', 'synth_dis', 'cos_sim' 'cos_div' 'cartography_easy' 'cartography_hard' 'cartography_easy_ambig' 'forgetting_most' 'forgetting_least')
generations=('para' 'gen')
models=('google-bert/bert-base-uncased' 'FacebookAI/roberta-base')

for generation in ${generations[@]}; do
    for seed in ${seeds[@]}; do
        for strategy in ${strategies[@]}; do
            for method in ${methods[@]}; do
                for model in ${models[@]}; do
                    python ICL_for_gen/finetuning_scripts/finetune_for_ood.py --no_epochs 50 --seed $seed --method_gen $generation --icl_strategy $strategy --base_model_type $model --batch_size 64 --batch_size_eval 512 --repeat 10 --dataset $1 --method $method
                done
            done
        done
    done
done