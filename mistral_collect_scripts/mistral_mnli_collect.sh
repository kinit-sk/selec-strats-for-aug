#!/bin/bash
methods=('uniform' 'only')
seeds=(0 1 2)
# strategies=('cos_sim' 'cos_div' 'cartography_easy' 'cartography_hard' 'cartography_easy_ambig' 'forgetting_most' 'forgetting_least')
# strategies=('cartography_easy' 'cartography_hard' 'cartography_easy_ambig' 'forgetting_most' 'forgetting_least')
strategies=('baseline' 'random' 'cos_sim')
datasets=('mnli')

for dataset in ${datasets[@]}; do
    for seed in ${seeds[@]}; do
        for strategy in ${strategies[@]}; do
            for method in ${methods[@]}; do
                python ICL_for_gen/mistral_collect_scripts/collect_data_mistral_v3.py --dataset $dataset --method $method --icl_strategy $strategy --seed $seed
            done
        done
    done
done