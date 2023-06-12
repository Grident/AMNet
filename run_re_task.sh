#!/usr/bin/env bash

DATASET_NAME="MRE"
BERT_NAME="bert-base-uncased"

#train & test
CUDA_VISIBLE_DEVICES=2 python -u run.py \
        --dataset_name=${DATASET_NAME} \
        --bert_name=${BERT_NAME} \
        --num_epochs=10 \
        --batch_size=16 \
        --lr=3e-5 \
        --warmup_ratio=0.06 \
        --eval_begin_epoch=1 \
        --seed=1234 \
        --do_train \
        --max_seq=80 \
        --use_prompt \
        --use_contrastive\
        --use_matching\
        --prompt_len=12 \
        --sample_ratio=1.0 \
        --save_path='ckpt/re'

#only test
# CUDA_VISIBLE_DEVICES=2 python -u run.py \
#         --dataset_name="MRE" \
#         --bert_name="bert-base-uncased" \
#         --seed=1234 \
#         --only_test \
#         --max_seq=80 \
#         --use_prompt \
#         --use_contrastive\
#         --use_matching\
#         --prompt_len=12 \
#         --sample_ratio=1.0 \
#         --load_path='ckpt/re/best_model.pth' 



