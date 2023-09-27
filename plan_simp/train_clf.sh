#!/bin/bash

CLF_TYPE="$1"

if [[ "$CLF_TYPE" = "baseline" ]]
then
  python plan_simp/scripts/train_clf.py \
  --train_file=$2 --val_file=$3 --x_col=complex \
  --y_col=label --batch_size=64 --hidden_dropout_prob 0.1 \
  --learning_rate=1e-5 --ckpt_metric=val_macro_f1 \
  --reading_lvl=s_level --src_lvl=c_level
else
  python plan_simp/scripts/train_clf.py \
  --train_file=$2 \
  --val_file=$3 \
  --x_col=complex \
  --y_col=label \
  --batch_size=64 \
  --learning_rate=1e-5 \
  --ckpt_metric=val_macro_f1 \
  --add_context \
  --context_doc_id=pair_id  \
  --context_dir=$4 \
  --context_window=13 \
  --doc_pos_embeds \
  --simple_context_doc_id=pair_id \
  --simple_context_dir=$5 \
  --reading_lvl=s_level
  fi
