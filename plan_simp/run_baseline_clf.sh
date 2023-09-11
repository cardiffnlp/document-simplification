python plan_simp/scripts/train_clf.py \
  --train_file=$1 --val_file=$2 --x_col=complex \
  --y_col=label --batch_size=64 \
  --learning_rate=1e-5 --ckpt_metric=val_macro_f1
