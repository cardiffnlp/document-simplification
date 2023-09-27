#!/bin/bash

python plan_simp/scripts/encode_contexts.py \
	--data=$1 \
	--x_col=$3 \
	--id_col=pair_id \
	--save_dir=$2/