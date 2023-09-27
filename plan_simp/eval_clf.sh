#!/bin/bash

CLF_TYPE="$1"

if [[ "$CLF_TYPE" = "baseline" ]]
then
  python plan_simp/scripts/eval_clf.py "$2" "$3" \
  --reading_lvl=s_level --src_lvl=c_level
else
  python plan_simp/scripts/eval_clf.py "$2" "$3" \
  --add_context=True \
  --context_dir=$4 \
  --context_doc_id=pair_id \
  --simple_context_dir=$5 \
  --simple_context_doc_id=pair_id \
  --reading_lvl=s_level \
  --by_level
fi
