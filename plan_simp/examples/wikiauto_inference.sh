: '
This script will illustrate an example of how the pretrained PG_Dyn system can be used to perform out-of-domain inference.

In this example we use the validation set of Wiki-auto, with models trained on Newsela-auto. 
Because Wikipedia does not have different reading levels we manually specify a constant target level of 3 (the second simplest).
'

# encode sentence-level context embeddings to be used by the planner
python plan_simp/scripts/encode_contexts.py \
	--data=../wiki_auto/wikiauto_docs_test.csv \
	--save_dir=simple_fake_context_dir/ \
	--x_col=simple

# run inference with pretrained models
#python plan_simp/scripts/generate.py dynamic \
#	--clf_model_ckpt=liamcripwell/pgdyn-plan \
#  --model_ckpt=liamcripwell/pgdyn-simp  \
#  --test_file=examples/wikiauto_sents_valid.csv \
#	--doc_id_col=pair_id \
#  --context_doc_id=pair_id \
#  --context_dir=fake_context_dir \
#	--reading_lvl=3 \
#  --out_file=test_out.csv

# evaluate simplification performance
#python plan_simp/scripts/eval_simp.py \
#  --input_data=examples/wikiauto_docs_valid.csv \
#  --output_data=test_out.csv \
#  --x_col=complex \
#  --r_col=simple \
#  --y_col=pred \
#  --doc_id_col=pair_id \
#  --prepro=True \
#  --sent_level=True
