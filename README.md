# Install the required packages with the below command
pip install -r requirements.txt

# For bert based exps
python run_entity.py    --do_train   --do_eval      --eval_test     --task conll03   --data_dir data/conll03  --model bert-base-cased    --output_dir out/conll03

python run_entity.py   --do_train    --do_eval      --eval_test     --task bc5cdr     --data_dir data/bc5cdr  --model dmis-lab/biobert-base-cased-v1.1    --output_dir out/bc5cdr

python run_entity.py   --do_train   --do_eval      --eval_test    --task twitter     --data_dir data/twitter  --model bert-base-cased   --output_dir out/twitter

python run_entity.py   --do_train   --do_eval      --eval_test    --task wiki     --data_dir data/wiki  --model bert-base-cased    --output_dir out/wiki


# For roberta based exps
python run_entity.py    --do_train   --do_eval      --eval_test     --task conll03     --data_dir  data/conll03  --model roberta-base  --output_dir out/rb_conll03

python run_entity.py   --do_train   --do_eval      --eval_test    --task twitter     --data_dir data/twitter  --model roberta-base    --output_dir out/rb_twitter

python run_entity.py   --do_train   --do_eval      --eval_test    --task wiki     --data_dir data/wiki  --model roberta-base  --output_dir out/rb_wiki

