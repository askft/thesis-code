# coding=utf-8

import os

os.system("python run_re.py"
          " --task_name=gad"
          " --do_train=true"
          " --do_eval=true"
          " --do_predict=true"
          " --vocab_file=./model/vocab.txt"
          " --bert_config_file=./model/bert_config.json"
          " --init_checkpoint=./model/model.ckpt-1000000"
          " --max_seq_length=128"
          " --train_batch_size=32"
          " --learning_rate=2e-5"
          " --num_train_epochs=10.0"
          " --do_lower_case=false"
          " --data_dir=./REdata/GAD/1"
          " --output_dir=./output/"
          )
