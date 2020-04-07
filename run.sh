python main.py --task_name TREC --do_train --do_eval --do_lower_case --data_dir ./wikiqa_data/ --bert_model /root/workspace/pretrained_models/uncased_L-12-H-768_A-12/  --max_seq_length 128 --train_batch_size 16 --learning_rate 9e-6 --num_train_epochs 2.0 --output_dir ./wiki_base_output
#python main.py --task_name TREC --do_train --do_eval --do_lower_case --data_dir ./trec_data/ --bert_model /root/workspace/pretrained_models/uncased_L-12-H-768_A-12/  --max_seq_length 128 --train_batch_size 16 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ./base_output2
#python main.py --task_name TREC --do_train --do_eval --do_lower_case --data_dir ./trec_data/ --bert_model /root/workspace/pretrained_models/uncased_L-12-H-768_A-12/  --max_seq_length 256 --train_batch_size 4 --learning_rate 6e-5 --num_train_epochs 3.0 --output_dir ./base_output3
