#$ -l m_mem_free=50G
#$ -j y
#$ -N bert_fourth_task
source /opt/rh/rh-python36/enable
source ../py3env/bin/activate
export BERT_BASE_DIR=uncased_L-12_H-768_A-12
export PDATA=data
python run_classifier.py \
  --task_name=MRPC \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$PDATA \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=./bert_output/
