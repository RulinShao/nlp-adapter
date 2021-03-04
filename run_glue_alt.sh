export GLUE_DIR=/home/hh239/GLUE-baselines/glue_data/
export TASK_NAME=cola
export WANDB_PROJECT=distilbert
export MODEL_NAME=distilbert-base-uncased

#python run_glue_alt.py \
python -m torch.distributed.launch \
  --nproc_per_node 8 run_glue_alt.py \
  --model_name_or_path $MODEL_NAME \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 1e-4 \
  --num_train_epochs 20.0 \
  --output_dir ./output/$MODEL_NAME/$TASK_NAME/ \
  --overwrite_output_dir \
  --train_adapter \
  --adapter_config pfeiffer


# bert-base-uncased
# distilbert-base-uncased
# roberta-base