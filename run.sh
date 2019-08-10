#!/bin/bash
PROBLEM=num_to_text
MODEL=transformer
WORKER_GPU=2
HPARAMS_SET=transformer_base_bs94_lrc1_do4_f

USR_DIR=.
DATA_DIR=$HOME/t2t_data
TMP_DIR=/tmp/t2t_datagen
TRAIN_DIR=$HOME/t2t_train/$PROBLEM/$MODEL-$HPARAMS_SET

pkill -15 t2t-trainer

tensorboard --logdir $TRAIN_DIR/../.. &

rm -r $DATA_DIR $TMP_DIR $TRAIN_DIR
mkdir -p $DATA_DIR $TMP_DIR

t2t-trainer \
    --generate_data \
    --problem=$PROBLEM \
    --data_dir=$DATA_DIR \
    --tmp_dir=$TMP_DIR \
    --output_dir=$TRAIN_DIR \
    --t2t_usr_dir=$USR_DIR \
    --hparams_set=$HPARAMS_SET \
    --model=$MODEL \
    --worker_gpu=$WORKER_GPU
