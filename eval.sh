#!/bin/bash
PROBLEM=num_to_text
MODEL=transformer
WORKER_GPU=2
HPARAMS_SET=transformer_base_bs94_lrc1_do4_f

USR_DIR=.
DATA_DIR=$HOME/t2t_data
TMP_DIR=/tmp/t2t_datagen
TRAIN_DIR=$HOME/t2t_train/$PROBLEM/$MODEL-$HPARAMS_SET

DECODE_FILE=$DATA_DIR/decode_this.txt
DECODE_TO_FILE=$DATA_DIR/decode_result.txt
echo "1 2 3 4 5 6" > $DECODE_FILE
echo "1 20 3 40 5 60" >> $DECODE_FILE
echo "1 2 30 4 50 6" >> $DECODE_FILE
echo "1001 2 3003 4 5005 6" >> $DECODE_FILE
echo "1,2,3,4,5,6" >> $DECODE_FILE
echo "1,20,3,40,5,60" >> $DECODE_FILE
echo "1,2,30,4,50,6" >> $DECODE_FILE
echo "1001,2,3003,4,5005,6 4 223 4553 1 0 0" >> $DECODE_FILE
echo "1001,2,3003,4,5005,6 , 43 ,1 ,21 3" >> $DECODE_FILE
echo "1001,2,3003,4,5005,6" >> $DECODE_FILE
echo "1001 m. vasario 3 d.,2,3003,4,5005,6" >> $DECODE_FILE
echo "101-11-05,2,3003,4,5005,6" >> $DECODE_FILE



BEAM_SIZE=10
ALPHA=0.6

t2t-decoder \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS_SET \
  --output_dir=$TRAIN_DIR \
  --t2t_usr_dir=$USR_DIR \
  --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \
  --decode_from_file=$DECODE_FILE \
  --decode_to_file=$DECODE_TO_FILE
