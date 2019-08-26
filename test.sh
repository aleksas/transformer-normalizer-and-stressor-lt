#!/bin/bash
PROBLEM=num_to_text
MODEL=transformer
WORKER_GPU=2
HPARAMS_SET=transformer_base_bs94_lrc1_do4_f

USR_DIR=.
DATA_DIR=$HOME/t2t_data
TMP_DIR=/tmp/t2t_datagen
TRAIN_DIR=$HOME/t2t_train/$PROBLEM/$MODEL-$HPARAMS_SET

ls $TRAIN_DIR

INPUT_FILE=$TMP_DIR/num2text-p8-v7/num2text_num_p8_v7.txt
OUTPUT_FILE=$TMP_DIR/num2text-p8-v7/num2text_txt_p8_v7.txt
PREDICTION_FILE=$TMP_DIR/num2text-p8-v7/result_eval.txt

DECODE_FROM_FILE=$INPUT_FILE
DECODE_TO_FILE=$PREDICTION_FILE

BEAM_SIZE=8
ALPHA=0.6


num_files=2
total_lines=$(wc -l <${DECODE_FROM_FILE})
((lines_per_file = (total_lines + num_files - 1) / num_files))

# Split the actual file, maintaining lines.

split --lines=${lines_per_file} ${DECODE_FROM_FILE} ${DECODE_FROM_FILE}.

export CUDA_VISIBLE_DEVICES=0

t2t-decoder \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS_SET \
  --output_dir=$TRAIN_DIR \
  --t2t_usr_dir=$USR_DIR \
  --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \
  --decode_from_file=$DECODE_FROM_FILE.aa \
  --decode_to_file=$DECODE_TO_FILE.aa &

export CUDA_VISIBLE_DEVICES=1

t2t-decoder \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS_SET \
  --output_dir=$TRAIN_DIR \
  --t2t_usr_dir=$USR_DIR \
  --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \
  --decode_from_file=$DECODE_FROM_FILE.ab \
  --decode_to_file=$DECODE_TO_FILE.ab

# wait for 
tail --pid=$(pgrep t2t-decoder) -f /dev/null
tail --pid=$(pgrep t2t-decoder) -f /dev/null

cat $DECODE_TO_FILE.aa $DECODE_TO_FILE.ab > $DECODE_TO_FILE

python evaluate_differences.py -i $INPUT_FILE -o $OUTPUT_FILE -p $PREDICTION_FILE

zip diff.zip diff_*.txt
rm diff_*.txt
