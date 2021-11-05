EXP_ID='0'
DATASET_NAME='f30k'
DATA_PATH='/DATA/data/'${DATASET_NAME}
VOCAB_PATH='/DATA/data/vocab'

CUDA_VISIBLE_DEVICES=0 python3 train.py \
  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH}\
  --logger_name runs/${DATASET_NAME}_butd_region_bigru_${EXP_ID}/log --model_name runs/${DATASET_NAME}_butd_region_bigru_${EXP_ID} \
  --num_epochs=25 --lr_update=15 --learning_rate=.0005 --precomp_enc_type basic --workers 10 \
  --log_step 200 --embed_size 1024 --vse_mean_warmup_epochs 1 \
  --drop_random --drop_mask --drop_remove --drop_cap --drop_img
