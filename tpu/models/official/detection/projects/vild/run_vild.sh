BATCH_SIZE=1

# [50/152]
RESNET_DEPTH=50

VILD_DIR=/Users/gsq/Desktop/tpu/models/official/detection/projects/vild
#!******************
DATA_DIR=${VILD_DIR}/1data_dir
MODEL_DIR=${VILD_DIR}/1mode_dir/
#//echo ${MODEL_DIR}


WEIGHTS_DIR=/Users/gsq/Desktop/tpu/models/official/detection/projects/vild/1weig_dir

EVAL_FILE_PATTERN="${DEST_DIR}/val*"
VAL_JSON_FILE="${DATA_DIR}/lvis_v1_val.json"

RARE_MASK_PATH="${WEIGHTS_DIR}/lvis_rare_masks.npy"
CLASSIFIER_WEIGHT_PATH="${WEIGHTS_DIR}/clip_synonym_prompt.npy"

CONFIG_FILE="/Users/gsq/Desktop/tpu/models/official/detection/projects/vild/configs/vild_resnet.yaml"
#//echo ${CONFIG_FILE}

#//echo ${CLASSIFIER_WEIGHT_PATH}
#//echo ${RARE_MASK_PATH}

cd ../../
python3 main.py \
  --model="vild" \
  --model_dir="${MODEL_DIR?}" \
  --mode=eval \
  --use_tpu=False \
  --config_file="${CONFIG_FILE?}" \
  --params_override="{\
    resnet: {resnet_depth: ${RESNET_DEPTH?}},\
    predict: {predict_batch_size: ${BATCH_SIZE?}}, \
    eval: {eval_batch_size: ${BATCH_SIZE?}, \
           val_json_file: ${VAL_JSON_FILE?}, \
           eval_timeout: 10,
           eval_file_pattern: ${EVAL_FILE_PATTERN?} \
           }, \
    frcnn_head: {classifier_weight_path: ${CLASSIFIER_WEIGHT_PATH?}}, \
    frcnn_class_loss: {rare_mask_path: ${RARE_MASK_PATH?}},
    postprocess: {rare_mask_path: ${RARE_MASK_PATH?}},
    }"
# postprocess: {rare_mask_path: ${RARE_MASK_PATH?}}\
# eval _ { num_steps_per_eval: 1 },
cd projects/vild/