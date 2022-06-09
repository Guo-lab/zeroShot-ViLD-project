DATA_DIR=1data_dir
DEST_DIR=1dest_dir

#//echo ${DATA_DIR}

VAL_JSON="${DATA_DIR}/lvis_v1_val.json"
#//echo ${VAL_JSON}

## source *.sh instead of ./*.sh 
#//echo ${DEST_DIR}
#//cd "${DEST_DIR}"
#//pwd 




# absl.flags._exceptions.UnparsedFlagAccessError: Trying to access flag --dest_dir before flags were parsed.
# https://tech-related.com/p/MJiKXQvfBn
# https://github.com/tensorflow/models/issues/4794
python3 preprocessing/create_lvis_tf_record.py \
  --image_dir="${DATA_DIR}" \
  --json_path="${VAL_JSON}" \
  --dest_dir="${DEST_DIR}" \
  --include_mask=True \
  --split='val' \
  --num_parts=100 \
  --max_num_processes=100