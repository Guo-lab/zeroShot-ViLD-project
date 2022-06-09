# Open-Vocabulary Detection via Vision and Language Knowledge Distillation
• [Paper](https://arxiv.org/abs/2104.13921) • [Colab](https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/models/official/detection/projects/vild/ViLD_demo.ipynb)

<p style="text-align:center;"><img src="https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/vild/assets/new_teaser.png" alt="teaser" width="500"/></p>

Xiuye Gu, Tsung-Yi Lin, Weicheng Kuo, Yin Cui,
[Open-Vocabulary Detection via Vision and Language Knowledge Distillation](https://arxiv.org/abs/2104.13921).

This repo contains the colab demo, code, and pretrained checkpoints for our open-vocabulary detection method, ViLD (**Vi**sion and **L**anguage **D**istillation).

Open-vocabulary object detection detects objects described by arbitrary text inputs. The fundamental challenge is the availability of training data. Existing object detection datasets only contain hundreds of categories, and it is costly to scale further. To overcome this challenge, we propose ViLD. Our method distills the knowledge from a pretrained open-vocabulary image classification model (teacher) into a two-stage detector (student). Specifically, we use the teacher model to encode category texts and image regions of object proposals. Then we train a student detector, whose region embeddings of detected boxes are aligned with the text and image embeddings inferred by the teacher. 

We benchmark on LVIS by holding out all rare categories as novel categories not seen during training. ViLD obtains 16.1 mask APr, even outperforming the supervised counterpart by 3.8 with a ResNet-50 backbone. The model can directly transfer to other datasets without finetuning, achieving 72.2 AP50, 36.6 AP and 11.8 AP on PASCAL VOC, COCO and Objects365, respectively. On COCO, ViLD outperforms previous SOTA by 4.8 on novel AP and 11.4 on overall AP.

The figure below shows an overview of ViLD's architecture.
![architecture overview](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/vild/assets/new_overview_new_font.png)


# Colab Demo
In this [colab](https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/models/official/detection/projects/vild/ViLD_demo.ipynb) or this [jupyter notebook](./ViLD_demo.ipynb), we created a demo with two examples. You can also try your own images and specify the categories you want to detect. 


# Getting Started
## Prerequisite
* Install [TensorFlow](https://www.tensorflow.org/install).
* Install the packages in [`requirements.txt`](./requirements.txt).


## Data preprocessing
1. Download and unzip the [LVIS v1.0](https://www.lvisdataset.org/dataset) validation sets to `DATA_DIR`.

The `DATA_DIR` should be organized as below:

```
DATA_DIR
+-- lvis_v1_val.json
+-- val2017
|   +-- ***.jpg
|   +-- ...
```

2. Create tfrecords for the validation set (adjust `max_num_processes` if needed; specify `DEST_DIR` to the tfrecords output directory):

```shell
DATA_DIR=[DATA_DIR]
DEST_DIR=[DEST_DIR]
VAL_JSON="${DATA_DIR}/lvis_v1_val.json"
python3 preprocessing/create_lvis_tf_record.py \
  --image_dir="${DATA_DIR}" \
  --json_path="${VAL_JSON}" \
  --dest_dir="${DEST_DIR}" \
  --include_mask=True \
  --split='val' \
  --num_parts=100 \
  --max_num_processes=100
```

## Trained checkpoints
| Method        | Backbone     | Distillation weight | APr   |  APc |  APf | AP   | config | ckpt |
|:------------- |:-------------| -------------------:| -----:|-----:|-----:|-----:|--------|------|
| ViLD          | ResNet-50    | 0.5                 | 16.6  | 19.8 | 28.2 | 22.5 | [vild_resnet.yaml](./configs/vild_resnet.yaml) |[ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/vild/pretrained_ckpts/resnet50_vild.tar.gz)|
| ViLD-ensemble | ResNet-50    | 0.5                 |  18	 | 24.7	| 30.6 | 25.9 | [vild_resnet.yaml](./configs/vild_resnet.yaml) |[ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/vild/pretrained_ckpts/resnet50_vild_ensemble.tar.gz)|
| ViLD          | ResNet-152   | 1.0                 | 19.6	 | 21.6	| 28.5 | 24.0 | [vild_ensemble_resnet.yaml](./configs/vild_ensemble_resnet.yaml) |[ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/vild/pretrained_ckpts/resnet152_vild.tar.gz)|
| ViLD-ensemble | ResNet-152   | 2.0                 | 19.2	 | 24.8	| 30.8 | 26.2 | [vild_ensemble_resnet.yaml](./configs/vild_ensemble_resnet.yaml) |[ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/vild/pretrained_ckpts/resnet152_vild_ensemble.tar.gz)|

## Inference
1. Download the [classification weights](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/vild/weights/clip_synonym_prompt.npy) (CLIP text embeddings) and the [binary masks](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/vild/weights/lvis_rare_masks.npy) for rare categories. And put them in `[WEIGHTS_DIR]`.
2. Download and unzip the trained model you want to run inference in `[MODEL_DIR]`.
3. Replace `[RESNET_DEPTH], [MODEL_DIR], [DATA_DIR], [DEST_DIR], [WEIGHTS_DIR], [CONFIG_FILE]` with your values in the script below and run it.

Please refer [getting_started.md](https://github.com/tensorflow/tpu/blob/master/models/official/detection/GETTING_STARTED.md) for more information.

```shell
BATCH_SIZE=1
RESNET_DEPTH=[RESNET_DEPTH]
MODEL_DIR=[MODEL_DIR]
EVAL_FILE_PATTERN="[DEST_DIR]/val*"
VAL_JSON_FILE="[DATA_DIR]/lvis_v1_val.json"
RARE_MASK_PATH="[WEIGHTS_DIR]/lvis_rare_masks.npy"
CLASSIFIER_WEIGHT_PATH="[WEIGHTS_DIR]/clip_synonym_prompt.npy"
CONFIG_FILE="tpu/models/official/detection/projects/vild/configs/[CONFIG_FILE]"
python3 tpu/models/official/detection/main.py \
  --model="vild" \
  --model_dir="${MODEL_DIR?}" \
  --mode=eval \
  --use_tpu=False \
  --config_file="${CONFIG_FILE?}" \
  --params_override="{ resnet: {resnet_depth: ${RESNET_DEPTH?}}, predict: {predict_batch_size: ${BATCH_SIZE?}}, eval: {eval_batch_size: ${BATCH_SIZE?}, val_json_file: ${VAL_JSON_FILE?}, eval_file_pattern: ${EVAL_FILE_PATTERN?} }, frcnn_head: {classifier_weight_path: ${CLASSIFIER_WEIGHT_PATH?}}, postprocess: {rare_mask_path: ${RARE_MASK_PATH?}}}"
```


# License
This repo is under the same license as  [tensorflow/tpu](https://github.com/tensorflow/tpu), see
[license](https://github.com/tensorflow/tpu/blob/master/LICENSE).

# Citation
If you find this repo to be useful to your research, please cite our paper:

```
@article{gu2021open,
  title={Open-Vocabulary Detection via Vision and Language Knowledge Distillation},
  author={Gu, Xiuye and Lin, Tsung-Yi and Kuo, Weicheng and Cui, Yin},
  journal={arXiv preprint arXiv:2104.13921},
  year={2021}
}
```

# Acknowledgements
In this repo, we use [OpenAI's CLIP model](https://github.com/openai/CLIP) as the open-vocabulary image classification model, i.e., the teacher model.

The code is built upon [Cloud TPU detection](https://github.com/tensorflow/tpu/tree/master/models/official/detection).













tpu result:  

2022-06-09 09:17:15.410936: W tensorflow/stream_executor/platform/default/dso_loader.cc:60] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /opt/conda/lib
2022-06-09 09:17:15.410996: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
I0609 09:17:18.255211 140411182167872 main.py:122] Model Parameters: {'anchor': {'anchor_size': 8,
            'aspect_ratios': [1.0, 2.0, 0.5],
            'num_scales': 1},
 'architecture': {'backbone': 'resnet',
                  'feat_distill_weight': 0.5,
                  'filter_distill_boxes_size': 0,
                  'include_mask': True,
                  'mask_target_size': 28,
                  'max_level': 6,
                  'max_num_rois': 300,
                  'min_level': 2,
                  'multilevel_features': 'fpn',
                  'normalize_feat_during_training': True,
                  'num_classes': 1204,
                  'parser': 'vild_parser',
                  'pre_parser': None,
                  'space_to_depth_block_size': 1,
                  'use_bfloat16': False,
                  'visual_feature_dim': 512,
                  'visual_feature_distill': 'vanilla'},
 'batch_norm_activation': {'activation': 'relu',
                           'batch_norm_epsilon': 0.0001,
                           'batch_norm_momentum': 0.997,
                           'batch_norm_trainable': True,
                           'use_sync_bn': True},
 'dropblock': {'dropblock_keep_prob': None, 'dropblock_size': None},
 'enable_summary': False,
 'eval': {'eval_batch_size': 8,
          'eval_dataset_type': 'tfrecord',
          'eval_file_pattern': '/val*',
          'eval_samples': 19809,
          'eval_timeout': 10,
          'min_eval_interval': 5,
          'num_steps_per_eval': 1000,
          'per_category_metrics': False,
          'skip_eval_loss': False,
          'suffix': '',
          'type': 'lvis_box_and_mask',
          'use_json_file': True,
          'val_json_file': '/kaggle/working/tpu/models/official/detection/projects/vild/1data_dir/lvis_v1_val.json'},
 'fpn': {'fpn_feat_dims': 256,
         'use_batch_norm': True,
         'use_separable_conv': False},
 'frcnn_box_loss': {'huber_loss_delta': 1.0},
 'frcnn_class_loss': {'mask_rare': True,
                      'rare_mask_path': '/kaggle/working/tpu/models/official/detection/projects/vild/1weig_dir/lvis_rare_masks.npy'},
 'frcnn_head': {'class_agnostic_bbox_pred': True,
                'classifier_weight_path': '/kaggle/working/tpu/models/official/detection/projects/vild/1weig_dir/clip_synonym_prompt.npy',
                'clip_dim': 512,
                'fc_dims': 1024,
                'normalize_classifier': True,
                'normalize_visual': True,
                'num_convs': 4,
                'num_fcs': 2,
                'num_filters': 256,
                'temperature': 100.0,
                'use_batch_norm': True,
                'use_separable_conv': False},
 'isolate_session_state': False,
 'mask_sampling': {'num_mask_samples_per_image': 128},
 'model_dir': '/kaggle/working/tpu/models/official/detection/projects/vild/1mode_dir/',
 'mrcnn_head': {'class_agnostic_mask_pred': True,
                'num_convs': 4,
                'num_filters': 256,
                'use_batch_norm': True,
                'use_separable_conv': False},
 'nasfpn': {'activation': None,
            'block_fn': 'conv',
            'fpn_feat_dims': 256,
            'init_drop_connect_rate': None,
            'num_repeats': 5,
            'use_separable_conv': False,
            'use_sum_for_combination': False},
 'platform': {'eval_master': None,
              'gcp_project': None,
              'tpu': 'grpc://10.0.0.2:8470',
              'tpu_zone': None},
 'postprocess': {'apply_nms': True,
                 'apply_sigmoid': False,
                 'discard_background': False,
                 'max_total_size': 300,
                 'nms_iou_threshold': 0.5,
                 'nms_version': 'v1',
                 'pre_nms_num_boxes': 1000,
                 'rare_mask_path': '/kaggle/working/tpu/models/official/detection/projects/vild/1weig_dir/lvis_rare_masks.npy',
                 'score_threshold': 0.0,
                 'use_batched_nms': False},
 'predict': {'predict_batch_size': 8},
 'resnet': {'init_drop_connect_rate': None, 'resnet_depth': 50},
 'roi_proposal': {'rpn_min_size_threshold': 0.0,
                  'rpn_nms_threshold': 0.7,
                  'rpn_post_nms_top_k': 1000,
                  'rpn_pre_nms_top_k': 2000,
                  'rpn_score_threshold': 0.0,
                  'test_rpn_min_size_threshold': 0.0,
                  'test_rpn_nms_threshold': 0.7,
                  'test_rpn_post_nms_top_k': 1000,
                  'test_rpn_pre_nms_top_k': 1000,
                  'test_rpn_score_threshold': 0.0,
                  'use_batched_nms': False},
 'roi_sampling': {'bg_iou_thresh_hi': 0.5,
                  'bg_iou_thresh_lo': 0.0,
                  'cascade_iou_thresholds': None,
                  'fg_fraction': 0.25,
                  'fg_iou_thresh': 0.5,
                  'mix_gt_boxes': True,
                  'num_samples_per_image': 512},
 'rpn_box_loss': {'huber_loss_delta': 0.1111111111111111},
 'rpn_head': {'anchors_per_location': None,
              'cast_to_float32': True,
              'num_convs': 2,
              'num_filters': 256,
              'use_batch_norm': True,
              'use_separable_conv': False},
 'rpn_score_loss': {'rpn_batch_size_per_im': 256},
 'spinenet': {'init_drop_connect_rate': None,
              'model_id': '49',
              'use_native_resize_op': False},
 'spinenet_mbconv': {'init_drop_connect_rate': None,
                     'model_id': '49',
                     'se_ratio': 0.2,
                     'use_native_resize_op': False},
 'tpu_job_name': None,
 'train': {'checkpoint': {'path': '', 'prefix': '', 'skip_variables_regex': ''},
           'frozen_variable_prefix': 'frcnn_layer_0/fast_rcnn_head/class-predict',
           'gradient_clip_norm': 0.0,
           'input_partition_dims': None,
           'iterations_per_loop': 100,
           'l2_weight_decay': 4e-05,
           'learning_rate': {'init_learning_rate': 0.32,
                             'learning_rate_levels': [0.032, 0.0032],
                             'learning_rate_steps': [162000, 171000, 175500],
                             'type': 'step',
                             'warmup_learning_rate': 0.0032,
                             'warmup_steps': 1000},
           'losses': 'all',
           'num_cores_per_replica': None,
           'num_shards': 8,
           'optimizer': {'momentum': 0.9, 'type': 'momentum'},
           'pre_parser_dataset': {'dataset_type': 'tfrecord',
                                  'file_pattern': ''},
           'regularization_variable_regex': '.*(kernel|weight):0$',
           'space_to_depth_block_size': 1,
           'total_steps': 180000,
           'train_batch_size': 256,
           'train_dataset_type': 'tfrecord',
           'train_file_pattern': '',
           'transpose_input': True},
 'type': 'vild',
 'use_tpu': True,
 'vild_parser': {'aug_rand_hflip': True,
                 'aug_scale_max': 2.0,
                 'aug_scale_min': 0.1,
                 'copy_paste': False,
                 'mask_crop_size': 112,
                 'max_num_instances': 300,
                 'output_size': [1024, 1024],
                 'regenerate_source_id': False,
                 'rpn_batch_size_per_im': 256,
                 'rpn_fg_fraction': 0.5,
                 'rpn_match_threshold': 0.7,
                 'rpn_unmatched_threshold': 0.3,
                 'skip_crowd_during_training': True}}
I0609 09:17:18.262247 140411182167872 postprocess_ops.py:468] in GenericDetectionGenerator, discard_background is set to False
W0609 09:17:18.306895 140411182167872 module_wrapper.py:138] From /kaggle/working/tpu/models/official/detection/executor/tpu_executor.py:101: The name tf.estimator.tpu.TPUConfig is deprecated. Please use tf.compat.v1.estimator.tpu.TPUConfig instead.

W0609 09:17:18.307259 140411182167872 module_wrapper.py:138] From /kaggle/working/tpu/models/official/detection/executor/tpu_executor.py:106: The name tf.estimator.tpu.InputPipelineConfig is deprecated. Please use tf.compat.v1.estimator.tpu.InputPipelineConfig instead.

W0609 09:17:18.307515 140411182167872 module_wrapper.py:138] From /kaggle/working/tpu/models/official/detection/executor/tpu_executor.py:110: The name tf.estimator.tpu.RunConfig is deprecated. Please use tf.compat.v1.estimator.tpu.RunConfig instead.

W0609 09:17:18.308045 140411182167872 module_wrapper.py:138] From /kaggle/working/tpu/models/official/detection/executor/tpu_executor.py:120: The name tf.estimator.tpu.TPUEstimator is deprecated. Please use tf.compat.v1.estimator.tpu.TPUEstimator instead.

I0609 09:17:18.309241 140411182167872 estimator.py:191] Using config: {'_model_dir': '/kaggle/working/tpu/models/official/detection/projects/vild/1mode_dir/', '_tf_random_seed': None, '_save_summary_steps': 100, '_save_checkpoints_steps': None, '_save_checkpoints_secs': 600, '_session_config': cluster_def {
  job {
    name: "worker"
    tasks {
      key: 0
      value: "10.0.0.2:8470"
    }
  }
}
, '_keep_checkpoint_max': 5, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': None, '_train_distribute': None, '_device_fn': None, '_protocol': None, '_eval_distribute': None, '_experimental_distribute': None, '_experimental_max_worker_delay_secs': None, '_session_creation_timeout_secs': 7200, '_checkpoint_save_graph_def': True, '_service': None, '_cluster_spec': ClusterSpec({'worker': ['10.0.0.2:8470']}), '_task_type': 'worker', '_task_id': 0, '_global_id_in_cluster': 0, '_master': 'grpc://10.0.0.2:8470', '_evaluation_master': 'grpc://10.0.0.2:8470', '_is_chief': True, '_num_ps_replicas': 0, '_num_worker_replicas': 1, '_tpu_config': TPUConfig(iterations_per_loop=100, num_shards=None, num_cores_per_replica=None, per_host_input_for_training=3, tpu_job_name=None, initial_infeed_sleep_secs=None, input_partition_dims=None, eval_training_input_configuration=2, experimental_host_call_every_n_steps=1, experimental_allow_per_host_v2_parallel_get_next=False, experimental_feed_hook=None), '_cluster': <tensorflow.python.distribute.cluster_resolver.tpu.tpu_cluster_resolver.TPUClusterResolver object at 0x7fb39e93e2d0>}
I0609 09:17:18.310059 140411182167872 tpu_context.py:271] _TPUContext: eval_on_tpu True
I0609 09:17:18.310461 140411182167872 checkpoint_utils.py:139] Waiting for new checkpoint at /kaggle/working/tpu/models/official/detection/projects/vild/1mode_dir/
I0609 09:17:18.311290 140411182167872 checkpoint_utils.py:148] Found new checkpoint at /kaggle/working/tpu/models/official/detection/projects/vild/1mode_dir/model.ckpt-180000
I0609 09:17:18.311465 140411182167872 main.py:203] Starting to evaluate.
<dataloader.input_reader.InputFn object at 0x7fb39e93eb10> 2476 /kaggle/working/tpu/models/official/detection/projects/vild/1mode_dir/model.ckpt-180000
<dataloader.input_reader.InputFn object at 0x7fb39e93eb10> 2476 /kaggle/working/tpu/models/official/detection/projects/vild/1mode_dir/model.ckpt-180000
I0609 09:17:18.703426 140411182167872 lvis.py:25] Loading annotations.
I0609 09:17:27.578833 140411182167872 lvis.py:39] Creating index.
I0609 09:17:27.945315 140411182167872 lvis.py:61] Index created.
<generator object TPUEstimator.predict at 0x7fb3872898d0>
defaultdict(<function TpuExecutor.evaluate.<locals>.<lambda> at 0x7fb39c5e10e0>, {})
HERE
I0609 09:17:27.953360 140411182167872 tpu_system_metadata.py:91] Querying Tensorflow master (grpc://10.0.0.2:8470) for TPU system metadata.
2022-06-09 09:17:27.954526: W tensorflow/core/distributed_runtime/rpc/grpc_session.cc:373] GrpcSession::ListDevices will initialize the session with an empty graph and other defaults because the session has not yet been created.
I0609 09:17:27.959538 140411182167872 tpu_system_metadata.py:159] Found TPU system:
I0609 09:17:27.959874 140411182167872 tpu_system_metadata.py:160] *** Num TPU Cores: 8
I0609 09:17:27.960032 140411182167872 tpu_system_metadata.py:161] *** Num TPU Workers: 1
I0609 09:17:27.960154 140411182167872 tpu_system_metadata.py:163] *** Num TPU Cores Per Worker: 8
I0609 09:17:27.960274 140411182167872 tpu_system_metadata.py:165] *** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:CPU:0, CPU, -1, -2401410492520276784)
I0609 09:17:27.960741 140411182167872 tpu_system_metadata.py:165] *** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:0, TPU, 17179869184, 7251767768187345869)
I0609 09:17:27.960905 140411182167872 tpu_system_metadata.py:165] *** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:1, TPU, 17179869184, 2125513058134798884)
I0609 09:17:27.961036 140411182167872 tpu_system_metadata.py:165] *** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:2, TPU, 17179869184, 5864655708398411167)
I0609 09:17:27.961140 140411182167872 tpu_system_metadata.py:165] *** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:3, TPU, 17179869184, 67423300940110412)
I0609 09:17:27.961267 140411182167872 tpu_system_metadata.py:165] *** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:4, TPU, 17179869184, 7494320012297361397)
I0609 09:17:27.961366 140411182167872 tpu_system_metadata.py:165] *** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:5, TPU, 17179869184, 4209798017705109089)
I0609 09:17:27.961507 140411182167872 tpu_system_metadata.py:165] *** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:6, TPU, 17179869184, -399230976003128631)
I0609 09:17:27.961616 140411182167872 tpu_system_metadata.py:165] *** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU:7, TPU, 17179869184, 1899076944214499125)
I0609 09:17:27.961734 140411182167872 tpu_system_metadata.py:165] *** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:TPU_SYSTEM:0, TPU_SYSTEM, 17179869184, -7911280150411542086)
I0609 09:17:27.961843 140411182167872 tpu_system_metadata.py:165] *** Available Device: _DeviceAttributes(/job:worker/replica:0/task:0/device:XLA_CPU:0, XLA_CPU, 17179869184, -873687104241492985)
I0609 09:17:27.963420 140411182167872 estimator.py:1162] Calling model_fn.
W0609 09:17:27.982392 140411182167872 deprecation.py:339] From /kaggle/working/tpu/models/official/detection/dataloader/input_reader.py:53: parallel_interleave (from tensorflow.python.data.experimental.ops.interleave_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.data.Dataset.interleave(map_func, cycle_length, block_length, num_parallel_calls=tf.data.AUTOTUNE)` instead. If sloppy execution is desired, use `tf.data.Options.experimental_deterministic`.
2022-06-09 09:17:29.684331: I tensorflow/compiler/jit/xla_cpu_device.cc:41] Not creating XLA devices, tf_xla_enable_xla_devices not set
2022-06-09 09:17:29.684741: W tensorflow/stream_executor/platform/default/dso_loader.cc:60] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /opt/conda/lib
2022-06-09 09:17:29.684791: W tensorflow/stream_executor/cuda/cuda_driver.cc:326] failed call to cuInit: UNKNOWN ERROR (303)
2022-06-09 09:17:29.684836: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (11fda7e4f5d1): /proc/driver/nvidia/version does not exist
W0609 09:17:29.900516 140411182167872 deprecation.py:537] From /kaggle/working/tpu/models/official/detection/projects/vild/dataloader/tf_example_decoder.py:100: calling map_fn (from tensorflow.python.ops.map_fn) with dtype is deprecated and will be removed in a future version.
Instructions for updating:
Use fn_output_signature instead
I0609 09:17:33.046100 140411182167872 nn_blocks.py:135] -----> Building bottleneck block.
I0609 09:17:33.227478 140411182167872 nn_blocks.py:135] -----> Building bottleneck block.
I0609 09:17:33.378254 140411182167872 nn_blocks.py:135] -----> Building bottleneck block.
I0609 09:17:33.534032 140411182167872 nn_blocks.py:135] -----> Building bottleneck block.
I0609 09:17:33.757845 140411182167872 nn_blocks.py:135] -----> Building bottleneck block.
I0609 09:17:33.931611 140411182167872 nn_blocks.py:135] -----> Building bottleneck block.
I0609 09:17:34.118063 140411182167872 nn_blocks.py:135] -----> Building bottleneck block.
I0609 09:17:34.312999 140411182167872 nn_blocks.py:135] -----> Building bottleneck block.
I0609 09:17:34.594292 140411182167872 nn_blocks.py:135] -----> Building bottleneck block.
I0609 09:17:34.809886 140411182167872 nn_blocks.py:135] -----> Building bottleneck block.
I0609 09:17:35.036895 140411182167872 nn_blocks.py:135] -----> Building bottleneck block.
I0609 09:17:35.272871 140411182167872 nn_blocks.py:135] -----> Building bottleneck block.
I0609 09:17:35.518234 140411182167872 nn_blocks.py:135] -----> Building bottleneck block.
I0609 09:17:35.777961 140411182167872 nn_blocks.py:135] -----> Building bottleneck block.
I0609 09:17:36.131151 140411182167872 nn_blocks.py:135] -----> Building bottleneck block.
I0609 09:17:36.402851 140411182167872 nn_blocks.py:135] -----> Building bottleneck block.
W0609 09:17:40.345469 140411182167872 deprecation.py:339] From /kaggle/working/tpu/models/official/detection/ops/spatial_transform_ops.py:455: div (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Deprecated in favor of operator or tf.math.divide.
I0609 09:17:41.458560 140411182167872 vild_head.py:258] visual: Tensor("frcnn_layer_0/fast_rcnn_head/project-to-clip/BiasAdd:0", shape=(1, 1000, 512), dtype=float32)
I0609 09:17:41.461146 140411182167872 vild_head.py:261] visual_norm: Tensor("frcnn_layer_0/fast_rcnn_head/visual_norm/Sqrt:0", shape=(1, 1000, 1), dtype=float32)
I0609 09:17:41.477202 140411182167872 vild_head.py:308] loaded_numpy.shape: (512, 1203); clip dim: 512; num_classes: 1204
I0609 09:17:41.521802 140411182167872 vild_head.py:326] classifier_norm: Tensor("frcnn_layer_0/fast_rcnn_head/norm/Squeeze:0", shape=(1203,), dtype=float32)
I0609 09:17:41.567126 140411182167872 vild_head.py:341] bg_classifier: <tf.Variable 'frcnn_layer_0/fast_rcnn_head/background-class-predict/kernel:0' shape=(512, 1) dtype=float32>
I0609 09:17:41.570755 140411182167872 vild_head.py:343] bg_classifier_norm: Tensor("frcnn_layer_0/fast_rcnn_head/norm_1/Squeeze:0", shape=(1,), dtype=float32)
(1, 1000, 1203)
0
I0609 09:17:41.698615 140411182167872 error_handling.py:115] prediction_loop marked as finished
W0609 09:17:41.698956 140411182167872 error_handling.py:149] Reraising captured error
Traceback (most recent call last):
  File "main.py", line 235, in <module>
    tf.app.run(main)
  File "/opt/conda/lib/python3.7/site-packages/tensorflow/python/platform/app.py", line 40, in run
    _run(main=main, argv=argv, flags_parser=_parse_flags_tolerate_undef)
  File "/opt/conda/lib/python3.7/site-packages/absl/app.py", line 303, in run
    _run_main(main, args)
  File "/opt/conda/lib/python3.7/site-packages/absl/app.py", line 251, in _run_main
    sys.exit(main(argv))
  File "main.py", line 206, in main
    executor.evaluate(eval_input_fn, eval_times, ckpt)
  File "/kaggle/working/tpu/models/official/detection/executor/tpu_executor.py", line 222, in evaluate
    outputs = six.next(predictor)
  File "/opt/conda/lib/python3.7/site-packages/tensorflow_estimator/python/estimator/tpu/tpu_estimator.py", line 3173, in predict
    rendezvous.raise_errors()
  File "/opt/conda/lib/python3.7/site-packages/tensorflow_estimator/python/estimator/tpu/error_handling.py", line 150, in raise_errors
    six.reraise(typ, value, traceback)
  File "/opt/conda/lib/python3.7/site-packages/six.py", line 703, in reraise
    raise value
  File "/opt/conda/lib/python3.7/site-packages/tensorflow_estimator/python/estimator/tpu/tpu_estimator.py", line 3167, in predict
    yield_single_examples=yield_single_examples):
  File "/opt/conda/lib/python3.7/site-packages/tensorflow_estimator/python/estimator/estimator.py", line 613, in predict
    self.config)
  File "/opt/conda/lib/python3.7/site-packages/tensorflow_estimator/python/estimator/tpu/tpu_estimator.py", line 2962, in _call_model_fn
    config)
  File "/opt/conda/lib/python3.7/site-packages/tensorflow_estimator/python/estimator/estimator.py", line 1163, in _call_model_fn
    model_fn_results = self._model_fn(features=features, **kwargs)
  File "/opt/conda/lib/python3.7/site-packages/tensorflow_estimator/python/estimator/tpu/tpu_estimator.py", line 3494, in _model_fn
    dequeue_fn)
  File "/opt/conda/lib/python3.7/site-packages/tensorflow_estimator/python/estimator/tpu/tpu_estimator.py", line 3749, in _predict_on_tpu_system
    device_assignment=ctx.device_assignment)
  File "/opt/conda/lib/python3.7/site-packages/tensorflow/python/tpu/tpu.py", line 1749, in split_compile_and_shard
    xla_options=xla_options)
  File "/opt/conda/lib/python3.7/site-packages/tensorflow/python/tpu/tpu.py", line 1439, in split_compile_and_replicate
    outputs = computation(*computation_inputs)
  File "/opt/conda/lib/python3.7/site-packages/tensorflow_estimator/python/estimator/tpu/tpu_estimator.py", line 3733, in multi_tpu_predict_steps_on_single_shard
    cond, single_tpu_predict_step, inputs=inputs, name=b'loop')
  File "/opt/conda/lib/python3.7/site-packages/tensorflow/python/tpu/training_loop.py", line 178, in while_loop
    condition_wrapper, body_wrapper, inputs, name="", parallel_iterations=1)
  File "/opt/conda/lib/python3.7/site-packages/tensorflow/python/ops/control_flow_ops.py", line 2696, in while_loop
    back_prop=back_prop)
  File "/opt/conda/lib/python3.7/site-packages/tensorflow/python/ops/while_v2.py", line 200, in while_loop
    add_control_dependencies=add_control_dependencies)
  File "/opt/conda/lib/python3.7/site-packages/tensorflow/python/framework/func_graph.py", line 990, in func_graph_from_py_func
    func_outputs = python_func(*func_args, **func_kwargs)
  File "/opt/conda/lib/python3.7/site-packages/tensorflow/python/ops/while_v2.py", line 178, in wrapped_body
    outputs = body(*_pack_sequence_as(orig_loop_vars, args))
  File "/opt/conda/lib/python3.7/site-packages/tensorflow/python/tpu/training_loop.py", line 121, in body_wrapper
    outputs = body(*(inputs + dequeue_ops))
  File "/opt/conda/lib/python3.7/site-packages/tensorflow_estimator/python/estimator/tpu/tpu_estimator.py", line 1949, in predict_step
    features, labels, is_export_mode=False)
  File "/opt/conda/lib/python3.7/site-packages/tensorflow_estimator/python/estimator/tpu/tpu_estimator.py", line 2072, in _call_model_fn
    estimator_spec = self._model_fn(features=features, **kwargs)
  File "/kaggle/working/tpu/models/official/detection/modeling/model_builder.py", line 51, in __call__
    return self._model.predict(features)
  File "/kaggle/working/tpu/models/official/detection/modeling/base_model.py", line 389, in predict
    outputs = self.build_outputs(images, labels, mode=mode_keys.PREDICT)
  File "/kaggle/working/tpu/models/official/detection/modeling/base_model.py", line 206, in build_outputs
    outputs = self._build_outputs(images, labels, mode)
  File "/kaggle/working/tpu/models/official/detection/projects/vild/modeling/vild_model.py", line 215, in _build_outputs
    fpn_features, rpn_rois, output_size=14)
  File "/kaggle/working/tpu/models/official/detection/ops/spatial_transform_ops.py", line 448, in multilevel_crop_and_resize
    _, num_boxes, _ = boxes.get_shape().as_list()
ValueError: not enough values to unpack (expected 3, got 1)