# multimodal sequence generation

## 데이터 생성 및 저장
```python
from datasets import load_dataset

raw_datasets = load_dataset(
                "pose_dataset_v2.py", 
                'body',
                cache_dir="huggingface_datasets",
                data_dir="data",
            )
raw_datasets.save_to_disk("preproc_data/proc")

```


## VQVAE 학습

```bash
PREPROC_DATA_PATH="preprocessed_data/proc"
PER_DEV_TRAIN_BATCH_SIZE=128
PER_DEV_EVAL_BATCH_SIZE=128
OUTPTU_DIR="output/vqvae/proc"
CHECKPOINTING_STEPS=1000
LOSS_BETA=1.0
LOADER_NUM_WORKERS=16

accelerate launch run_vqvae.py \
    --preproc_data_path ${PREPROC_DATA_PATH} \
    --per_device_train_batch_size ${PER_DEV_TRAIN_BATCH_SIZE} \
    --per_device_eval_batch_size ${PER_DEV_EVAL_BATCH_SIZE} \
    --output_dir ${OUTPTU_DIR} \
    --checkpointing_steps ${CHECKPOINTING_STEPS} \
    --beta ${LOSS_BETA} \
    --loader_num_workers ${LOADER_NUM_WORKERS}
```


## Convert sequence feature to latent index

```bash
PREPROC_DATA_PATH="preprocessed_data/proc"
VQVAE_MODEL_DIR="output/vqvae/proc"
VQVAE_MODEL_PATH=${VQVAE_MODEL_DIR}/pytorch_model.bin
OUTPTU_DIR="preprocessed_data/proc_idx"

python cvt_seq2latent.py \
--saved_dataset_path
    --vqvae_model_path ${VQVAE_MODEL_PATH} \
    --saved_dataset_path ${PREPROC_DATA_PATH} \
    --output_path ${OUTPTU_DIR}
```

## Sequene Generator 학습

```bash
PREPROCESSING_NUM_WORKERS=128
LR=0.001
SCHEDULER_TYPE="linear"
EPOCHS=100
DDP_TIMEOUT=36000

ENCODER_MODEL_NAME_OR_PATH="klue/bert-base"
DECODER_MODEL_NAME_OR_PATH="gpt2"
SAVED_DATASET_PATH="preprocessed_data/proc_idx"
OUTPUT_DIR="output/motion_gen"
PER_DEV_TRAIN_BATCH_SIZE=128
PER_DEV_EVAL_BATCH_SIZE=64
GRADIENT_ACCUMULATION_STEPS=1
EVAL_ACCUMULATION_STEPS=2


python -m torch.distributed.launch --nproc_per_node 8 test_training.py \
    --encoder_model_name_or_path ${ENCODER_MODEL_NAME_OR_PATH} \
    --decoder_model_name_or_path ${DECODER_MODEL_NAME_OR_PATH} \
    --saved_dataset_path ${SAVED_DATASET_PATH} \
    --preprocessing_num_workers ${PREPROCESSING_NUM_WORKERS} \
    --per_device_train_batch_size ${PER_DEV_TRAIN_BATCH_SIZE} \
    --per_device_eval_batch_size ${PER_DEV_EVAL_BATCH_SIZE} \
    --lr_scheduler_type ${SCHEDULER_TYPE} \
    --learning_rate ${LR} \
    --num_train_epochs ${EPOCHS} \
    --ddp_timeout ${DDP_TIMEOUT} \
    --save_strategy "epoch" \
    --evaluation_strategy "epoch" \
    --do_train --do_eval \
    --overwrite_output_dir \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --eval_accumulation_steps ${EVAL_ACCUMULATION_STEPS} \
    --log_level "error" \
    --output_dir ${OUTPUT_DIR}


```


