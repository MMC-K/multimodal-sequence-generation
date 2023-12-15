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

