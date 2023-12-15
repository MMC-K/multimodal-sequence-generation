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
PREPROC_DATA_PATH="preprocessed_data/proc_shard10"
PER_DEV_TRAIN_BATCH_SIZE=128
PER_DEV_EVAL_BATCH_SIZE=128
OUTPTU_DIR="test_output2/proc_shard10"
CHECKPOINTING_STEPS=1000
LOSS_BETA=1.0
LOADER_NUM_WORKERS=16

accelerate launch test_model.py \
    --preproc_data_path ${PREPROC_DATA_PATH} \
    --per_device_train_batch_size ${PER_DEV_TRAIN_BATCH_SIZE} \
    --per_device_eval_batch_size ${PER_DEV_EVAL_BATCH_SIZE} \
    --output_dir ${OUTPTU_DIR} \
    --checkpointing_steps ${CHECKPOINTING_STEPS} \
    --beta ${LOSS_BETA} \
    --loader_num_workers ${LOADER_NUM_WORKERS}
```





