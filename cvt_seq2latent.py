from datasets import load_dataset, load_from_disk

from transformers import AutoTokenizer
import numpy as np
import torch
from tqdm import tqdm
from vqvae import VectorQuantizedVAE

raw_datasets = load_from_disk("preprocessed_data/proc")

model = VectorQuantizedVAE(153, 256, 256, 2048)
model.load_state_dict(torch.load("results/vqvae_1209_prec/step_28000/pytorch_model.bin"))
model.eval()
print("model loaded!!")

@torch.no_grad()
def preroc_fn(examples):
    new_pose_feature = [model.encode(torch.tensor(np.array(pf).reshape((1, -1, 153)), dtype=torch.float32)).tolist()[0] for pf in examples["pose_feature"]]
    
    return {
        "index": new_pose_feature,
        "text": examples["text"]
    }

train_columns = raw_datasets["train"].column_names
    
raw_datasets = raw_datasets.map(
    preroc_fn, 
    batched=True,
    batch_size=250,
    remove_columns=["pose_feature"], 
    num_proc=4, 
    desc="indexing..."
)

raw_datasets.save_to_disk("preprocessed_data/proc_idx_text")