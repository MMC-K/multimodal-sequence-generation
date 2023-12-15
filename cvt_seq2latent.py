import argparse

from datasets import load_dataset, load_from_disk

from transformers import AutoTokenizer
import numpy as np
import torch
from tqdm import tqdm
from vqvae import VectorQuantizedVAE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_dataset_path", type=str, default="preproc_data/proc", help="The path to saved data")
    parser.add_argument("--vqvae_model_path", type=str, default="test_output2/proc_shard10/pytorch_model.bin", help="The path to vqvae data")
    parser.add_argument("--output_path", type=str, default="preproc_data/proc_idx", help="The path to output directory")
    
    parser.add_argument("--vq_codebook_size", type=int, default=2048, help="vq_codebook_size")
    parser.add_argument("--vq_input_size", type=int, default=153, help="vq_input_size")
    parser.add_argument("--vq_hidden_size", type=int, default=256, help="vq_hidden_size")
    parser.add_argument("--vq_output_size", type=int, default=256, help="vq_output_size")
    return parser.parse_args()


def main():
    args = parse_args()

    raw_datasets = load_from_disk(args.saved_dataset_path)

    model = VectorQuantizedVAE(args.vq_input_size, args.vq_hidden_size, args.vq_output_size, args.vq_codebook_size)
    model.load_state_dict(torch.load(args.vqvae_model_path))
    model.eval()

    @torch.no_grad()
    def preroc_fn(examples):
        new_pose_feature = [model.encode(torch.tensor(np.array(pf).reshape((1, -1, 153)), dtype=torch.float32)).tolist()[0] for pf in examples["pose_feature"]]
        
        return {
            "index": new_pose_feature,
            "text": examples["text"]
        }
        
    raw_datasets = raw_datasets.map(
        preroc_fn, 
        batched=True,
        batch_size=250,
        remove_columns=["pose_feature"], 
        num_proc=4, 
        desc="indexing..."
    )

    raw_datasets.save_to_disk(args.output_path)

if __name__=="__main__":
    main()
    
    
