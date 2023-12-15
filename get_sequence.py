import argparse

import json
from tqdm import tqdm

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    EncoderDecoderModel,
    EncoderDecoderConfig,
    AutoModelForTextEncoding,
    AutoModelForCausalLM,
)
from datasets import load_from_disk

from vqvae import VectorQuantizedVAE

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="test_output_motion_shards50", help="The path to model")
    parser.add_argument("--saved_dataset_path", type=str, default="preprocessed_data/proc_idx_text_shards50", help="The path to saved data")
    parser.add_argument("--vqvae_model_path", type=str, default="results/vqvae_1209_prec/step_28000/pytorch_model.bin", help="The path to vqvae data")
    parser.add_argument("--output_path", type=str, default="train.json", help="The path to output json")
    
    parser.add_argument("--vq_codebook_size", type=int, default=2048, help="vq_codebook_size")
    parser.add_argument("--vq_input_size", type=int, default=153, help="vq_input_size")
    parser.add_argument("--vq_hidden_size", type=int, default=256, help="vq_hidden_size")
    parser.add_argument("--vq_output_size", type=int, default=256, help="vq_output_size")
    
    parser.add_argument("--num_sample", type=int, default=10, help="num_sample")
    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = EncoderDecoderModel.from_pretrained(args.model_name_or_path)
    model.config.decoder_start_token_id = model.decoder.config.bos_token_id

    raw_datasets = load_from_disk(args.saved_dataset_path)


    vqmodel = VectorQuantizedVAE(args.vq_input_size, args.vq_hidden_size, args.vq_output_size, args.vq_codebook_size)
    vqmodel.load_state_dict(torch.load(args.vqvae_model_path))
    vqmodel.eval()

    out_json = []
    sample_cnt = 0

    with torch.no_grad():
        for item in raw_datasets["train"]:
            
            if item["text"] == "":
                continue
            else:
                sample_cnt+=1
            
            origin = torch.tensor(item["index"]).unsqueeze(0)
            input_ids = tokenizer(item["text"], return_tensors="pt").input_ids
            outputs = model.generate(
                input_ids,
                max_length=1024
            )
            
            origin_frame = vqmodel.decode(origin)[0]
            pred_frame = vqmodel.decode(outputs[:,1:-1])[0]
            
            out_json.append({
                "text": item["text"],
                "origin": origin_frame.tolist(),
                "predicted": pred_frame.tolist(),
            })
            
            if sample_cnt >= args.num_sample:
                break
            
    json.dump(out_json, open(args.output_path, "w"), indent=4)

if __name__=="__main__":
    main()