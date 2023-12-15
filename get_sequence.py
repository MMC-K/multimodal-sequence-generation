
# from transformers import (
#     CONFIG_MAPPING,
#     MODEL_FOR_TEXT_ENCODING_MAPPING,
#     AutoTokenizer,
#     EncoderDecoderModel,
#     EncoderDecoderConfig,
#     AutoModelForTextEncoding,
#     AutoModelForCausalLM,
#     AutoConfig
# )

# MODEL_CONFIG_CLASSES = list(MODEL_FOR_TEXT_ENCODING_MAPPING.keys())
# MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


# out_dir = "test_model1"
# text_pretrained_model_path = "KETI-AIR/ke-t5-base"

# # text_encoder_config = CONFIG_MAPPING["t5"].from_pretrained(text_pretrained_model_path)

# text_encoder_config = AutoConfig.from_pretrained(text_pretrained_model_path)
# text_encoder_model = AutoModelForTextEncoding.from_pretrained(text_pretrained_model_path)
# text_tokenizer = AutoTokenizer.from_pretrained(text_pretrained_model_path)

# motion_decoder_config = AutoConfig.from_pretrained(
#     "gpt2", 
#     vocab_size=2051,
#     bos_token_id=2050,
#     eos_token_id=2048,
#     pad_token_id=2049,
#     hidden_size=128,
#     n_head=8,
#     add_cross_attention=True
# )
# motion_decoder_model = AutoModelForCausalLM.from_config(motion_decoder_config)


# encdec_config = EncoderDecoderConfig.from_encoder_decoder_configs(text_encoder_config, motion_decoder_config)
# model = EncoderDecoderModel(encdec_config, text_encoder_model, motion_decoder_model)

# model.save_pretrained(out_dir)

# model = EncoderDecoderModel.from_pretrained(out_dir)

# test_text = "Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch."

# input_ids = text_tokenizer(test_text, return_tensors="pt").input_ids




# outputs = model.generate(
#     input_ids, 
#     num_beams=4,
#     max_length=1024,
# )
# print(outputs)





# from datasets import load_dataset, load_from_disk, DatasetDict, Split

# raw_datasets = load_from_disk("preprocessed_data/proc_filter_token")
# # raw_datasets = load_from_disk("preprocessed_data/proc_shard50")
# print(raw_datasets)
# test_size=0.08
# validation_size=0.01

# raw_datasets = raw_datasets.train_test_split(test_size=test_size, shuffle=True)
# raw_datasets_tmp = raw_datasets["train"].train_test_split(test_size=validation_size, shuffle=True)
# raw_datasets["train"] = raw_datasets_tmp["train"]
# raw_datasets["validation"] = raw_datasets_tmp["test"]
# print(raw_datasets)
# raw_datasets.save_to_disk("preprocessed_data/proc_filter_token_dict")


# # print(new_raw_datasets)

# # for idx, item in zip(range(4), raw_datasets["train"]):
# #     print(item)
















import json

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

model_name_or_path = "test_output_motion_shards50"
saved_dataset_path = "preprocessed_data/proc_idx_text_shards50"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = EncoderDecoderModel.from_pretrained(model_name_or_path)
model.config.decoder_start_token_id = model.decoder.config.bos_token_id

raw_datasets = load_from_disk(saved_dataset_path)


vqmodel = VectorQuantizedVAE(153, 256, 256, 2048)
vqmodel.load_state_dict(torch.load("results/vqvae_1209_prec/step_28000/pytorch_model.bin"))
vqmodel.eval()

out_json = []
out_json_file_path = "train.json"
num_sample = 5
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
        
        print(item["text"])
        print(origin)
        print(outputs[:,1:-1])
        
        origin_frame = vqmodel.decode(origin)[0]
        pred_frame = vqmodel.decode(outputs[:,1:-1])[0]
        print(origin_frame.size())
        print(pred_frame.size())
        
        print(len(origin_frame.tolist()))
        print(len(pred_frame.tolist()))
        
        out_json.append({
            "text": item["text"],
            "origin": origin_frame.tolist(),
            "predicted": pred_frame.tolist(),
        })
        
        if sample_cnt >= num_sample:
            break
        
        
json.dump(out_json, open(out_json_file_path, "w"), indent=4)

