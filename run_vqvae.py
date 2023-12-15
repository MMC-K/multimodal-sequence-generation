
#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning a 🤗 Transformers model on summarization.
"""
# You can also adapt this script on your own summarization task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import os
import random
from dataclasses import dataclass
from typing import Union, Optional
from pathlib import Path

import datasets
import nltk
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from transformers import (
    PreTrainedTokenizerBase,
    SchedulerType,
    get_scheduler,
    AutoTokenizer,
)
from transformers.utils import PaddingStrategy
import torch.nn.functional as F
from smplx import SMPLX

from vqvae import VectorQuantizedVAE

logger = get_logger(__name__)


@dataclass
class DataCollatorForPoseSeq:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    pad_to_multiple_of: Optional[int] = 8
    return_tensors: str = "pt"
    
    def __call__(self, examples, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
            
        pose_features = [feature["pose_feature"] for feature in examples]
        text_feature = [feature["text"] for feature in examples]
        
        max_pose_feature_length = max(len(p) for p in pose_features)
        
        if self.pad_to_multiple_of is not None:
                max_pose_feature_length = (
                    (max_pose_feature_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )
        
        pose_feature_stack = []
        for idx, pose_feature in enumerate(pose_features):
            remainder = max_pose_feature_length - len(pose_feature)
            pose_array = np.array(pose_feature)
            pose_array = pose_array.reshape((-1, 153))
            pose_array = np.pad(pose_array, ((0, remainder), (0, 0)), mode="edge")
            pose_feature_stack.append(pose_array)

        pose_feature_stack = np.stack(pose_feature_stack, axis=0)

        # model_inputs = {
        #     "pose_feature": torch.tensor(pose_feature_stack, dtype=torch.float32)
        # }

        model_inputs = self.tokenizer(text_feature, padding=self.padding, return_tensors=return_tensors,)
        model_inputs["pose_feature"] = torch.tensor(pose_feature_stack, dtype=torch.float32)
        model_inputs["text"] = text_feature
        return model_inputs
    

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a VQVAE model on pose sequences")
    parser.add_argument(
        "--preproc_data_path",
        type=str,
        default="preprocessed_data/proc_shard50",
        help="The path of preprocessed data to use.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=100, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--loader_num_workers",
        type=int,
        default=0,
        help="The number of processes to use for the dataloader.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help=(
            "The weight for Commitment loss."
        ),
    )
    
    parser.add_argument(
        "--smplx_path",
        type=str,
        default="models/smplx",
        help="The path to smpl model",
    )
    parser.add_argument(
        "--model_gender",
        type=str,
        default="neutral",
        help="The gender of smpl model",
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
    else:
        datasets.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()


    raw_datasets = load_from_disk(args.preproc_data_path)
    
    model = VectorQuantizedVAE(153, 256, 256, 2048)

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names

    model_name = "KETI-AIR/long-ke-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    data_collator = DataCollatorForPoseSeq(
        tokenizer, pad_to_multiple_of=8
    )

    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["validation"]
    test_dataset = raw_datasets["test"]

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size,
        num_workers=args.loader_num_workers
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size,
                                 num_workers=args.loader_num_workers)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size,
                                 num_workers=args.loader_num_workers)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("summarization_no_trainer", experiment_config)

    # # load smplx model
    # layer_arg = {'create_global_orient': False, 'create_body_pose': False, 'create_left_hand_pose': False, 'create_right_hand_pose': False, 'create_jaw_pose': False, 'create_leye_pose': False, 'create_reye_pose': False, 'create_betas': False, 'create_expression': False, 'create_transl': False}
    # smplx_object = SMPLX(model_path=args.smplx_path, gender=args.model_gender, use_pca=False, use_face_contour=True, **layer_arg).cuda()


    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                x_tilde, z_e_x, z_q_x = model(batch["pose_feature"])

                pose_feature = batch["pose_feature"].cuda()
                x_tilde, z_e_x, z_q_x = model(pose_feature)

                # Reconstruction loss
                loss_recons = F.mse_loss(x_tilde, pose_feature)
                # Vector quantization objective
                loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
                # Commitment objective
                loss_commit = F.mse_loss(z_e_x, z_q_x.detach())

                loss = loss_recons + loss_vq + args.beta * loss_commit
                
                
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

        model.eval()

        for step, batch in enumerate(eval_dataloader):
            loss_recons_total = None
            loss_vq_total = None
            with torch.no_grad():
                pose_feature = batch["pose_feature"].cuda()
                x_tilde, z_e_x, z_q_x = model(pose_feature)
                loss_recons = F.mse_loss(x_tilde, pose_feature)
                loss_vq = F.mse_loss(z_q_x, z_e_x)
                
                loss_recons, loss_vq = accelerator.gather_for_metrics((loss_recons, loss_vq))
                loss_recons = loss_recons.cpu().numpy()
                loss_vq = loss_vq.cpu().numpy()
                
                if loss_recons_total is None:
                    loss_recons_total = loss_recons
                else:
                    loss_recons_total = np.concatenate((loss_recons_total, loss_recons))
                if loss_vq_total is None:
                    loss_vq_total = loss_vq
                else:
                    loss_vq_total = np.concatenate((loss_vq_total, loss_vq))

        result = {
            "loss_recons": float(np.mean(loss_recons_total)),
            "loss_vq": float(np.mean(loss_vq_total)),
        }

        logger.info(result)

        if args.with_tracking:
            result["train_loss"] = total_loss.item() / len(train_dataloader)
            result["epoch"] = epoch
            result["step"] = completed_steps
            accelerator.log(result, step=completed_steps)


        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(unwrapped_model.state_dict(), open(os.path.join(args.output_dir, "final.pt"), 'wb'))
        
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)

            all_results = {f"eval_{k}": v for k, v in result.items()}
            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump(all_results, f)

    model.eval()
    for step, batch in enumerate(test_dataloader):
        if step > 0:
            break
        with torch.no_grad():
            # batch["pose"] = batch.pop("pose_feature")
            pose_feature = batch["pose_feature"].cuda()
            x_tilde, z_e_x, z_q_x = model(pose_feature)  # [bsz, seq, 153]
            os.makedirs(os.path.join(args.output_dir, 'json', f'step_{starting_epoch}'), exist_ok=True)
            for i in range(batch['pose_feature'].size(0)):
                json_dump = {}
                json_dump['pose'] = batch['pose_feature'][i].tolist()
                json_dump['predicted_pose'] = x_tilde[i].squeeze().tolist(),
                with open(os.path.join(os.path.dirname(args.resume_from_checkpoint) if args.resume_from_checkpoint is not None else args.output_dir, 'json', f'step_{starting_epoch}', f"pose_results_{step}_{i}.json"), "w") as f:
                    json.dump(json_dump, f, indent=4)

if __name__ == "__main__":
    main()










