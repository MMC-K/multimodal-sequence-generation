# Copyright 2023 san kim
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

import os
import csv
import json
import copy
import textwrap

from transformers import AutoTokenizer
import datasets

_VERSION = datasets.Version("2.0.0", "")

_URL = ""

_CITATION = """\
There is no citation information
"""

_DESCRIPTION = """\
"""


NUM_BODY_JOINTS = 21
NUM_HAND_JOINTS = 15

BODY_HANDS_KEYS = ["smplx_body_pose", "smplx_rhand_pose", "smplx_lhand_pose"]
CLIP_INFO_PATH = "youtube_clips.json"

def get_subseq_pose(pose_data, frame_idx):
    for pose_frame in pose_data[frame_idx:]:
        if len(pose_frame) > 0:
            return pose_frame[0]
    return None

def stuffing_pose_data(pose_data):
    pose_list = []
    prev_pose = None
    for frame_idx, pose_frame in enumerate(pose_data):
        if len(pose_frame) == 0:
            if prev_pose is not None:
                pose_list.append(copy.deepcopy(prev_pose))
            else:
                subseq_pose = get_subseq_pose(pose_data, frame_idx)
                if subseq_pose is not None:
                    pose_list.append(copy.deepcopy(subseq_pose))
                # If there is no prev_pose and subseq_pose, skip the frame
        else:
            prev_pose = copy.deepcopy(pose_frame[0])
            pose_list.append(prev_pose)
    return pose_list


def fmt_body_hands_seq(pose_data):
    pose_feature = []
    
    for pose_frame in pose_data:
        pose_feature.append(pose_frame["smplx_body_pose"]+pose_frame["smplx_lhand_pose"]+pose_frame["smplx_rhand_pose"])

    return {
        "pose_feature": pose_feature,
    }


def text_pose_generator_from_clip_info_list(clip_info_list, manual_dir, **kwargs):
    for clip_info in clip_info_list:
        pose_path = os.path.join(manual_dir, clip_info["pose_path"])
        
        try:
            pose_data = json.load(open(pose_path, "r"))

            person0_pose = stuffing_pose_data(pose_data)
            pose_dict = fmt_body_hands_seq(person0_pose)
            
            if len(pose_dict["pose_feature"]) == 0:
                continue
            
            pose_dict["clip_id"] = clip_info["clip_id"]
            pose_dict["text"] = clip_info["text"]
            pose_dict["url"] = clip_info["url"]
            
            yield pose_dict
        except Exception as e:
            print(f"Error: {e}")


def audio_pose_generator_from_clip_info_list(clip_info_list, manual_dir, **kwargs):
    for clip_info in clip_info_list:
        pose_path = os.path.join(manual_dir, clip_info["pose_path"])
        audio_path = os.path.join(manual_dir, clip_info["clip_audio_path"])
        
        try:
            pose_data = json.load(open(pose_path, "r"))

            person0_pose = stuffing_pose_data(pose_data)
            pose_dict = fmt_body_hands_seq(person0_pose)
            
            if len(pose_dict["pose_feature"]) == 0:
                continue
            
            pose_dict["clip_id"] = clip_info["clip_id"]
            pose_dict["text"] = clip_info["text"]
            pose_dict["url"] = clip_info["url"]
            pose_dict["audio"] = os.path.abspath(audio_path)
            
            yield pose_dict
        except Exception as e:
            print(f"Error: {e}")


TEXT_POSE_FEATURE = datasets.Features(
    {
        "pose_feature": datasets.Sequence(feature=datasets.Sequence(feature=datasets.Value("float32"))),
        "clip_id": datasets.Value("string"),
        "text": datasets.Value("string"),
        "url": datasets.Value("string"),
    }
)

AUDIO_POSE_FEATURE = datasets.Features(
    {
        "pose_feature": datasets.Sequence(feature=datasets.Sequence(feature=datasets.Value("float32"))),
        "clip_id": datasets.Value("string"),
        "text": datasets.Value("string"),
        "url": datasets.Value("string"),
        "audio": datasets.features.Audio(sampling_rate=48_000),
    }
)

class PoseDatasetConfig(datasets.BuilderConfig):
  def __init__( self,
                gen_type="text",
                features=TEXT_POSE_FEATURE,
                **kwargs):
    super(PoseDatasetConfig, self).__init__(
      **kwargs
    )
    self.gen_type = gen_type
    self.features = features


class PoseDataset(datasets.GeneratorBasedBuilder):
    """Pose Dataset"""

    BUILDER_CONFIGS = [
        PoseDatasetConfig(
            name="text",
            gen_type="text",
            features=TEXT_POSE_FEATURE,
            description="Pose Dataset. only body pose (body, left hand, right hand) with text",
        ),
        PoseDatasetConfig(
            name="audio",
            gen_type="audio",
            features=AUDIO_POSE_FEATURE,
            description="Pose Dataset. only body pose (body, left hand, right hand) with audio",
        ),
    ]
    
    BUILDER_CONFIG_CLASS = PoseDatasetConfig
    DEFAULT_CONFIG_NAME = "audio"

    def _info(self):
        return datasets.DatasetInfo(
            description=self.config.description,
            features=self.config.features,
            supervised_keys=None,  # Probably needs to be fixed.
            homepage=_URL,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        clip_info_list = json.load(open(os.path.join(dl_manager.manual_dir, CLIP_INFO_PATH), "r"))
        
        clip_info_list = [clip_info for clip_info in clip_info_list if clip_info["text"].strip() != ""]
        
        num_clip = len(clip_info_list)
        train_sp = int(num_clip*0.90)
        valid_sp = int(num_clip*0.92)

        path_kv = {
            datasets.Split.TRAIN:(
                clip_info_list[:train_sp], 
                dl_manager.manual_dir, 
                self.config.gen_type,
            ),
            datasets.Split.VALIDATION:(
                clip_info_list[train_sp:valid_sp], 
                dl_manager.manual_dir, 
                self.config.gen_type,
            ),
            datasets.Split.TEST:(
                clip_info_list[valid_sp:], 
                dl_manager.manual_dir, 
                self.config.gen_type,
            ),
        }

        return [
                datasets.SplitGenerator(name=k, gen_kwargs={
                    'clip_info_list': v[0],
                    'manual_dir': v[1],
                    'gen_type': v[2],
                }) for k, v in path_kv.items()
        ]

    def _generate_examples(self, clip_info_list, manual_dir, gen_type="text"):
        """Yields examples."""
        if gen_type == "text":
            generator = text_pose_generator_from_clip_info_list
        else:
            generator = audio_pose_generator_from_clip_info_list
            
        for idx, item in enumerate(generator(clip_info_list, manual_dir, gen_type=gen_type)):
            yield idx, item































