# Copyright 2025 The HuggingFace Team. All rights reserved.
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
import re
import pathlib
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from babel.numbers import parse_decimal
from utils.math import compute_score
from datasets import Dataset, Features, Value, Sequence, load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from trainer import VLMGRPOTrainer, GRPOConfig
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
import PIL
from Levenshtein import ratio
from open_r1.utils.pycocotools.coco import COCO
from open_r1.utils.pycocotools.cocoeval import COCOeval
import json
import math
from json_repair import repair_json

from open_r1.vlm_modules import *

from typing import Tuple
from transformers.utils import logging
from transformers import AutoProcessor, AutoTokenizer

from openai import OpenAI

import numpy as np
from collections import defaultdict
import cv2

logger = logging.get_logger(__name__)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "sk-proj-1234567890"),
    base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
)

from open_r1.qwen2_5vl_monkey_patch import monkey_patch_qwen2_5vl_flash_attn, monkey_patch_qwen2_5vl_forward, monkey_patch_torch_load
monkey_patch_qwen2_5vl_flash_attn()    
monkey_patch_torch_load()

tokenizer = None

def initialize_tokenizer(model_path):
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.
    """
    data_file_paths: str = field(
        default=None,
        metadata={"help": "Paths to data files, separated by ':'"},
    )
    image_folders: str = field(
        default=None,
        metadata={"help": "Paths to image folders, separated by ':'"},
    )
    arrow_cache_dir: str = field(
        default=None,
        metadata={"help": "Path to arrow cache directory"},
    )
    val_split_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio of validation split, default 0.0"},
    )
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image (for QwenVL)"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image (for QwenVL)"},
    )
    max_anyres_num: Optional[int] = field(
        default=12,
        metadata={"help": "Maximum number of anyres blocks for the image (for InternVL)"},
    )
    reward_method: Optional[str] = field(
        default=None,
        metadata={
            "help": "Choose reward method: 'default', 'mcp', ..."
        },
    )
    is_reward_customized_from_vlm_module: bool = field(
        default=False,
        metadata={"help": "Whether to use a customized reward from vlm module"},
    )
    accuracy_reward_weight: float = field(
        default=1,
        metadata={"help": "Weight for accuracy reward"}
    )
    format_reward_weight: float = field(
        default=1,
        metadata={"help": "Weight for format reward"}
    )

def clean_text(text, exclude_chars=['\n', '\r']):
    # Extract content between <answer> and </answer> if present
    answer_matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_matches:
        # Use the last match
        text = answer_matches[-1]

    for char in exclude_chars:
        if char in ['\n', '\r']:
            # If there is a space before the newline, remove the newline
            text = re.sub(r'(?<=\s)' + re.escape(char), '', text)
            # If there is no space before the newline, replace it with a space
            text = re.sub(r'(?<!\s)' + re.escape(char), ' ', text)
        else:
            text = text.replace(char, ' ')
    
    # Remove leading and trailing spaces and convert to lowercase
    return text.strip().rstrip('.').lower()

def accuracy_reward(completions, **kwargs):
    def to_float(pred):
        try:
            pred = float(pred)
        except BaseException as e:
            pred = None
        return pred

    def abs_dist_norm(pred, target):
        return abs(pred - target) / target

    def mean_relative_accuracy(pred, target, start=.5, end=.95, interval=.05):
        if pred is None or target is None:
            return 0.0
        
        num_pts = (end - start) / interval + 2
        conf_intervs = np.linspace(start, end, int(num_pts))
        accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
        return accuracy.mean()

    def mca_reward(content, answer):
            content = clean_text(content)
            answer = clean_text(answer)
            return 1.0 if content == answer else 0.0
        
    def na_reward(content, answer):
        content = clean_text(content)
        answer = clean_text(answer)
        return mean_relative_accuracy(to_float(content), to_float(answer))
    
    """Reward function that checks if the completion is correct using symbolic verification, exact string matching, or fuzzy matching."""
    answers = kwargs.get("answer")
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, answer, accu_reward_method in zip(contents, answers, kwargs.get("accu_reward_method")):
        # if accu_reward_method is defined, use the corresponding reward function, otherwise use the default reward function
        if accu_reward_method == "mca":
            reward = mca_reward(content, answer)
        elif accu_reward_method == "na":
            reward = na_reward(content, answer)
        else:
            reward = 0.0
        rewards.append(reward * script_args.accuracy_reward_weight)
        
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            with open(log_path.replace(".txt", "_accuracy.txt"), "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"accu_reward_method: {accu_reward_method}\n")
                f.write(f"Content: {content}\n")
                f.write(f"Answer: {answer}\n")     
        
    return rewards

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format.
    
    Requirements:
    1. Must have exactly one <think>...</think> block followed by one <answer>...</answer> block
    2. No nested <think> tags inside the <think> block
    3. No extra <think> or <answer> tags outside the main structure
    """
    
    # Basic format check: one think block + one answer block
    basic_pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    
    rewards = []
    
    for content in completion_contents:
        # Strip leading and trailing whitespace
        content = content.strip()
        
        # Check basic format
        basic_match = re.fullmatch(basic_pattern, content, re.DOTALL)
        
        if not basic_match:
            # Basic format doesn't match
            rewards.append(0.0)
            continue
        
        # Extract think content to check for nested think tags
        think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        if think_match:
            think_content = think_match.group(1)
            
            # Check if think content contains nested <think> tags
            if "<think>" in think_content:
                # Found nested think tags, no reward
                rewards.append(0.0)
                continue
        
        # Check for multiple think or answer blocks
        think_count = len(re.findall(r"<think>", content))
        answer_count = len(re.findall(r"<answer>", content))
        
        if think_count != 1 or answer_count != 1:
            # Multiple think or answer blocks, no reward
            rewards.append(0.0)
            continue
        
        # All checks passed, give reward
        rewards.append(float(script_args.format_reward_weight))
    
    # Debug logging (keep original format)
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH")
        with open(log_path.replace(".txt", "_format.txt"), "a", encoding='utf-8') as f:
            for content, reward in zip(completion_contents, rewards):
                f.write(f"------------- {current_time} Format reward -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Has format: {bool(reward > 0)}\n")

    return rewards

reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False

def get_vlm_module(model_name_or_path):
    if "qwen" in model_name_or_path.lower():
        return Qwen2VLModule
    elif "internvl" in model_name_or_path.lower():
        return InvernVLModule
    else:
        # raise ValueError(f"Unsupported model: {model_name_or_path}")
        return Qwen2VLModule

MCA_QUESTION_TYPES = [
    "relative distance",
    "relative direction",
    "appearance order",
]

NA_QUESTION_TYPES = [
    "object size",
    "room size",
    "object count",
    "absolute distance",
]

def main(script_args, training_args, model_args):
    # Load the VLM module
    vlm_module_cls = get_vlm_module(model_args.model_name_or_path)
    print("using vlm module:", vlm_module_cls.__name__)

    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)

    # Load the JSONL datasets
    import json
    from datasets import Dataset
    
    data_files = script_args.data_file_paths.split(":")
    image_folders = script_args.image_folders.split(":")
    
    if len(data_files) != len(image_folders):
        raise ValueError("Number of data files must match number of image folders")
    
    if len(data_files) != len(image_folders):
        raise ValueError("Number of data files must match number of image folders")
    
    all_data = []
    for data_file, image_folder in zip(data_files, image_folders):
        with open(data_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                if 'image' in item:
                    if isinstance(item['image'], str):
                        # Store image path instead of loading the image
                        item['image_path'] = [os.path.join(image_folder, item['image'])]
                        del item['image'] # remove the image column so that it can be loaded later
                    elif isinstance(item['image'], list):
                        # if the image is a list, then it is a list of images (for multi-image input)
                        item['image_path'] = [os.path.join(image_folder, image) for image in item['image']]
                        del item['image'] # remove the image column so that it can be loaded later
                    else:
                        raise ValueError(f"Unsupported image type: {type(item['image'])}")
                    
                # Remove immediate image loading
                item["question"] = item["question"].replace("<image>", '')
                if item["options"] is not None and len(item["options"]) > 0:
                    item["question"] += '\n' + "Options:\n" + '\n'.join(item["options"])
                
                if item["question_type"] in MCA_QUESTION_TYPES:
                    item["accu_reward_method"] = "mca"
                elif item["question_type"] in NA_QUESTION_TYPES:
                    item["accu_reward_method"] = "na"
                else:
                    raise ValueError(f"Unsupported question sub type: {item['question_type']}")
                
                item["options"] = [] if item["options"] is None else item["options"]
                item["answer"] = str(item["answer"])
                item["data_type"] = item.get("data_type", "single_image")

                all_data.append(item)
                
    features = Features({
        'question_id': Value('int64'),
        'question': Value('string'),
        'options': Sequence(Value('string')),
        'question_type': Value('string'),
        'image_path': Sequence(Value('string')),
        'answer': Value('string'),
        'accu_reward_method': Value('string'),
        'data_type': Value('string')
    })

    dataset = Dataset.from_list(all_data, features=features)
    # dataset = Dataset.from_list(all_data)
    
    def get_question_prompt(question_type, question):
        pre_prompt = (
            f"Question: {question} \n"
            "Please Think about this question as if you were a human pondering deeply. "
            "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
            "It's encouraged to include self-reflection or verification in the reasoning process. \n"
        )
        
        if question_type in MCA_QUESTION_TYPES:
            post_prompt = (
                "Please provide your detailed reasoning between the <think> </think> tags, "
                "and then answer the question with the option's letter from the given choices (e.g., A, B, etc.) within the <answer> </answer> tags."
            )
        elif question_type in NA_QUESTION_TYPES:
            post_prompt = (
                "Please provide your detailed reasoning between the <think> </think> tags, "
                "and then answer the question with a numerical value (e.g., 42 or 3.1) within the <answer> </answer> tags."
            )
        else:
            raise ValueError(f"Unsupported question sub type: {question_type}")
        
        prompt = pre_prompt + post_prompt
        
        return prompt
            
    def make_conversation_from_jsonl(example):
        question_prompt = get_question_prompt(example["question_type"], example["question"])

        if 'image_path' in example and example['image_path'] is not None:
            assert all(os.path.exists(p) for p in example['image_path']), f"Image paths do not exist: {example['image_path']}"
            # Don't load image here, just store the path
            return {
                'image_path': [p for p in example['image_path']],  # Store path instead of loaded image
                'question': example['question'],
                'answer': example['answer'],
                'accu_reward_method': example['accu_reward_method'],
                'data_type': example['data_type'],
                'prompt': [{
                    'role': 'user',
                    'content': [
                        *({'type': 'image', 'text': None, 'image': image_path} for image_path in example['image_path']),
                        {'type': 'text', 'text': question_prompt, 'image': None}
                    ]
                }]
            }
        else:
            return {
                'question': example['question'],
                'answer': example['answer'],
                'accu_reward_method': example['accu_reward_method'],
                'data_type': example['data_type'],
                'prompt': [{
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': question_prompt}
                    ]
                }]
            }

    # Map the conversations
    dataset = dataset.map(make_conversation_from_jsonl, num_proc=8)

    # Split dataset for validation if requested
    splits = {'train': dataset}
    if script_args.val_split_ratio > 0:
        train_val_split = dataset.train_test_split(
            test_size=script_args.val_split_ratio
        )
        splits['train'] = train_val_split['train']
        splits['validation'] = train_val_split['test']

    # Select trainer class based on vlm_trainer argument
    trainer_cls = VLMGRPOTrainer
    print("using trainer:", trainer_cls.__name__)
    initialize_tokenizer(model_args.model_name_or_path)
    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        vlm_module=vlm_module_cls(),
        train_dataset=splits['train'],
        eval_dataset=splits.get('validation') if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        max_anyres_num=script_args.max_anyres_num,
    )

    # Train and push the model to the Hub
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    if training_args.deepspeed and "zero3" in training_args.deepspeed:
        print("zero3 is used, qwen2_5vl forward monkey patch is applied")
        monkey_patch_qwen2_5vl_forward()
    main(script_args, training_args, model_args)
