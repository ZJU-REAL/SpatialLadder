import os
import json
import random
import requests
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import math

from datasets import load_dataset
from transformers import (
    AutoProcessor,
    Qwen2VLProcessor,
    Qwen2_5_VLForConditionalGeneration
)
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
)
from accelerate import Accelerator
from qwen_vl_utils import process_vision_info

from datasets import Dataset, DatasetDict
import wandb

@dataclass
class CustomScriptArguments(ScriptArguments):
    """Extended script arguments with additional parameters."""
    
    image_folders: Optional[str] = field(
        default="./images",
        metadata={"help": "Directory containing image/video files"}
    )
    
    max_pixels: int = field(
        default=128 * 28 * 28,
        metadata={"help": "Maximum number of pixels for image/video processing"}
    )
    
    min_pixels: int = field(
        default=16 * 28 * 28,
        metadata={"help": "Minimum number of pixels for image/video processing"}
    )

def get_current_device():
    """Get the current device. For GPU we return the local process index to enable multiple GPU training."""
    return Accelerator().local_process_index if torch.cuda.is_available() else "cpu"

def prepare_dataset(example: Dict[str, Any], image_folders: str) -> Dict[str, List[Dict[str, Any]]]:
    """Prepare dataset example for training."""

    system_message = "You are a helpful assistant"
    
    PROMPT_TEMPLATE = {
        "pre_prompt": "Question: {question}\n",
        "mca_post_prompt": "Please answer with the option's letter from the given choices directly.",
        "na_post_prompt": "Please answer the question using a numerical value (e.g., 42 or 3.1) directly.",
    }
    
    question = example["question"]    
    if example["options"] is not None:
        question += 'Options:\n' + "\n".join(example["options"])
        prompt = PROMPT_TEMPLATE["pre_prompt"].format(question=question) + "\n" + PROMPT_TEMPLATE["mca_post_prompt"]
    else:
        prompt = PROMPT_TEMPLATE["pre_prompt"].format(question=question) + "\n" + PROMPT_TEMPLATE["na_post_prompt"]
    answer = example["answer"]
    
    if example["data_type"] == "video":
        content = [{
            "type": "video",
            "video": [os.path.join(image_folders, img) for img in example["image"]]
        }]
    else:
        content = [{
            "type": "image",
            "image": os.path.join(image_folders, img)
        } for img in example["image"]]

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}]
        },
        {
            "role": "user",
            "content": [
                *content,
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": answer}]
        }
    ]

    return {"messages": messages}

def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate batch of examples for training."""
    # For mixed image/video data, process one example at a time to avoid indexing issues
    if len(examples) == 1:
        # Single example - process normally
        example = examples[0]
        text = processor.apply_chat_template(example["messages"], tokenize=False)
        image_inputs, video_inputs, video_kwargs = process_vision_info(example["messages"], return_video_kwargs=True)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True
        )
    else:
        # Multiple examples - use the safer approach
        texts = []
        all_image_inputs = []
        all_video_inputs = []

        for i, example in enumerate(examples):
            try:
                text = processor.apply_chat_template(example["messages"], tokenize=False)
                texts.append(text)
                
                image_inputs, video_inputs, video_kwargs = process_vision_info(example["messages"], return_video_kwargs=True)
                
                if image_inputs:
                    all_image_inputs.extend(image_inputs)
                if video_inputs:
                    all_video_inputs.extend(video_inputs)
                    
            except Exception as e:
                raise ValueError(f"Failed to process example {i}: {e}")

        # Only use one type to avoid mixing issues
        images_to_use = all_image_inputs if all_image_inputs and not all_video_inputs else None
        videos_to_use = all_video_inputs if all_video_inputs and not all_image_inputs else None
        
        inputs = processor(
            text=texts,
            images=images_to_use,
            videos=videos_to_use,
            return_tensors="pt",
            padding=True
        )

    labels = inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Handle visual tokens
    visual_tokens = [151652, 151653, 151656] if isinstance(processor, Qwen2VLProcessor) else [
        processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    ]

    for visual_token_id in visual_tokens:
        labels[labels == visual_token_id] = -100

    inputs["labels"] = labels
    return inputs

if __name__ == "__main__":
    # Parse arguments
    parser = TrlParser((CustomScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()
    
    # Configure training args
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    # Load dataset
    if script_args.dataset_name.endswith('.json') or script_args.dataset_name.endswith('.jsonl'):
        dataset =  DatasetDict({"train": Dataset.from_json(script_args.dataset_name)})
    else:
        # Load the dataset
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # Setup model
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )

    # # Quantization configuration for 4-bit training
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )

    # Model initialization
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch_dtype,
        # device_map=get_kbit_device_map(),
        # quantization_config=bnb_config,
    )
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)


    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        use_fast=True,
        max_pixels=128*28*28
    )

    # Prepare dataset
    prepared_dataset = [prepare_dataset(example, script_args.image_folders) for example in dataset['train']]

    # Initialize wandb if specified
    if training_args.report_to == "wandb":
        wandb.init(project="video-llm-training")

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=prepared_dataset,
        data_collator=collate_fn,
        peft_config=get_peft_config(model_config),
        # tokenizer=processor.tokenizer
    )

    # Train model
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint if training_args.resume_from_checkpoint != True else None)

    # Save final model

    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)

    if trainer.accelerator.is_main_process:
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    # Cleanup
    del model
    del trainer
    torch.cuda.empty_cache()
    wandb.finish()