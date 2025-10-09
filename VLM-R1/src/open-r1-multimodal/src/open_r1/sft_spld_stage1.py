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
    
    PROMPT_TEMPLATE = (
        "Question: {question}\n"
        "Please carefully observe the image first to identify the object(s) referred to in the question. "
        "Note that each object type appears only once in the image. "
        "Please provide the 2D bounding box coordinates and labels of the related objects in JSON format. "
    )
    
    question = example["question"]    
    prompt = PROMPT_TEMPLATE.format(question=question)
    answer = "```json\n" + json.dumps(example["answer"], indent=4) + "\n```"
    
    content = {
        "type": "image",
        "image": os.path.join(image_folders, example["image"][0])
    }

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}]
        },
        {
            "role": "user",
            "content": [
                content,
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

def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 16 * 28 * 28, max_pixels: int = 128 * 28 * 28
):
    """Rescales the image so that the following conditions are met:
    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar

def convert_to_qwen25vl_format(bbox, orig_height, orig_width, factor, min_pixels, max_pixels):
    new_height, new_width = smart_resize(orig_height, orig_width, factor, min_pixels, max_pixels)
    scale_w = new_width / orig_width
    scale_h = new_height / orig_height
    
    x1, y1, x2, y2 = bbox
    x1_new = round(x1 * scale_w)
    y1_new = round(y1 * scale_h)
    x2_new = round(x2 * scale_w)
    y2_new = round(y2 * scale_h)
    
    x1_new = max(0, min(x1_new, new_width - 1))
    y1_new = max(0, min(y1_new, new_height - 1))
    x2_new = max(0, min(x2_new, new_width - 1))
    y2_new = max(0, min(y2_new, new_height - 1))
    
    return [x1_new, y1_new, x2_new, y2_new]

if __name__ == "__main__":    
    # Parse arguments
    parser = TrlParser((CustomScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()
    
    max_pixels = script_args.max_pixels
    min_pixels = script_args.min_pixels
    height = 968
    width = 1296
    
    # Configure training args
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    # Load dataset
    if script_args.dataset_name.endswith('.json') or script_args.dataset_name.endswith('.jsonl'):
        import json
        data = []
        with open(script_args.dataset_name, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                item = json.loads(line.strip())
                if 'answer' in item:
                    if isinstance(item['answer'], list):
                        sol_bboxes = item['answer']
                        for sol_bbox in sol_bboxes:
                            sol_bbox["bbox_2d"] = convert_to_qwen25vl_format(
                                sol_bbox["bbox_2d"], height, width, 28, min_pixels, max_pixels
                            )
                        item['answer'] = json.dumps(sol_bboxes, indent=4)
                    elif not isinstance(item['answer'], str):
                        item['answer'] = str(item['answer'])
                data.append(item)
        dataset =  DatasetDict({"train": Dataset.from_list(data)})
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
        max_pixels=max_pixels,
        min_pixels=min_pixels,
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
    trainer.train(resume_from_checkpoint=True if training_args.resume_from_checkpoint else None)

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