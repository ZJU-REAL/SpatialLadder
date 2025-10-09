from data_utils.vsi_utils import *
from utils import *
from loguru import logger as eval_logger
import time
from accelerate import Accelerator
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
import numpy as np
from tqdm import tqdm
from PIL import Image
import pandas as pd
import os

def evaluate_spbench(rank, world_size, parquet_file, image_dir, model_name, output_dir, log_file, gpu_ids, num_frames=4, max_pixels=512*28*28, min_pixels=16*28*28, debug=False, batch_size=1, debug_size=12, params_dict=None, prompt_type="thinking"):    
    eval_task = os.path.basename(parquet_file).split('.')[0]
    
    logger = setup_logger(rank, log_file, params_dict)
    start_time_process = time.time()

    selected_gpu = allocate_gpu(rank, gpu_ids, world_size)
    logger.info(f"Rank {rank}/{world_size} Selected GPU: {selected_gpu}, Torch Device: {torch.cuda.current_device()}")

    accelerator = Accelerator()
    device = accelerator.device
    logger.info(f"Rank {rank} using device: {device}")

    df = pd.read_parquet(parquet_file)
    if debug:
        df = df.sample(n=debug_size)
        logger.info(f"Process {rank} Debug mode enabled, randomly processing {debug_size} samples.")

    if world_size > 1:
        df_shard = np.array_split(df, world_size)[rank]
    else:
        df_shard = df
    logger.info(f"Rank {rank} Shard size: {len(df_shard)}")
    
    processor = AutoProcessor.from_pretrained(model_name, use_fast=True, max_pixels=max_pixels, min_pixels=min_pixels)
    processor.tokenizer.padding_side = 'left'
        
    if world_size == 1 and len(gpu_ids.split(',')) > 1:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        model = accelerator.prepare(model)
        model.eval()
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16, 
            attn_implementation="flash_attention_2"
        ).eval().to(device)
        model = accelerator.prepare(model)
        
    results = []
    total_samples = len(df_shard)
    if total_samples == 0:
        logger.info(f"Rank {rank} has empty shard, skipping processing.")
        return os.path.join(output_dir, f"{eval_task}_results_rank_{rank}.jsonl"), 0

    prompt_template = PROMPT_TEMPLATES.get(prompt_type, PROMPT_TEMPLATES["default"])
    
    for start_index in tqdm(range(0, total_samples, batch_size), desc=f"Process {rank}", total=(total_samples + batch_size - 1) // batch_size):
        batch_df = df_shard.iloc[start_index:min(start_index + batch_size, total_samples)]
        batch_messages_list = []
        batch_row_infos = []
        prompt_list = []
        
        # Prepare all batch data first
        for _, row in batch_df.iterrows():
            if not os.path.exists(image_dir):
                print("Warning: image not found at: ", image_dir)
                continue
            image_paths = [os.path.join(image_dir, row['scene_name'], frame_id) for frame_id in row['images']]
                    
            question = row['question']
            options = row.get('options')
            if options is not None and len(options) > 0:
                options = options.tolist()                    
                question += "\nOptions:\n" + "\n".join(options)
                
            # Build prompt text
            prompt_text = prompt_template["pre_prompt"].format(question=question)
            if row['question_type'] in MCA_QUESTION_TYPES:
                prompt_text += "\n" + prompt_template["mca_post_prompt"]
            elif row['question_type'] in NA_QUESTION_TYPES:
                prompt_text += "\n" + prompt_template["na_post_prompt"]    
            prompt_list.append(prompt_text)
            
            images = []
            for image_path in image_paths:
                if os.path.exists(image_path):
                    img = Image.open(image_path).convert('RGB')
                    images.append(img)
                else:
                    logger.warning(f"Image not found: {image_path}")
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": image} for image in images],
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            
            batch_messages_list.append(messages)
            batch_row_infos.append(row)

        if not batch_row_infos:
            continue

        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in batch_messages_list
        ]
        image_inputs_batch, video_inputs_batch = process_vision_info(batch_messages_list)
        inputs_batch = processor(
            text=texts,
            images=image_inputs_batch,
            videos=video_inputs_batch,
            padding=True,
            return_tensors="pt",
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        ).to(device)
        
        max_new_token = 128 if prompt_type == "default" else 1024
        generated_ids_batch = model.generate(
            **inputs_batch, 
            use_cache=True, 
            max_new_tokens=max_new_token, 
            temperature=0.01
        )
        generated_ids_trimmed_batch = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs_batch.input_ids, generated_ids_batch)
        ]
        predicted_answers_batch = processor.batch_decode(
            generated_ids_trimmed_batch, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # Save results
        for i, predicted_answer in enumerate(predicted_answers_batch):
            row = batch_row_infos[i]
            ground_truth = row['ground_truth']
            question_type = row['question_type']
            prompt_text = prompt_list[i]
   
            results.append({
                'id': row['id'],
                'dataset': row['dataset'],
                'scene_name': row['scene_name'],
                'question': row['question'],
                'ground_truth': ground_truth,
                'predicted_answer': predicted_answer,
                'question_type': question_type,
                'prompt': prompt_text
            })

    # Write results to file
    process_output_file = os.path.join(output_dir, f"{eval_task}_results_rank_{rank}.jsonl")
    with open(process_output_file, 'w') as f:
        for result in results:
            json.dump(result, f, ensure_ascii=False)
            f.write("\n")

    end_time_process = time.time()
    elapsed_time_process = end_time_process - start_time_process

    elapsed_time_process_formatted = format_time(elapsed_time_process)
    logger.info(f"Rank {rank} results saved to: {process_output_file}, time usage: {elapsed_time_process_formatted}")
    return process_output_file, elapsed_time_process