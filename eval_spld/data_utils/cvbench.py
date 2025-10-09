from data_utils.vsi_utils import *
from utils import *
from loguru import logger as eval_logger
import time
from accelerate import Accelerator
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration,Qwen2VLForConditionalGeneration, AutoProcessor
import numpy as np
from tqdm import tqdm
from PIL import Image
import pandas as pd
import os
import io

THINKING_TEMPLATE = (
    "Question: {question}\n"
    "Please think about this question as if you were a human pondering deeply. "
    "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
    "It's encouraged to include self-reflection or verification in the reasoning process. \n"
)

PROMPT_TEMPLATES = {
    "default": {
        "pre_prompt": "Question: {question}\n",
        "mca_post_prompt": "Please answer with the option's letter from the given choices directly.",
    },
    "thinking":
    {   
        "pre_prompt": THINKING_TEMPLATE,
        "mca_post_prompt": (
            "Please provide your detailed reasoning between the <think> </think> tags, "
            "and then answer the question with the option's letter from the given choices (e.g., A, B, etc.) within the <answer> </answer> tags."
        ),
    }
}

def read_image_with_pil(image_data):
    try:
        byte_array = None
        
        if isinstance(image_data, dict):
            if 'bytes' in image_data:
                image_bytes_dict = image_data['bytes']
            else:
                image_bytes_dict = image_data
            
            if isinstance(image_bytes_dict, dict):
                byte_keys = sorted([int(k) for k in image_bytes_dict.keys()])
                byte_array = bytes([image_bytes_dict[str(k)] for k in byte_keys])
            elif isinstance(image_bytes_dict, bytes):
                byte_array = image_bytes_dict
                
        elif isinstance(image_data, bytes):
            byte_array = image_data
            
        elif isinstance(image_data, (list, np.ndarray)):
            byte_array = bytes(image_data)
            
        else:
            raise ValueError(f"Unsupported image data type: {type(image_data)}")
        
        if byte_array is None:
            raise ValueError("Could not extract byte array from image data")
        
        image = Image.open(io.BytesIO(byte_array))
        return image
        
    except Exception as e:
        print(f"Error reading image with PIL: {e}")
        print(f"Image data type: {type(image_data)}")
        if hasattr(image_data, 'keys'):
            print(f"Image data keys: {list(image_data.keys())[:10]}...")
        return None

def evaluate_cvbench(rank, world_size, parquet_file, image_dir, model_name, output_dir, log_file, gpu_ids, num_frames=4, max_pixels=512*28*28, min_pixels=16*28*28, debug=False, batch_size=1, debug_size=12, params_dict=None, prompt_type="thinking"):   
    logger = setup_logger(rank, log_file, params_dict)
    start_time_process = time.time()

    selected_gpu = allocate_gpu(rank, gpu_ids, world_size)
    logger.info(f"Rank {rank}/{world_size} Selected GPU: {selected_gpu}, Torch Device: {torch.cuda.current_device()}")

    accelerator = Accelerator()
    device = accelerator.device
    logger.info(f"Rank {rank} using device: {device}")

    dfs = [pd.read_parquet(f) for f in parquet_file]
    df = pd.concat(dfs, ignore_index=True)
    
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
        return os.path.join(output_dir, f"CV_Bench_results_rank_{rank}.jsonl"), 0

    prompt_template = PROMPT_TEMPLATES.get(prompt_type, PROMPT_TEMPLATES["default"])
    
    for start_index in tqdm(range(0, total_samples, batch_size), desc=f"Process {rank}", total=(total_samples + batch_size - 1) // batch_size):
        batch_df = df_shard.iloc[start_index:min(start_index + batch_size, total_samples)]
        batch_messages_list = []
        batch_row_infos = []
        prompt_list = []
        
        # Prepare all batch data first
        for _, row in batch_df.iterrows():
            question = row['prompt']
                
            # Build prompt text
            prompt_text = prompt_template["pre_prompt"].format(question=question)
            prompt_text += "\n" + prompt_template["mca_post_prompt"]
            prompt_list.append(prompt_text)
            
            image_bytes_dict = row['image']['bytes']
            image = read_image_with_pil(image_bytes_dict)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
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
            prompt_text = prompt_list[i]
   
            results.append({
                'question': row['question'],
                'predicted_answer': predicted_answer,
                'ground_truth': row['answer'],
                'prompt': prompt_text,
                'type': row['type'],
                'task': row['task'],
                'source': row['source'],
            })

    # Write results to file
    process_output_file = os.path.join(output_dir, f"CV-Bench_results_rank_{rank}.jsonl")
    with open(process_output_file, 'w') as f:
        for result in results:
            json.dump(result, f, ensure_ascii=False)
            f.write("\n")

    end_time_process = time.time()
    elapsed_time_process = end_time_process - start_time_process

    elapsed_time_process_formatted = format_time(elapsed_time_process)
    logger.info(f"Rank {rank} results saved to: {process_output_file}, time usage: {elapsed_time_process_formatted}")
    return process_output_file, elapsed_time_process

def fuzzy_matching(pred):
    match = re.search(r'^[\(\[\{]?[A-D][\)\]\}]?\.?$', pred.split(' ')[0].strip(), re.I)
    if match:
        pred = match.group(0)
        pred = re.sub(r'[^A-D]', '', pred.upper())
        return pred
    return pred.strip()


def calculate_accuracy(df, source):
    source_df = df[df['source'] == source]
    if len(source_df) == 0:
        return np.nan
    accuracy = source_df['accuracy'].mean()
    return accuracy

def cvbench_aggregate_results(results):
    results_df = pd.DataFrame(results)
    if results_df.empty:
        return {'CV-Bench_Accuracy': 0.0}
    
    accuracy_2d_ade = calculate_accuracy(results_df, 'ADE20K')
    accuracy_2d_coco = calculate_accuracy(results_df, 'COCO') 
    accuracy_3d_omni = calculate_accuracy(results_df, 'Omni3D')
    
    valid_2d_accuracies = [acc for acc in [accuracy_2d_ade, accuracy_2d_coco] if not np.isnan(acc)]
    accuracy_2d = np.mean(valid_2d_accuracies) if valid_2d_accuracies else np.nan
    accuracy_3d = accuracy_3d_omni
    
    valid_type_accuracies = [acc for acc in [accuracy_2d, accuracy_3d] if not np.isnan(acc)]
    combined_accuracy = np.mean(valid_type_accuracies) if valid_type_accuracies else np.nan
    
    output = {
        'CV-Bench_Accuracy': combined_accuracy,
        '2D_Accuracy': accuracy_2d,
        '3D_Accuracy': accuracy_3d,
        'ADE20K_Accuracy': accuracy_2d_ade,
        'COCO_Accuracy': accuracy_2d_coco,
        'Omni3D_Accuracy': accuracy_3d_omni
    }
    
    eval_logger.info(f"Evaluation results: {output}")
    
    return output

def cvbench_eval(jsonl_file_path, mode="thinking"): 
    results = []   
    with open(jsonl_file_path, 'r') as f:
        for line in f:
            doc = json.loads(line)
            if mode == "thinking" and "<answer>" in doc["predicted_answer"]:
                doc["predicted_answer"] = extract_answer_text(doc["predicted_answer"])
            if fuzzy_matching(doc['predicted_answer']) == fuzzy_matching(doc['ground_truth']):
                doc['accuracy'] = 1.0
            else:
                doc['accuracy'] = 0.0
                    
            results.append(doc)
        
    aggregated_results = cvbench_aggregate_results(results)
    return aggregated_results