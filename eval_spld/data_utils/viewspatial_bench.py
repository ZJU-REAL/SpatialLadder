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

SCANNET_TASKS = [
    "Camera perspective - Relative Direction",
    "Person perspective - Scene Simulation Relative Direction"
]

COCO_TASKS = [
    "Camera perspective - Object View Orientation",
    "Person perspective - Object View Orientation",
    "Person perspective - Relative Direction"
]

PERSON_TASKS = [
    "Person perspective - Scene Simulation Relative Direction",
    "Person perspective - Object View Orientation",
    "Person perspective - Relative Direction"
]

CAMERA_TASKS = [
    "Camera perspective - Relative Direction",
    "Camera perspective - Object View Orientation"
]

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

def evaluate_viewspatial_bench(rank, world_size, json_file, image_dir, model_name, output_dir, log_file, gpu_ids, num_frames=4, max_pixels=512*28*28, min_pixels=16*28*28, debug=False, batch_size=1, debug_size=12, params_dict=None, prompt_type="thinking"):    
    logger = setup_logger(rank, log_file, params_dict)
    start_time_process = time.time()

    selected_gpu = allocate_gpu(rank, gpu_ids, world_size)
    logger.info(f"Rank {rank}/{world_size} Selected GPU: {selected_gpu}, Torch Device: {torch.cuda.current_device()}")
    
    accelerator = Accelerator()
    device = accelerator.device
    logger.info(f"Rank {rank} using device: {device}")

    # Read JSON file - try different formats
    try:
        # First try reading as lines format (JSONL)
        df = pd.read_json(json_file, lines=True)
        logger.info(f"Successfully loaded JSON file as JSONL format")
    except:
        try:
            # Try reading as standard JSON array
            df = pd.read_json(json_file)
            logger.info(f"Successfully loaded JSON file as standard JSON format")
        except Exception as e:
            logger.error(f"Failed to read JSON file {json_file}: {e}")
            raise e
    
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
        return os.path.join(output_dir, f"ViewSpatial-Bench_results_rank_{rank}.jsonl"), 0

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
            image_paths = [image_path.replace("ViewSpatial-Bench", image_dir) for image_path in row["image_path"]]
            
            question = row['question']
            options = row.get('choices')
            if options is not None and len(options) > 0:
                # Handle options whether they are lists or need conversion
                if isinstance(options, list):
                    options = options
                elif isinstance(options, str):
                    options = options.split('\n')
                else:
                    options = options.tolist() if hasattr(options, 'tolist') else list(options)
                question += "\nOptions:\n" + "\n".join(options)
                
            # Build prompt text
            prompt_text = prompt_template["pre_prompt"].format(question=question) + "\n" + prompt_template["mca_post_prompt"]                    
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

        if not batch_messages_list:
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
            ground_truth = row['answer']
            question_type = row['question_type']
            prompt_text = prompt_list[i]
   
            results.append({
                'question': row['question'],
                'ground_truth': ground_truth,
                'predicted_answer': predicted_answer,
                'task': question_type,
                'prompt': prompt_text
            })

    # Write results to file
    process_output_file = os.path.join(output_dir, f"ViewSpatial-Bench_results_rank_{rank}.jsonl")
    with open(process_output_file, 'w') as f:
        for result in results:
            json.dump(result, f, ensure_ascii=False)
            f.write("\n")

    end_time_process = time.time()
    elapsed_time_process = end_time_process - start_time_process

    elapsed_time_process_formatted = format_time(elapsed_time_process)
    logger.info(f"Rank {rank} results saved to: {process_output_file}, time usage: {elapsed_time_process_formatted}")
    return process_output_file, elapsed_time_process

def viewspatial_bench_aggregate_results(results):
    results_df = pd.DataFrame(results)
    if results_df.empty:
        eval_logger.warning("Input results for viewspatial_bench_aggregate_results is empty.")
        empty_output = {}
        empty_output['overall_accuracy'] = np.nan
        return empty_output
    
    output = {} 

    # --- Calculate per-task average accuracies AND get task counts ---
    task_accuracies = {}
    task_counts = results_df.groupby('task').size().to_dict() 

    for task_name, group_df in results_df.groupby('task'):
        if 'accuracy' in group_df.columns and not group_df['accuracy'].isnull().all():
            task_accuracies[task_name] = group_df['accuracy'].mean()
        else:
            task_accuracies[task_name] = np.nan
            if task_name not in task_counts: 
                 task_counts[task_name] = 0
            eval_logger.warning(f"Task '{task_name}' has no valid accuracy data or was empty. Its accuracy set to NaN.")

    output.update(task_accuracies)

    # --- Calculate WEIGHTED overall_accuracy, scannet_accuracy and coco_accuracy ---
    weighted_sum_overall, weighted_sum_scannet, weighted_sum_coco, weighted_sum_person, weighted_sum_camera = 0.0, 0.0, 0.0, 0.0, 0.0
    total_weight_overall, total_weight_scannet, total_weight_coco, total_weight_person, total_weight_camera = 0.0, 0.0, 0.0, 0.0, 0.0
    valid_overall_tasks_contributing = 0

    for task_name in task_accuracies:
        if task_name in task_counts and not pd.isna(task_accuracies[task_name]):
            accuracy = task_accuracies[task_name]
            count = task_counts.get(task_name, 0) 
            if count > 0: 
                weighted_sum_overall += accuracy * count
                total_weight_overall += count
                valid_overall_tasks_contributing +=1
                if task_name in SCANNET_TASKS:
                    weighted_sum_scannet += accuracy * count
                    total_weight_scannet += count
                elif task_name in COCO_TASKS:
                    weighted_sum_coco += accuracy * count
                    total_weight_coco += count
                if task_name in PERSON_TASKS:
                    weighted_sum_person += accuracy * count
                    total_weight_person += count
                elif task_name in CAMERA_TASKS:
                    weighted_sum_camera += accuracy * count
                    total_weight_camera += count

    if total_weight_overall > 0:
        output['overall_accuracy'] = weighted_sum_overall / total_weight_overall
    else:
        output['overall_accuracy'] = np.nan 
    if total_weight_scannet > 0:
        output['scannet_accuracy'] = weighted_sum_scannet / total_weight_scannet
    else:
        output['scannet_accuracy'] = np.nan
    if total_weight_coco > 0:
        output['coco_accuracy'] = weighted_sum_coco / total_weight_coco
    else:
        output['coco_accuracy'] = np.nan
    if total_weight_person > 0:
        output['person_accuracy'] = weighted_sum_person / total_weight_person
    else:
        output['person_accuracy'] = np.nan
    if total_weight_camera > 0:
        output['camera_accuracy'] = weighted_sum_camera / total_weight_camera
    else:
        output['camera_accuracy'] = np.nan

    def nan_to_null_serializer(obj):
        if isinstance(obj, float) and np.isnan(obj):
            return None
        if isinstance(obj, np.generic): 
             if np.isnan(obj):
                 return None
             return obj.item() 
        return obj

    eval_logger.info(f"ViewSpatial-Bench Weighted Evaluation results: {json.dumps(output, indent=2, default=nan_to_null_serializer)}")
    
    return output

def fuzzy_matching(pred):
    match = re.search(r'^[A-D]\.?$', pred.split(' ')[0].strip())
    if match:
        pred=match.group(0).rstrip('.').upper()
        pred=pred.strip()
        return pred
    return pred.strip() 

def viewspatial_bench_eval(jsonl_file_path, mode="thinking"):
    results = []
    with open(jsonl_file_path, 'r') as f:
        for line in f:
            doc = json.loads(line)
            if mode in ("thinking", "map") and "<answer>" in doc["predicted_answer"] :
                doc["predicted_answer"]=extract_answer_text(doc["predicted_answer"])
            if fuzzy_matching(doc["predicted_answer"]) == fuzzy_matching(doc["ground_truth"][0]):
                doc["accuracy"] = 1.0
            else:
                doc["accuracy"] = 0.0
            results.append(doc)

    aggregated_results = viewspatial_bench_aggregate_results(results)  # Aggregate results after processing
    return aggregated_results
