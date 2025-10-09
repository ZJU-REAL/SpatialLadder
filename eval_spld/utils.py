import json
import re
import numpy as np
from PIL import Image
from datetime import datetime, timedelta
import logging
import torch
from decord import VideoReader, cpu

def extract_answer_text(text_with_tags):
    match = re.search(r"<answer>(.*?)</answer>", text_with_tags, re.DOTALL)
    if match:
        return match.group(1).strip()  
    else:
        return "None"

def format_time(elapsed_seconds):
    time_delta = timedelta(seconds=int(elapsed_seconds))
    hours = time_delta.seconds // 3600
    minutes = (time_delta.seconds % 3600) // 60
    seconds = time_delta.seconds % 60
    return f"{hours:02}h{minutes:02}m{seconds:02}s"

def setup_logger(rank, log_file, params_dict):
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_with_timestamp = log_file.replace(".log", f"_{timestamp_str}_rank_{rank}.log")
    logging.basicConfig(
        filename=log_file_with_timestamp,
        level=logging.INFO,
        format=f'%(asctime)s - [Rank {rank}] - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Starting process with rank {rank}")
    logger.info("Running parameters:")
    for key, value in params_dict.items():
        logger.info(f"  {key}: {value}")
    return logger

def allocate_gpu(rank, gpu_ids, world_size):
    if isinstance(gpu_ids, str):
        gpu_ids_list = gpu_ids.split(',')
    else:
        gpu_ids_list = [str(gpu_id) for gpu_id in gpu_ids]
    num_gpus_available = len(gpu_ids_list)
    if world_size == 1 and num_gpus_available > 1:
        selected_gpu = ",".join(gpu_ids_list)
    elif world_size > 1:
        if rank < num_gpus_available:
            selected_gpu = gpu_ids_list[rank]
            torch.cuda.set_device(rank)
        else:
            selected_gpu = gpu_ids_list[rank % num_gpus_available]
            torch.cuda.set_device(rank)
            logger = logging.getLogger(__name__)
            logger.warning(f"Rank {rank}: Not enough GPUs, reusing GPU: {selected_gpu}. Reduce number of processes or increase GPUs.")
    else: 
        selected_gpu = gpu_ids_list[rank % num_gpus_available] if gpu_ids_list else "0"  # Default to GPU 0

    logger = logging.getLogger(__name__)
    logger.info(f"Rank {rank}: Selected GPU: {selected_gpu}, CUDA_VISIBLE_DEVICES: {gpu_ids}")
    return selected_gpu

def load_video_frames(video_path, num_frames=4, target_pixels=512*28*28):
    """Use decord to read video frames and return timestamps of those frames."""
    
    def calculate_target_size(original_width, original_height, target_pixels):
        """Calculate target size maintaining aspect ratio based on target pixels."""
        # Calculate aspect ratio
        aspect_ratio = original_width / original_height
        
        # Calculate target dimensions
        # target_pixels = target_width * target_height
        # aspect_ratio = target_width / target_height
        # So: target_height = sqrt(target_pixels / aspect_ratio)
        # And: target_width = sqrt(target_pixels * aspect_ratio)
        
        target_height = int((target_pixels / aspect_ratio) ** 0.5)
        target_width = int((target_pixels * aspect_ratio) ** 0.5)
        
        return target_width, target_height
    
    def resize_image_to_target_pixels(image, target_pixels):
        """Resize image maintaining aspect ratio to approximate target pixels."""
        original_width, original_height = image.size
        target_width, target_height = calculate_target_size(original_width, original_height, target_pixels)
        
        return image.resize((target_width, target_height), Image.Resampling.LANCZOS)

    try:
        vr = VideoReader(video_path, ctx=cpu())
        total_frames = len(vr)
        video_duration = total_frames / vr.get_avg_fps() if vr.get_avg_fps() > 0 else total_frames / 30  # Estimate duration
        video_duration = int(video_duration)
        target_frames = num_frames
        
        frame_indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frames_np = vr.get_batch(frame_indices).asnumpy()
        
        # Resize frames based on target_pixels
        frames_pil = [resize_image_to_target_pixels(Image.fromarray(f), target_pixels) for f in frames_np]
        
        timestamps = [int(idx / vr.get_avg_fps()) for idx in frame_indices] if vr.get_avg_fps() > 0 else [int(idx / 30) for idx in frame_indices]  # Get integer timestamps
        
        return frames_pil, timestamps, video_duration
    except Exception as e:
        return None, None, None
    