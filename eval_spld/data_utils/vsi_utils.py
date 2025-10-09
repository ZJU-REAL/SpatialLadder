import re
from functools import partial
import os
import json
import pandas as pd
import numpy as np

MCA_QUESTION_TYPES = [
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_direction",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
]
NA_QUESTION_TYPES = [
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
]

QUESTION_TYPES_W_BBOX = [
    "object_size_estimation",
    "object_abs_distance",
    "object_rel_distance",
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_direction",
    "object_rel_distance",
    "obj_appearance_order",
]

QUESTION_TYPES_WO_BBOX = [
    "object_counting",
    "room_size_estimation",
]

METRICS_FOR_MCA = {
    "accuracy": "exact_match",
}

METRICS_FOR_NA = {
    "MRA:.5:.95:.05": "partial(mean_relative_accuracy, start=.5, end=.95, interval=.05)",
}

WORST_CASE_FOR_METRICS = {
    "accuracy": 0.,
    "MRA:.5:.95:.05": 0.,
}

THINKING_TEMPLATE = (
    "Question: {question}\n"
    "Please think about this question as if you were a human pondering deeply. "
    "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
    "It's encouraged to include self-reflection or verification in the reasoning process. \n"
)

PROMPT_TEMPLATES = {
    "default": 
    {
        "pre_prompt": "Question: {question}\n",
        "mca_post_prompt": "Please answer with the option's letter from the given choices (e.g., A, B, etc.) directly.",
        "na_post_prompt": "Please answer the question using a numerical value (e.g., 42 or 3.1) directly."
    },
    "thinking":
    {   
        "pre_prompt": THINKING_TEMPLATE,
        "mca_post_prompt": (
            "Please provide your detailed reasoning between the <think> </think> tags, "
            "and then answer the question with the option's letter from the given choices (e.g., A, B, etc.) within the <answer> </answer> tags."
        ),
        "na_post_prompt": (
            "Please provide your detailed reasoning between the <think> </think> tags, "
            "and then answer the question with a numerical value (e.g., 42 or 3.1) within the <answer> </answer> tags."
        )
    },
}

def exact_match(pred, target):
    return 1. if pred.lower() == target.lower() else 0.

def abs_dist_norm(pred, target):
    return abs(pred - target) / target

def mean_relative_accuracy(pred, target, start, end, interval):
    if pred is None or target is None:
        return 0.
    
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()

def to_float(pred):
    try:
        pred = float(pred)
    except BaseException as e:
        pred = None
    return pred

def fuzzy_matching_num(pred):

    pred = pred.strip().lower() 

    number_words = {
        'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
        'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
        'eleven': '11', 'twelve': '12', 'thirteen': '13', 'fourteen': '14', 'fifteen': '15',
        'sixteen': '16', 'seventeen': '17', 'eighteen': '18', 'nineteen': '19', 'twenty': '20',
        'thirty': '30', 'forty': '40', 'fifty': '50', 'sixty': '60', 'seventy': '70', 'eighty': '80', 'ninety': '90',
        'zero': '0', 'a': '1', 'an': '1'  
    }

    for word, digit in number_words.items():
        if re.search(r'\b' + word + r'\b', pred):  
            return digit  

    number_match = re.search(r'(\d+(\.\d+)?)', pred)  
    if number_match:
        return number_match.group(1)  

    return "None"  

def vsibench_process_results(doc): 
    if doc['question_type'] in MCA_QUESTION_TYPES:
        for key, value in METRICS_FOR_MCA.items():
            doc['predicted_answer']=doc['predicted_answer'].replace("Answer:","")
            doc[key] = eval(value)(fuzzy_matching(doc['predicted_answer']), doc['ground_truth'])
    elif doc['question_type'] in NA_QUESTION_TYPES:
        for key, value in METRICS_FOR_NA.items():
            try:
                doc[key] = eval(value)(to_float(fuzzy_matching_num(doc['predicted_answer'])), to_float(doc['ground_truth'])) # Use 'predicted_answer'
            except TypeError:
                doc[key] = WORST_CASE_FOR_METRICS[key]
    else:
        raise ValueError(f"Unknown question type: {doc['question_type']}")

    return doc 

def fuzzy_matching(pred):
    match = re.search(r'^[A-D]\.?$', pred.split(' ')[0].strip())
    if match:
        pred=match.group(0).rstrip('.').upper()
        pred=pred.strip()
        return pred
    return pred.strip() 