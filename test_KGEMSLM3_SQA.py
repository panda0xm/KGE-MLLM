import argparse
import torch
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from llava.model.builder import load_pretrained_model

from PIL import Image
from transformers import TextStreamer
from llava.conversation import conv_templates, SeparatorStyle


import json
import os
import random
import argparse
import numpy as np
import pandas as pd


'''
CUDA_VISIBLE_DEVICES=0 python test_KGEMSLM.py 
CUDA_VISIBLE_DEVICES=0,1 python -m llava.serve.cli     --model-path liuhaotian/llava-v1.5-7b     --image-file "1.jpg"    --load-4bit
CUDA_VISIBLE_DEVICES=0,1 python -m llava.serve.cli     --model-path checkpoints/llava-v1.5-7b     --image-file "1.jpg"    --load-4bit
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m llava.serve.cli     --model-path checkpoints/llava-v1.5-7b-reasoning     --image-file "./playground/data/vqav2/test2015/COCO_test2015_000000262144.jpg"
'''



#有了思维链和知识图谱，生成最终答案，使用不同的模型


#参数，多使用默认参数
parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
parser.add_argument("--image-folder", type=str, default="/home/gtl/LLaVA-main/playground/data/ScienceQA/")
parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
parser.add_argument("--answers-file", type=str, default="answer.jsonl")
parser.add_argument("--model-base", type=str, default=None)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--conv-mode", type=str, default=None)
parser.add_argument("--temperature", type=float, default=0.2)
parser.add_argument("--max-new-tokens", type=int, default=512)
parser.add_argument("--load-8bit", action="store_true")
parser.add_argument("--load-4bit", action="store_true")
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()



#初始化
model_name = get_model_name_from_path(args.model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

def KGEM_2(image_tensor,image_size,in_text,in_graph,out_think):
    conv = conv_templates["llava_v0"].copy()
    conv.system = f"A conversation between a curious human and an artificial intelligence assistant. {out_think}. According to the user's input and the assistant's thoughts, answer the user's question with A/B/C/D. only reply A/B/C/D"#系统提示词
    roles = conv.roles
    inp = f"{roles[0]}: {in_text}"

    inp = DEFAULT_IMAGE_TOKEN + '\n' + inp

    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    print(f"Prompt: {prompt}")

    #根据思考结果和知识生成输出
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            streamer=streamer,
            use_cache=True)

    out_text = tokenizer.decode(output_ids[0]).strip()
    print(f"out_text: {out_text}")
    return out_text


f=open('dataset_output.json', 'r', encoding='utf-8') 
alldata = json.load(f)
print(f"Total data entries: {len(alldata)}")
dataset = []
for data in alldata:
    id = data['id']
    question = data['question']
    print(f"Processed question_id: {id}")
    answer = data['answer']
    imagepath = data['image']

    in_image_name = os.path.join(args.image_folder, imagepath)
    in_image_name = os.path.normpath(in_image_name)
    if not os.path.exists(in_image_name):
        print(f"Image {in_image_name} does not exist.")
        continue

    image = Image.open(in_image_name).convert('RGB')
    image_size = image.size
    # Similar operation in model_worker.py
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)
    
    # 推理模型，分析用户意图
    out_think =data['out_think']
    print(f"##########out_think: {out_think}")
    
    # 二次检索
    triplets_str2 = data['triplets2']
    #triplets_str2 = '\n'.join(found_triplets2)
    print(f"found_triplets2: {triplets_str2}")
    
    # 根据用户意图和知识图谱生成答案
    out_text = KGEM_2(image_tensor, image_size, question, triplets_str2, out_think)
    print(f"###########out_text: {out_text}")
    data['out_text'] = out_text
    # 生成知识图谱
    # out_graph=KGEM_3(image_tensor,image_size,in_text,in_graph,out_think,out_text)
    dataset.append(data)

# 保存dataset为json文件
with open('dataset_output3.json', 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)