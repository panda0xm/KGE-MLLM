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

#参数，多使用默认参数


parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default="checkpoints/llava-v1.5-7b-reasoning")
parser.add_argument("--image-folder", type=str, default="/home/gtl/LLaVA-main/playground/data/ScienceQA/")
parser.add_argument("--question-file", type=str, default="./playground/data/ScienceQA/text/problems.json")
parser.add_argument("--answers-file", type=str, default="dataset_output.json")



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


triplets = []
try:
    with open('triplets.txt', 'r', encoding='utf-8') as f:
        for line in f:
            triplet = line.strip()
            if triplet:
                triplets.append(triplet)
except FileNotFoundError:
    print("triplets.txt 文件未找到")


#输入
in_image_name='1.jpg'
in_text='discribe the image\n'
graph_all=None


def KGEM(image_name,in_text,in_graph):
    image = Image.open(image_name).convert('RGB')
    image_size = image.size
    # Similar operation in model_worker.py
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)



    out_think=KGEM_1(image,image_tensor,image_size,in_text,in_graph)
    out_text=KGEM_2(image_tensor,image_size,in_text,in_graph,out_think)
    out_graph=KGEM_3(image_tensor,image_size,in_text,in_graph,out_think,out_text)
    return out_think,out_text,out_graph
    
def KGEM_1(image,image_tensor,image_size,in_text,in_graph):
    conv = conv_templates["llava_v0"].copy()
    conv.system = f"A conversation between a curious human and an artificial intelligence assistant. Assistant analyzes human behavior and purpose，Refer to the following knowledge{in_graph}, provides relevant thoughts, and offers possible answers from different perspectives."#系统提示词
    roles = conv.roles
    inp = f"{roles[0]}: {in_text}"

    if image is not None:
        # first message
        if model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        image = None

    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    print(f"Prompt: {prompt}")

    #根据图片和问题加上现有知识，分析思考用户意图
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

    out_think = tokenizer.decode(output_ids[0]).strip()
    print(f"out_think: {out_think}")
    return out_think
    
def KGEM_2(image_tensor,image_size,in_text,in_graph,out_think):
    conv = conv_templates["llava_v0"].copy()
    conv.system = f"A conversation between a curious human and an artificial intelligence assistant. {out_think}. According to the user's input and the assistant's thoughts, provide a detailed answer to the user's question."#系统提示词
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
    
def KGEM_3(image_tensor,image_size,in_text,in_graph,out_think,out_text):
    conv = conv_templates["llava_v0"].copy()
    conv.system = f"A conversation between a curious human and an artificial intelligence assistant.{out_think}.{out_text}"#系统提示词
    roles = conv.roles
    inp = f"{roles[0]}: {in_text}. Organize the conversation content into a knowledge graph in triplet form."

    inp = DEFAULT_IMAGE_TOKEN + '\n' + inp

    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    print(f"Prompt: {prompt}")

    #根据输入和输出内容总结知识，以三元组形式输出    
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

    out_graph = tokenizer.decode(output_ids[0]).strip()
    print(f"out_graph: {out_graph}")
    return out_graph
    
    
def Get_key_word(in_text):
    #根据关键词检索知识图谱，将问题中逐个词查找三元组，并返回查找到的三元组
    key_words = []
    found_triplets = []
    

    # 分词（简单按空格分割，可根据需要改进分词方式）
    words = in_text.strip().replace('\n', ' ').split(' ')
    words = [w.strip(',.?!"\'') for w in words if w.strip(',.?!"\'')]

    # 查找每个词是否出现在三元组中
    for word in words:
        for triplet in triplets:
            if word and word.lower() in triplet.lower():
                key_words.append(word)
                found_triplets.append(triplet)
                break  # 一个词只需出现一次即可

    # 去重
    key_words = list(set(key_words))
    found_triplets = list(set(found_triplets))
    return key_words, found_triplets
    
def Merge_graph(graph_all,out_graph):
    pass
    #去掉重复
    #合并
    #清洗
    return graph_all

def Search(graph_all,key_words):
    pass
    #根据关键词检索知识图谱
    in_graph = {}
    return in_graph



def process_scienceQA_response(filename):
    dataset = []
    with open(filename, 'r', encoding='utf-8') as f:
        alldata = json.loads(f.read())
        
        for id in alldata.keys():
            question=alldata[id]['question']
            choices=alldata[id]['choices']

            choices = '\n'.join(choices)

            answer=alldata[id]['answer']
            imagename=alldata[id]['image']
            #imagepath=f'.\\train\\{id}\\{imagename}'
            lecture=alldata[id]['lecture']
            solution=alldata[id]['solution']
            split=alldata[id]['split']
            if(imagename is None   or imagename == ''):
                print(f"Skipping question_id: {id} due to missing image")
                continue
            if split == 'test':
                print(f"Processed question_id: {id}")
                data = {}
                data['id']  = id
                imagepath = os.path.join('train', str(id), imagename)
                data['image'] = imagepath
                data['answer'] = answer
                print(f"##############question: {question}'\n'{choices}")
                data['question'] = question+'\n'+choices
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

                #检索                #获取关键词
                key_words, found_triplets=Get_key_word(question+choices)
                triplets_str = '\n'.join(found_triplets)
                print(f"found_triplets: {triplets_str}")
                data['triplets'] = triplets_str
                #应该将查询到的子图扩大，此处暂时省略

               

                #推理模型，分析用户意图
                out_think=KGEM_1(image,image_tensor,image_size,question+'\n'+choices,triplets_str)
                print(f"##########out_think: {out_think}")
                data['out_think'] = out_think
                #二次检索
                key_words2, found_triplets2=Get_key_word(out_think)
                triplets_str2 = '\n'.join(found_triplets2)
                print(f"found_triplets2: {triplets_str2}")
                data['triplets2'] = triplets_str2
                #根据用户意图和知识图谱生成答案
                out_text=KGEM_2(image_tensor,image_size,question+'\n'+choices,triplets_str2,out_think)
                print(f"###########out_text: {out_text}")
                data['out_text'] = out_text
                #生成知识图谱
                #out_graph=KGEM_3(image_tensor,image_size,in_text,in_graph,out_think,out_text)

                #生成，此部分可替换为llava1.5-7b，Qwen-VL-7B,deepseek-vl等模型
                #out_think,out_text,out_graph=KGEM(in_image_name,question,found_triplets)

                dataset.append(data)
                
    return dataset


dataset = process_scienceQA_response(args.question_file)

# 保存dataset为json文件
with open(args.answer_file, 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

#完善知识图谱
#graph_all=Merge_graph(graph_all,out_graph)