import argparse
import torch
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from llava.model.builder import load_pretrained_model

from PIL import Image
from transformers import TextStreamer
from llava.conversation import conv_templates, SeparatorStyle

'''
CUDA_VISIBLE_DEVICES=0 python test_KGEMSLM.py 
CUDA_VISIBLE_DEVICES=0,1 python -m llava.serve.cli     --model-path liuhaotian/llava-v1.5-7b     --image-file "1.jpg"    --load-4bit


'''

#参数，多使用默认参数
parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
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
    conv.system = "A conversation between a curious human and an artificial intelligence assistant. Assistant analyzes human behavior and purpose, provides relevant thoughts, and offers possible answers from different perspectives."#系统提示词
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
    pass
    key_words = []
    return key_words
    
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



#检索
#获取关键词
key_words=Get_key_word(in_text)
in_graph=Search(graph_all,key_words)

#生成，此部分可替换为llava1.5-7b，Qwen-VL-7B,deepseek-vl等模型
out_think,out_text,out_graph=KGEM(in_image_name,in_text,in_graph)

#完善知识图谱
graph_all=Merge_graph(graph_all,out_graph)