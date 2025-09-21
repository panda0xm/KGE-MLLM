import requests
import json
import time     

#nohup python3 triplet.py > triplet.log 2>&1 &


#读取文件前10行
def read_first_n_lines(file_path, n=10):
    with open(file_path, 'r', encoding='utf-8') as file:
        for _ in range(n):
            line = file.readline()
            if not line:  # 如果文件结束，提前退出
                break
            print(line.strip()) 

#读取json中数组的第一条记录
def read_json(file_path, n=1):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        print(len(data))  
        if isinstance(data, list) and len(data) > 0:
            for i in range(len(data)):
                if i >= n:
                    break
                id=data[i]["id"]
                text=json.dumps(data[i], ensure_ascii=False, indent=2)
                print(f"\n\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Processing question_id: {id} at index {i}")
                print(text)
                ret=qwq(text)  # 调用函数，生成文本并打印
                #ret={'model': 'qwq', 'created_at': '2025-06-29T08:25:35.421505757Z', 'message': {'role': 'assistant', 'content': '<think>\n好的，我现在需要处理用户提供的JSON数据，并从中提取三元组。首先，\n\n不过用户的问题问的是公交车是在行驶还是停在路边，所以主要点在于驾驶中，因此“driving_status”更准确。\n</think>\n\n<h, r, t>  \n1. <bus, color, white and red>  \n2. <bus, back_feature, advertisement>  \n3. <bus, driving_status, driving down the street>  \n4. <street, crowded_with, people and other vehicles>'}, 'done_reason': 'stop', 'done': True, 'total_duration': 71088403587, 'load_duration': 26292754384, 'prompt_eval_count': 272, 'prompt_eval_duration': 440000000, 'eval_count': 1257, 'eval_duration': 43855000000}
                print(ret)

                #提取</think>之后的部分
                if '</think>' in ret['message']['content']:
                    think_content = ret['message']['content'].split('</think>')[0].strip()
                    content = ret['message']['content'].split('</think>')[1].strip()
                    print("\n===========提取的内容:===================\n", content)
                    #以json格式保存，包括id和三元组
                    result = {
                        "id": id,
                        "think": think_content,
                        "answer": content
                    }
                    with open('vqav2_triplet.jsonl', 'a', encoding='utf-8') as output_file:
                        output_file.write(json.dumps(result, ensure_ascii=False) + '\n')
                else:
                    print("没有找到</think>标记")
        else:
            print("文件内容不是一个数组或数组为空")


def qwq(text):
    # 生成文本
    response = requests.post(
        "http://219.216.65.79:11434/api/chat",
        json={
            "model": "qwq",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a knowledge graph construction tool. Output triplets in<head-entity, relation, tail-entity> or <entity, attribute, value> format. Strictly follow the output format and should not contain any other content."
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            "stream": False
        }
    )
    return response.json()

#read_first_n_lines('llava_v1_5_mix665k.json', 100)  # 调用函数，读取前10行并打印
read_json('llava_v1_5_mix665k.json', 100000)  # 调用函数，读取前10条记录并打印

