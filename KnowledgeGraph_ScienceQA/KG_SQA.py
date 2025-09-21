import requests
import json
import time 


#读数据

def read_data(file_path):
    data = json.load(open(file_path))
    j=0
    for id in data.keys():


        try:
            with open('sqa_kg.jsonl', 'r', encoding='utf-8') as result_file:
                results = [json.loads(line) for line in result_file]
                if any(result['id'] == id for result in results):
                    print(f"Skipping id {id} as it has already been processed.")
                    continue
        except FileNotFoundError:
                    print("Result file not found, will create a new one.")
            
        question=data[id]['question']
        imagename=data[id]['image']
        imagepath=f'.\\train\\{id}\\{imagename}'
        lecture=data[id]['lecture']
        solution=data[id]['solution']
        print(id)
        print(question)
        print(lecture)
        print(solution)
        print(imagepath)
        input_text=question + ' ' + lecture + ' ' + solution 
        content=ollama(input_text)

        result = {
                    "id": id,
                    "model":"qwen2.5:72b",
                    "answer": content
                }
        with open('sqa_kg.jsonl', 'a', encoding='utf-8') as output_file:
            output_file.write(json.dumps(result, ensure_ascii=False) + '\n')

        j=j+1
        if j>30000:
            break
#生成结果

def ollama(text):
    # 生成文本
    response = requests.post(
        "http://219.216.65.107:11434/api/chat",
        json={
            "model": "qwen2.5:72b",
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
    ret= response.json()
    print(ret)
    return ret['message']['content']
    
#主函数
if __name__ == "__main__":
    read_data('D:/ScienceQA/text/problems.json')