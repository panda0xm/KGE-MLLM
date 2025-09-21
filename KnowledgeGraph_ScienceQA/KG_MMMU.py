import requests
import json
import time 
import pandas as pd
    

#读数据
def get_response_by_id(data, target_id):
    """
    根据id查找response
    :param data: list, 数据列表
    :param target_id: str, 目标id
    :return: str, response内容或None
    """
    #data=None
    #print(data)
    for item in data:
        if item['id'] == target_id:
            return item.get('response', None)
    return None
#读数据
def read_data(response_data,file_path):
    dataset=pd.read_parquet(file_path)
        
    j=0
    for index in range(len(dataset)):
        id=dataset.loc[index,'id']    
    
        #通过id查询已经保存的结果文件，如果有结果，则跳过
        try:
            with open('kg_mmmu.jsonl', 'r', encoding='utf-8') as result_file:
                results = [json.loads(line) for line in result_file]
                if any(result['id'] == id for result in results):
                    print(f"Skipping id {id} as it has already been processed.")
                    continue
        except FileNotFoundError:
                    print("Result file not found, will create a new one.")

        question=dataset.loc[index,'question']
        subfield=dataset.loc[index,'subfield']
        print(id)
        print(question)
        print(subfield)

        response=get_response_by_id(response_data,id)
        
        print(response)
        input_text=question + ' ' + response

        content=ollama(input_text)

        result = {
                    "id": id,
                    "model":"qwen2.5:72b",
                    "answer": content
                }
        with open('kg_mmmu.jsonl', 'a', encoding='utf-8') as output_file:
            output_file.write(json.dumps(result, ensure_ascii=False) + '\n')

        j=j+1
        if j>10000:
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
    subjects = [
        "Accounting", "Agriculture", "Architecture_and_Engineering", "Art", "Art_Theory",
        "Basic_Medical_Science", "Biology", "Chemistry", "Clinical_Medicine", "Computer_Science",
        "Design", "Diagnostics_and_Laboratory_Medicine", "Economics", "Electronics", "Energy_and_Power",
        "Finance", "Geography", "History", "Literature", "Manage",
        "Marketing", "Materials", "Math", "Mechanical_Engineering", "Music",
        "Pharmacy", "Physics", "Psychology", "Public_Health", "Sociology"
]
    for field in subjects:
        response_data = json.load(open(f'D:\\MMMU-main\\mmmu\\example_outputs\\qwen_vl\\{field}\\output.json'))
        read_data(response_data,f"D:/MMMU/{field}/validation-00000-of-00001.parquet")
    