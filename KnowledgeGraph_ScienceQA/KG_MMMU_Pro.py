import requests
import json
import time 
import pandas as pd
import pyarrow.dataset as ds    

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
    
    responses = [json.loads(line) for line in response_data]

    j=0
    for res in responses:
        id= res['id']
        question = res['question']
        response = res['response']

        #通过id查询已经保存的结果文件，如果有结果，则跳过
        try:
            with open('kg_mmmu_pro.jsonl', 'r', encoding='utf-8') as result_file:
                results = [json.loads(line) for line in result_file]
                if any(result['id'] == id for result in results):
                    print(f"Skipping id {id} as it has already been processed.")
                    continue
        except FileNotFoundError:
                    print("Result file not found, will create a new one.")

        print(id)
        print(question)
        print(response)

        input_text=question + ' ' + response

        content=ollama(input_text)

        result = {
                    "id": id,
                    "model":"qwen2.5:72b",
                    "answer": content
                }
        with open('kg_mmmu_pro.jsonl', 'a', encoding='utf-8') as output_file:
            output_file.write(json.dumps(result, ensure_ascii=False) + '\n')
        j=j+1
        if j>3000:
            break

    # # 创建数据集对象
    # dataset = ds.dataset(file_path, format='parquet')

    # # 转换为Pandas DataFrame
    # df = dataset.to_table().to_pandas()

    # j=0
    # for index in range(len(df)):
    #     id=df.loc[index,'id']    
    
    #     #通过id查询已经保存的结果文件，如果有结果，则跳过
    #     try:
    #         with open('kg_mmmu_pro.jsonl', 'r', encoding='utf-8') as result_file:
    #             results = [json.loads(line) for line in result_file]
    #             if any(result['id'] == id for result in results):
    #                 print(f"Skipping id {id} as it has already been processed.")
    #                 continue
    #     except FileNotFoundError:
    #                 print("Result file not found, will create a new one.")

    #     question=df.loc[index,'question']
    #     subfield=df.loc[index,'subject']
    #     print(id)
    #     print(question)
    #     print(subfield)

    #     response=get_response_by_id(response_data,id)
        
    #     print(response)
    #     input_text=question + ' ' + response

    #     content=ollama(input_text)

    #     result = {
    #                 "id": id,
    #                 "model":"qwen2.5:72b",
    #                 "answer": content
    #             }
    #     with open('kg_mmmu_pro.jsonl', 'a', encoding='utf-8') as output_file:
    #         output_file.write(json.dumps(result, ensure_ascii=False) + '\n')

    #     j=j+1
    #     if j>3:
    #         break



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
    response_data = open('D:\\MMMU-main\\mmmu-pro\\output\\gpt-4o_standard(10 options)_cot.jsonl')
    read_data(response_data,"D:\\MMMU_Pro\\standard (10 options)")
    