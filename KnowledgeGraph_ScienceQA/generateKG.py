import json

input_path = 'kg_mmmu_pro.jsonl'  # 输入文件路径
output_path = 'triplets_mmmu_pro.txt'  # 输出文件路径

with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
    for line in infile:
        data = json.loads(line)
        answer = data.get('answer', '')
        # 按行分割三元组并写入输出文件
        for triplet in answer.strip().split('\n'):
            triplet = triplet.strip()
            if triplet:
                outfile.write(triplet + '\n')