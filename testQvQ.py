import argparse
import os
import dashscope
import json
import time 

def QvQ(image,question):
    messages = [
        {
            "role": "user",
            "content": [
                {"image": image},
                {"text": question}
            ]
        }
    ]

    response = dashscope.MultiModalConversation.call(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key='sk-93662772a89b49889e23094815ab9aac',
        model="qvq-max",  # 此处以qvq-max为例，可按需更换模型名称。
        messages=messages,
        stream=True,
    )

    # 定义完整思考过程
    reasoning_content = ""
    # 定义完整回复
    answer_content = ""
    # 判断是否结束思考过程并开始回复
    is_answering = False

    print("=" * 20 + "思考过程" + "=" * 20)

    for chunk in response:
        # 如果思考过程与回复皆为空，则忽略
        message = chunk.output.choices[0].message
        reasoning_content_chunk = message.get("reasoning_content", None)
        if (chunk.output.choices[0].message.content == [] and
            reasoning_content_chunk == ""):
            pass
        else:
            # 如果当前为思考过程
            if reasoning_content_chunk != None and chunk.output.choices[0].message.content == []:
                print(chunk.output.choices[0].message.reasoning_content, end="")
                reasoning_content += chunk.output.choices[0].message.reasoning_content
            # 如果当前为回复
            elif chunk.output.choices[0].message.content != []:
                if not is_answering:
                    print("\n" + "=" * 20 + "完整回复" + "=" * 20)
                    is_answering = True
                print(chunk.output.choices[0].message.content[0]["text"], end="")
                answer_content += chunk.output.choices[0].message.content[0]["text"]

    # 如果您需要打印完整思考过程与完整回复，请将以下代码解除注释后运行
    # print("=" * 20 + "完整思考过程" + "=" * 20 + "\n")
    # print(f"{reasoning_content}")
    # print("=" * 20 + "完整回复" + "=" * 20 + "\n")
    # print(f"{answer_content}")
    return reasoning_content, answer_content


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0, help='起始索引（从0开始）')
    parser.add_argument('--end', type=int, default=10, help='结束索引（包含）')
    args = parser.parse_args()

    filename = r'./playground/data/eval/vqav2/llava_vqav2_mscoco_test-dev2015.jsonl'

    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            if args.start <= idx <= args.end:
                question = json.loads(line)
                image_path = "file://./playground/data/eval/vqav2/test2015/" + question["image"]
                print(f"\n\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Processing question_id: {question['question_id']} at index {idx }")
                try:
                    reasoning_content, answer_content = QvQ(image_path, question["text"])
                    result = {
                        "question_id": question["question_id"],
                        "reasoning": reasoning_content,
                        "answer": answer_content
                    }
                    with open(f'qvq_result_{args.start}_{args.end}.jsonl', 'a', encoding='utf-8') as fout:
                        fout.write(json.dumps(result, ensure_ascii=False) + '\n')
                except Exception as e:
                    print(f"Error processing question_id {question['question_id']}: {e}")
                    continue

