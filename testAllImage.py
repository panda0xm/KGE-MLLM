import os
import json
from PIL import Image


data_path='./playground/data/finetune_for_reasoning.json'
image_folder ='./playground/data'
list_data_dict = json.load(open(data_path, "r"))
for i in range(len(list_data_dict)):
    if i<0:
        continue
    sources = list_data_dict[i]
    if isinstance(i, int):
        sources = [sources]
    assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
    if 'image' in sources[0]:
        image_file = list_data_dict[i]['image']
        image_folder = image_folder
        image_file_path=os.path.join(image_folder, image_file)
        print(f"Processing image {i} : {image_file_path}")
        image = Image.open(image_file_path).convert('RGB')