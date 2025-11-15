from PIL import Image
import json
from tqdm import tqdm
from datasets import load_dataset

data = load_dataset("/NAS/fengjw/Project/base_datasets/RLAIF-V-Dataset")['train']
dataset = []
dataset_path = "/NAS/fengjw/Project/base_datasets/training_data_repo/gqa.json"

for item in tqdm(data):
    if 'gqa' in item['image_path']:
        question = item['question']
        chosen = item['chosen']
        rejected = item['rejected']
        image_path = item['image_path']
        origin_dataset = item['origin_dataset']
        origin_split = item['origin_split']
        idx = item['idx']
        dataset.append({
            'question': question,
            'chosen': chosen,
            'rejected': rejected,
            'image_path': image_path,
            'origin_dataset': origin_dataset,
            'origin_split': origin_split,
            'idx': idx,
        })

with open(dataset_path, 'w') as f:
    json.dump(dataset, f, indent=4)