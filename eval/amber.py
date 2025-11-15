import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
project_root = os.path.abspath(os.path.join("/data/fengjw/Project/DAMA/DAMA/"))
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from PIL import Image
from matplotlib import pyplot as plt
from PIL import Image
import json
from tqdm import tqdm

model_base = None
# model_base = "/NAS/fengjw/.cache/base_models/llava-v1.5-7b"

# model_path = "/NAS/fengjw/.cache/output/DAMA/llava7b_mdpo_model_05/checkpoint-5000"
# model_path = "/NAS/fengjw/.cache/output/DAMA/main/llava7b_agpo_reference_free_gamma_15_gamma_copo_45_copocoef_02"
model_path = "/NAS/jinda/output/MDDPO-ckpt/llava15_7b_DPO-llava15_rlaifv/fine-tune"
# model_path = "/NAS/jinda/output/MDDPO-llava-13b/llava15_7b_DPO-llava15_rlaifv/fine-tune"
# output_file = "/NAS/fengjw/Project/results/DAMA/amber/mdpo_05.json"
# output_file = "/NAS/fengjw/Project/results/main/llava15_7b_agpo_True_True_True_1.5_4.5_0.2_1.0.json"
output_file = "/NAS/fengjw/Project/results/main/DAMA-7B.json"


disable_torch_init()
model_path = os.path.expanduser(model_path)
model_name = get_model_name_from_path(model_path)

print('Loading RePO...')
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name)

def inference(model, question, image_file):

    qs = DEFAULT_IMAGE_TOKEN + '\n' + question
    conv = conv_templates['v1'].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image = Image.open(image_file).convert('RGB')
    image_tensor = process_images([image], image_processor, model.config)[0]
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()


    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            do_sample=False,
            temperature=0.0,
            top_p=None,
            num_beams=1,
            max_new_tokens=128,
            use_cache=True)

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()

    return outputs

image_file='/NAS/fengjw/Project/eval_data/AMBER/image/AMBER_{id}.jpg'
question="Describe the image in detail."
amber_results = []

for id in tqdm(range(1004)):
    image_file=f'/NAS/fengjw/Project/eval_data/AMBER/image/AMBER_{id+1}.jpg'
    # print(image_file)
    outputs = inference(model, question, image_file)
    amber_results.append({
        "id":id+1,
        "response": outputs
    })

with open(output_file, "w") as f:
    json.dump(amber_results, f, indent=4)