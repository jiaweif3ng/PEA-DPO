import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
project_root = os.path.abspath(os.path.join("/data/fengjw/Project/DAMA/DAMA/llava", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import io
import PIL.Image as PIL_image
import torch.utils.data as torch_data
import json
from functools import partial
from llava.train.train_utils import encode_multimodal_preference_sample, SFT_collator_fn, preprocess_v1
import torch
import tqdm
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import transformers
import copy
import pandas as pd
import datasets
from llava.model import *
from llava import conversation as conversation_lib
import itertools

def bytes_to_PIL_image(img_buffer):
    img_io = io.BytesIO(img_buffer)
    img_io.seek(0)
    image = PIL_image.open(img_io).convert('RGB')
    return image

@dataclass
class ModelArguments:
    # model_name_or_path: Optional[str] = field(default="/NAS/fengjw/.cache/base_models/llava-v1.5-7b")
    model_name_or_path: Optional[str] = field(default="/NAS/fengjw/.cache/output/DAMA/ablation/llava7b_dpo_reference_free_bate_01")
    version: Optional[str] = field(default="llava_v1") # llava 1.5
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default="openai/clip-vit-large-patch14-336")
    mm_vision_select_layer: Optional[int] = field(
        default=-2)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='mlp2x_gelu')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    lazy_preprocess: bool = True
    is_multimodal: bool = False
    image_token_len: int = 0
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    parquet: bool = False
    data_source_names: str = ''
    data_source_weights: str = '1'
    eval_data_source_names: Optional[str] = field(default=None)
    # data_dir: str = '/NAS/fengjw/.cache/base_datasets/test_522/RLAIF-V-Dataset_22k' # !!!
    data_dir: str = '/NAS/fengjw/.cache/base_datasets/test_522/RLAIF-V-Dataset_22k_logps' # !!!
    kto_win_data_source_names: str = '100'
    kto_win_data_source_weights: str = '100'
    kto_rej_data_source_names : str = '100'
    kto_rej_data_source_weights: str = '100'

    dpo_beta: float = 0.5
    dpo_token_weight: float = 1.0

    shuffle_data: bool = True


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    force_fsdp: bool = field(default=False)
    model_max_length: int = field(
        default=2048,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    max_steps: int = field(default=2672)
    no_randaug: bool = False

    task: str = field(
        default='DPO',
        metadata={
            'help': 'LM for language modeling. DPO for direct preference optimization'
        }
    )
    dpo_use_average: bool = False
    dpo_token_weighted: bool = False

    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    fully_tune: bool = True
    output_dir: str = field(default="/NAS/fengjw/.cache/output/llava7b_dpo_model")

class PreferenceInferenceDataset(torch_data.Dataset):
    def __init__(self,
                 data,
                 tokenizer,
                 image_token_len,
                 img_processor,
                 use_im_start_end=True):

        self.data = data

        self.mm_cfg = {
            'image_processor': img_processor,
            'is_multimodal': True,
            'image_token_len': image_token_len,
            'use_im_start_end': use_im_start_end,
            'keep_image_tag': True
        }
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        sample = self.data[index]
        metainfo = {
            "origin_dataset": sample['origin_dataset'],
            "origin_split": json.loads(sample['origin_split']),
            "origin_idx": sample['idx'],
            "image_id": sample['image_path'],
        }
        question = {'from': 'human', 'value': f"<image>\n{sample['question']}"}
        chosen = {'from': 'gpt', 'value': sample['chosen']}
        rejected = {'from': 'gpt', 'value': sample['rejected']}
        image = bytes_to_PIL_image(sample['image']['bytes'])
        masked_images_file = sample['masked_images_file']

        formated_sample = {
            'image': image,
            "question": question,
            "chosen": chosen,
            "rejected": rejected,
            "masked_images_file": masked_images_file,
            "idx": sample['idx'],
            "metainfo": metainfo
        }
        # print(f"Sample: {formated_sample}")
        preprocess_func= partial(preprocess_v1, has_image=True)
        rej_data_dict, win_data_dict = encode_multimodal_preference_sample(
            formated_sample, self.tokenizer, self.mm_cfg, preprocess_func=preprocess_func)
        return rej_data_dict, win_data_dict # win_data_dict{'inputs_ids', 'labels', 'images'} tensor

    def __len__(self):
        return len(self.data)

def concate_pad(tensorA, tensorB, padding_value):
    out = torch.nn.utils.rnn.pad_sequence(
        list(tensorA) + list(tensorB),
        batch_first=True,
        padding_value=padding_value)
    return out

def preference_collator_fn(instances, pad_token_id): # instances: rej_data_dict(input_ids, labels, image), win_data_dict
    rej_instances, win_instances = list(zip(*instances)) # rej_instances(input_ids, labels, image)
    rej_batch = SFT_collator_fn(rej_instances, pad_token_id) # rej_batch(input_ids, labels, attention_mask, image)
    win_batch = SFT_collator_fn(win_instances, pad_token_id)

    concatenated_input_ids = concate_pad(win_batch['input_ids'], rej_batch['input_ids'], pad_token_id)
    concatenated_labels = concate_pad(win_batch['labels'], rej_batch['labels'], -100)
    concatenated_attention_mask = concatenated_input_ids.ne(pad_token_id)

    # batch = dict(
    #     concatenated_input_ids=concatenated_input_ids,
    #     concatenated_labels=concatenated_labels,
    #     concatenated_attention_mask=concatenated_attention_mask,
    #     win_input_ids=win_batch['input_ids'],
    #     rej_input_ids=rej_batch['input_ids'],
    #     win_labels=win_batch['labels'],
    #     rej_labels=rej_batch['labels'],
    #     win_attention_mask=win_batch['attention_mask'],
    #     rej_attention_mask=rej_batch['attention_mask'],
    #     images=win_batch['images'],
    # )
    batch = dict(
        concatenated_input_ids=concatenated_input_ids,
        concatenated_labels=concatenated_labels,
        concatenated_attention_mask=concatenated_attention_mask,
        win_input_ids=win_batch['input_ids'],
        rej_input_ids=rej_batch['input_ids'],
        win_labels=win_batch['labels'],
        rej_labels=rej_batch['labels'],
        win_attention_mask=win_batch['attention_mask'],
        rej_attention_mask=rej_batch['attention_mask'],
        images=win_batch['images'],
        masked_images=win_batch['masked_images'],
    )
    return batch

class InferenceSampler(torch.utils.data.sampler.Sampler): # !!!

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        # self._rank = torch.distributed.get_rank()
        # self._world_size = torch.distributed.get_world_size()
        # -----------------------------------------------------------------------------
        # 如果分布式未初始化，则退化为单进程
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self._rank = torch.distributed.get_rank()
            self._world_size = torch.distributed.get_world_size()
        else:
            self._rank = 0
            self._world_size = 1
        # -----------------------------------------------------------------------------
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)

def get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, return_per_token_logp=False, return_all=False, tokenizer=None) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape, f'logits.shape[:-1]={logits.shape[:-1]}, labels.shape={labels.shape}'

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2,
                                   index=labels.unsqueeze(2)).squeeze(2)

    log_prob = (per_token_logps * loss_mask).sum(-1)
    average_log_prob = log_prob / loss_mask.sum(-1)

    # print("==>", labels)

    # print(per_token_logps.shape, labels.shape)
    if return_per_token_logp:
        return per_token_logps
    # print(f"per_token_prob:{per_token_logps}")
    # print(f"log_prob:{log_prob}")
    # print(f"avg_log_prob:{average_log_prob}")
    if return_all:
        return per_token_logps, log_prob, average_log_prob
    return log_prob, average_log_prob

    
def get_multimodal_sample_logps(model, dataloader, tokenizer, is_llava15=False): # 编码输入，然后得到 ref model 的对数概率 !!!
    win_logp_list = []
    rej_logp_list = []

    win_avg_logp_list = []
    rej_avg_logp_list = []

    win_per_token_logp_list = []
    rej_per_token_logp_list = []

    new_win_logp_list = []
    new_rej_logp_list = []

    new_win_avg_logp_list = []
    new_rej_avg_logp_list = []

    new_win_per_token_logp_list = []
    new_rej_per_token_logp_list = []

    with torch.inference_mode():
        idx=0
        for batch in tqdm.tqdm(dataloader):
            # print(f"Batch:\n{batch.keys()}")
            for key in ['win', 'rej']:
                input_ids = batch[f'{key}_input_ids'].cuda()
                # tokens = tokenizer.batch_decode(copy.deepcopy(input_ids))
                # print(tokens)
                labels = batch[f'{key}_labels'].cuda()
                attention_mask = batch[f'{key}_attention_mask'].cuda()

                if is_llava15: # True
                    # print("is llava15")
                    (
                        _,
                        _,
                        _,
                        _,
                        inputs_embeds,
                        labels
                    ) = model.prepare_inputs_labels_for_multimodal(
                        input_ids=input_ids,
                        position_ids=None,
                        attention_mask=None,
                        past_key_values=None,
                        labels=labels,
                        images=batch['images'].to(dtype=torch.bfloat16, device='cuda'),
                    ) # tensor -> embeds
                    output = model.forward(
                        inputs_embeds=inputs_embeds,
                        labels=None,
                    ) # embeds -> output
                else:
                    output = model(
                        input_ids=input_ids,
                        labels=labels,
                        attention_mask=attention_mask,
                        images=batch['images'].to(dtype=torch.bfloat16, device='cuda'),
                    )
                per_token_logp, log_prob, average_log_prob = get_batch_logps(output.logits, labels, return_all=True) # reference model logprob

                # print(per_token_logp.shape, input_ids.shape, labels.shape, flush=True)
                assert per_token_logp.size(1) >= input_ids.size(1) - 1
                per_token_logp = per_token_logp.tolist()
                # per_token_logp = [x[:input_ids[i].ne(tokenizer.pad_token_id).sum().item()] for i, x in enumerate(per_token_logp)]
                log_prob = log_prob.tolist()
                average_log_prob = average_log_prob.tolist()

                if key == 'win':
                    win_logp_list += log_prob
                    win_avg_logp_list += average_log_prob
                    win_per_token_logp_list += per_token_logp
                else:
                    rej_logp_list += log_prob
                    rej_avg_logp_list += average_log_prob
                    rej_per_token_logp_list += per_token_logp
            # -------------------------------------------------------- mask image --------------------------------------------------------
            for copo_key in ['', 'masked_']:
                new_input_ids = batch['win_input_ids'].cuda()
                # tokens = tokenizer.batch_decode(copy.deepcopy(input_ids))
                # print(tokens)
                new_labels = batch['win_labels'].cuda()
                new_attention_mask = batch['win_attention_mask'].cuda()

                if is_llava15: # True
                    # print("is llava15")
                    (
                        _,
                        _,
                        _,
                        _,
                        new_inputs_embeds,
                        new_labels
                    ) = model.prepare_inputs_labels_for_multimodal(
                        input_ids=new_input_ids,
                        position_ids=None,
                        attention_mask=None,
                        past_key_values=None,
                        labels=new_labels,
                        images=batch[f'{copo_key}images'].to(dtype=torch.bfloat16, device='cuda'),
                    ) # tensor -> embeds
                    new_output = model.forward(
                        inputs_embeds=new_inputs_embeds,
                        labels=None,
                    ) # embeds -> output
                else:
                    new_output = model(
                        input_ids=new_input_ids,
                        labels=new_labels,
                        attention_mask=new_attention_mask,
                        images=batch[f'{copo_key}images'].to(dtype=torch.bfloat16, device='cuda'),
                    )
                new_per_token_logp, new_log_prob, new_average_log_prob = get_batch_logps(new_output.logits, new_labels, return_all=True) # reference model logprob
                # print(f"per_token_prob:{new_per_token_logp}")
                # print(f"log_prob:{new_log_prob}")
                # print(f"avg_log_prob:{new_average_log_prob}")
                # print(per_token_logp.shape, input_ids.shape, labels.shape, flush=True)
                assert new_per_token_logp.size(1) >= new_input_ids.size(1) - 1
                new_per_token_logp = new_per_token_logp.tolist()
                # per_token_logp = [x[:input_ids[i].ne(tokenizer.pad_token_id).sum().item()] for i, x in enumerate(per_token_logp)]
                new_log_prob = new_log_prob.tolist()
                new_average_log_prob = new_average_log_prob.tolist()
                
                if copo_key == '':
                    new_win_logp_list += new_log_prob
                    new_win_avg_logp_list += new_average_log_prob
                    new_win_per_token_logp_list += new_per_token_logp
                else:
                    new_rej_logp_list += new_log_prob
                    new_rej_avg_logp_list += new_average_log_prob
                    new_rej_per_token_logp_list += new_per_token_logp

    return win_logp_list, win_avg_logp_list, win_per_token_logp_list, rej_logp_list, rej_avg_logp_list, rej_per_token_logp_list, new_win_logp_list, new_win_avg_logp_list, new_win_per_token_logp_list, new_rej_logp_list, new_rej_avg_logp_list, new_rej_per_token_logp_list

def write_logp_to_preference_parquet(origin_data, cache_file, logps, new_logps, overwrite_logps=False):
    out_data = []

    for index in range(len(logps)): # logps list[tuple]
        line = origin_data[index]
        logp_data = {}
        logp_data['dpo_logps']=logps[index]
        new_logp_data = {} # 5.7 add
        new_logp_data['dpo_new_logps']=new_logps[index] # 5.7 add
        new_line = copy.deepcopy(line)

        if 'dpo_logps' in new_line.keys():
            assert overwrite_logps, 'Found existing logp data, pass overwrite_logps=True to force overwritting'
            new_line['dpo_logps'] = json.dumps(logp_data)
            new_line['dpo_new_logps'] = json.dumps(new_logp_data)
        else:
            assert (('question' in list(new_line.keys()))
                    and ('chosen' in list(new_line.keys()))
                    and ('rejected' in list(new_line.keys()))), \
                f'Undefined data structure, expecting [Q, Win, Rej] in keys, got {new_line.keys()}'
            new_line['dpo_logps'] = json.dumps(logp_data)
            new_line['dpo_new_logps'] = json.dumps(new_logp_data)

        out_data.append(new_line)

    if torch.distributed.get_rank() == 0:
        step = 5000
        for idx, start in enumerate(range(0, len(out_data), step)):
            temp_data = out_data[start: min(start+step, len(out_data))]
            df = pd.DataFrame(temp_data)
            df.to_parquet(os.path.join(cache_file, f'RLAIF-V-Dataset-with_dpo_logp_{idx:03}-{len(temp_data)}.parquet'))

    torch.distributed.barrier()

def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # os.environ['WANDB_CACHE_DIR'] = "/NAS/fengjw/Project/RePO/.cache"
    # print(f"<<<{training_args.report_to == 'wandb'}>>>")
    
    
    data_args.data_source_names = data_args.data_source_names.split('#')
    data_args.data_source_weights = [int(x) for x in data_args.data_source_weights.split('#')]

    data_args.eval_data_source_names = data_args.eval_data_source_names.split('#') if data_args.eval_data_source_names is not None else None

    data_args.kto_win_data_source_names = data_args.kto_win_data_source_names.split('#')
    data_args.kto_win_data_source_weights = list(map(int, data_args.kto_win_data_source_weights.split('#')))
    data_args.kto_rej_data_source_names = data_args.kto_rej_data_source_names.split('#')
    data_args.kto_rej_data_source_weights = list(map(int, data_args.kto_rej_data_source_weights.split('#')))
    # print(data_args)


    # # print(model_args, '\n', data_args, '\n', training_args)
    # data = datasets.load_dataset(data_args.data_dir)['train'].cast_column("image", datasets.Image(decode=False))
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype = torch.bfloat16,
    )
    model.config.use_cache = False
    model.enable_input_require_grads()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=2048,
        padding_side="right",
        use_fast=False,
        truncation_side='right',
    )
    # for llava 1.5
    tokenizer.pad_token = tokenizer.unk_token
    conversation_lib.default_conversation = conversation_lib.conv_templates['llava_v1']
    print("conv template:", conversation_lib.default_conversation)
    # print(f"fsdp: {training_args.device}.")
    model.get_model().initialize_vision_modules(
        model_args=model_args,
        fsdp=training_args.fsdp
    )
    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16, device=training_args.device)
    data_args.image_processor = lambda x: vision_tower.image_processor(x)['pixel_values'][0]
    data_args.is_multimodal = True

    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length

    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter

    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
    
    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_projector_lr = training_args.mm_projector_lr
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    model.requires_grad_(True)
    params_no_grad = [n for n, p in model.named_parameters() if not p.requires_grad]
    # if is_main_process():
    #     print(f'No grad params are : {params_no_grad}', flush=True)

    reference_model = copy.deepcopy(model).cuda()
    multimodal_cfg=dict(
        is_multimodal=data_args.is_multimodal,
        image_token_len=data_args.image_token_len, # image_token_len
        image_folder=data_args.image_folder,
        image_aspect_ratio=data_args.image_aspect_ratio,
        use_im_start_end=getattr(data_args, 'mm_use_im_start_end', False), # use_im_start_end
        image_processor=getattr(data_args, 'image_processor', None), # img_processor
        data_source_names=getattr(data_args, 'data_source_names'),
        data_source_weights=getattr(data_args, 'data_source_weights'),
        shuffle_data=data_args.shuffle_data
    )
    # print(multimodal_cfg)

    data_path = [file for file in os.listdir(data_args.data_dir) if file.endswith('.parquet') and 'logp' in file]
    data_path_self = data_args.data_dir
    # hf_data = datasets.load_dataset(data_path_self)['train'].cast_column("image", datasets.Image(decode=False))
    hf_data_tmp = datasets.load_dataset(data_path_self)['train']
    hf_data_shuffled = hf_data_tmp.shuffle(seed=42)
    hf_data = hf_data_shuffled.cast_column("image", datasets.Image(decode=False))
    
    image_token_len = multimodal_cfg['image_token_len']
    img_processor = multimodal_cfg['image_processor']
    use_im_start_end = multimodal_cfg['use_im_start_end']
    # model = model.to(dtype=torch.bfloat16, device='cuda') # ***

    dataset = PreferenceInferenceDataset( # tensor 数据集
        tokenizer=tokenizer,
        data = hf_data,
        image_token_len=image_token_len,
        img_processor=img_processor,
        use_im_start_end=use_im_start_end
    )
    # print(f"Data: {type(dataset[0])}") # tuple: (rej_data_dict(input_ids, labels, image), win_data_dict(input_ids, labels, image))
    collate_fn = partial(preference_collator_fn, pad_token_id=tokenizer.pad_token_id) #
    dataloader = torch_data.DataLoader(dataset, batch_size=1, collate_fn=collate_fn,
                                       num_workers=0, shuffle=False, sampler=InferenceSampler(len(dataset)))
    outputs = get_multimodal_sample_logps(reference_model, dataloader, tokenizer, is_llava15=True) # 计算 ref model 的对数概率
    # print(f"!!!!{type(outputs)}!!!!")
    # outputs_mask = get_multimodal_sample_logps_mask(reference_model, dataloader, tokenizer, is_llava15=True)

    world_size = torch.distributed.get_world_size()
    merged_outputs = [[None for _ in range(world_size)] for i in range(len(outputs))]
    for i in range(len(outputs)):
        torch.distributed.all_gather_object(merged_outputs[i], outputs[i])
        merged_outputs[i] = [_ for _ in itertools.chain.from_iterable(merged_outputs[i])]


    win_logp_list, win_avg_logp_list, win_per_token_logp_list, rej_logp_list, rej_avg_logp_list, rej_per_token_logp_list, new_win_logp_list, new_win_avg_logp_list, new_win_per_token_logp_list, new_rej_logp_list, new_rej_avg_logp_list, new_rej_per_token_logp_list\
        = merged_outputs

    logps = list(zip(win_logp_list, win_avg_logp_list, win_per_token_logp_list, rej_logp_list, rej_avg_logp_list, rej_per_token_logp_list))

    new_logps = list(zip(new_win_logp_list, new_win_avg_logp_list, new_win_per_token_logp_list, new_rej_logp_list, new_rej_avg_logp_list, new_rej_per_token_logp_list))
    # print(f"{new_logps}")
    write_logp_to_preference_parquet(dataset.data, data_path_self, logps, new_logps, overwrite_logps=False)

    torch.distributed.barrier()
    
if __name__ == "__main__":
    main()