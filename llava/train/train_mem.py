# from llava.train.train_dpo import train
import sys
import os
project_root = os.path.abspath(os.path.join("/data/fengjw/Project/DAMA/DAMA/llava", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# from llava.train.train_repo import train
# from llava.train.train_repo_new import train    
# from llava.train.train_dpo_new import train

# from llava.train.train_repo_527 import train
from llava.train.train_repo_611 import train
# from llava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
# replace_llama_attn_with_flash_attn()

if __name__ == "__main__":
    train()
