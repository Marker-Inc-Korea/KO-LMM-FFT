from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

if __name__ == "__main__":
    login(token='...your_token_name...')
    final_save_folder = '...checkpoint...'
    repo_name = '...repo_name...'

    model = AutoModelForCausalLM.from_pretrained(final_save_folder, 
                                                 torch_dtype=torch.float16,
                                                 trust_remote_code=True)
    model.push_to_hub(repo_name, token=True, safe_serialization=True)
    #tokenizer = AutoTokenizer.from_pretrained(final_save_folder)
    #tokenizer.push_to_hub(repo_name, token=True)