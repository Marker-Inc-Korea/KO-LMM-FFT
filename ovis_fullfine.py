import os
import sys
from typing import List

import fire
import torch
import transformers

from datasets import load_dataset
from transformers import Trainer
from transformers import AutoModelForCausalLM

from torch.nn import functional as F

from utils.prompter import ConversationDataset, DataCollatorForMultimodalDataset

#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def train(
    # model/data params
    base_model: str = "", 
    data_path: str = "",
    val_data_path: str = "",
    output_dir: str = "",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 8,
    num_epochs: int = 1,
    learning_rate: float = 3e-4,
    cutoff_len: int = 4096,
    val_flag: bool = False,
    lr_scheduler: str = "cosine",
    # llm hyperparams
    add_eos_token: bool = False,
    resume_from_checkpoint = None,  # either training checkpoint or final adapter
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"val_data_path: {val_data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_flag: {val_flag}\n"
            f"lr_scheduler: {lr_scheduler}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    ## Huggingface Login
    from huggingface_hub import login
    login(token='...your token...')
    gradient_accumulation_steps = batch_size // micro_batch_size


    ## ddp or device map setting
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1 # world_size = 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} # auto
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
        print("gradient_accumulation_steps: ", gradient_accumulation_steps)
    print("############ DDP:", ddp) # if use torchrun, it is true
    print("############ N GPU:", torch.cuda.device_count())


    ## Model loading
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map=device_map,
        cache_dir="/data/cache/",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        multimodal_max_length=cutoff_len # 2048
    )
    model.requires_grad_(False)

    print(type(model))
    print(model)
    model.get_llm().config.use_cache = False
    model.config.use_cache = False
    
    
    ## select train modules
    text_tokenizer = model.get_text_tokenizer()
    visual_tokenizer = model.get_visual_tokenizer()
    model.get_llm().requires_grad_(True) # Stage 3
    #model.requires_grad_(True)
    
    #for name, param in model.named_parameters():
    #    if param.requires_grad:
    #        print(name)

    
    ## data path
    print("================== private dataset")
    train_data = load_dataset(data_path, token=True)
    if val_flag:
        val_data = load_dataset(val_data_path, token=True)
        
    train_data = ConversationDataset(train_data['train'], cutoff_len, text_tokenizer, visual_tokenizer, model, add_eos_token)
    if val_flag:
        val_data = ConversationDataset(val_data['validation'], cutoff_len, text_tokenizer, visual_tokenizer, model, add_eos_token)
    else:
        val_data = None
    
    data_module = dict(
        train_dataset = train_data,
        eval_dataset = val_data,
        data_collator = DataCollatorForMultimodalDataset(text_tokenizer)
    )
    
    
    ## training module (ovis paper)
    training_args = transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=0.05,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=1,
            optim="paged_adamw_32bit", # paged_adamw_32bit, adamw_torch
            max_grad_norm = 1.0,
            weight_decay = 0,
            evaluation_strategy="steps" if val_flag else "no",
            save_strategy="steps",
            eval_steps = 14 if val_flag else None,
            save_steps = 14, # oringinal: 1000
            save_safetensors=True, # 일단 false로
            lr_scheduler_type=lr_scheduler,
            output_dir=output_dir,
            save_total_limit=15,
            load_best_model_at_end=True if val_flag else False,
            ddp_find_unused_parameters=False, #if ddp else None,
            group_by_length=False
        )
    
    ## trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        **data_module
    )
    
    
    ## Training
    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    
    if resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train()
    trainer.save_state()
    
    
    ## Save model
    model.get_llm().config.use_cache = True
    model.config.use_cache = True
    trainer.save_model()


if __name__ == "__main__":
    torch.cuda.empty_cache() 
    fire.Fire(train)

