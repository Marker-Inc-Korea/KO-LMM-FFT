# KO-LMM-FFT (full fine tuning)
Korean Large MultiModal FFT Code
  
🍚Gukbap-LMM Series models🍚  
**HumanF-MarkrAI/Gukbap-Gemma2-9B-VL🍚:** [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/HumanF-MarkrAI/Gukbap-Gemma2-9B-VL)   
**HumanF-MarkrAI/Gukbap-Qwen2-34B-VL🍚:** [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/HumanF-MarkrAI/Gukbap-Qwen2-34B-VL)  
  
# LMM Training⭐
In training, the datasets are private.
  
## Gukbap-Gemma2-9B-VL🍚
```sh
# ovis_fullfine.sh

python ovis_fullfine.py \
    --base_model AIDC-AI/Ovis1.6-Gemma2-9B \
    --data-path  MarkrAI/Markr_WizardLM_train_ver4 \
    --val_data_path MarkrAI/WizardLM_Evol_valid \
    --output_dir ...output_path...  \
    --batch_size 64 \
    --micro_batch_size 1 \
    --num_epochs 10 \
    --learning_rate 1e-5 \
    --cutoff_len 2048 \
    --val_flag False \
    --add_eos_token True \
    --lr_scheduler 'cosine'
```
> transformers==4.44.2 (recommend)
  
## Gukbap-Qwen2-34B-VL🍚
```sh
# ovis_fullfine.sh

python ovis_fullfine.py \
    --base_model AIDC-AI/Ovis2-34B \
    --data-path  MarkrAI/Markr_WizardLM_train_ver4 \
    --val_data_path MarkrAI/WizardLM_Evol_valid \
    --output_dir ...output_path...  \
    --batch_size 128 \
    --micro_batch_size 1 \
    --num_epochs 10 \
    --learning_rate 2e-5 \
    --cutoff_len 2048 \
    --val_flag False \
    --add_eos_token True \
    --lr_scheduler 'cosine'
```
> transformers==4.46.2 (recommend)

# Blog🔥
[Gukbap=LMM blog🔥]().
  
# Result🤗

  
# Citation
[VLMEvalKit](https://github.com/open-compass/VLMEvalKit).  
[AIDC-AI](https://huggingface.co/AIDC-AI).  
[NCSoft](https://huggingface.co/NCSOFT).
