# KO-LMM-FFT (full fine tuning)
🍚Gukbap-LMM Series models🍚  
**HumanF-MarkrAI/Gukbap-Gemma2-9B-VL🍚:** [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/HumanF-MarkrAI/Gukbap-Gemma2-9B-VL)   
**HumanF-MarkrAI/Gukbap-Qwen2-34B-VL🍚:** [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/HumanF-MarkrAI/Gukbap-Qwen2-34B-VL)  
  
# LMM Training⭐
In training, the datasets are private.  
Our dataset consists of **text only**!!!  
   
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
[Gukbap-LMM blog🔥]().
  
# Result🤗

## English VLM Evaluation
| Model | 0-shot | 5-shot | 10-shot | 50-shot |
| --- | --- | --- | --- | --- |
| HumanF-MarkrAI/Gukbap-Qwen2-34B-VL | 0.5247 | 0.5260 | 0.5278 | 0.5427 |
| HumanF-MarkrAI/Gukbap-Gemma2-9B-VL | 0.5707 | 0.5830 | 0.5670 | 0.5787 |
| [Polyglot-ko-5.8b](https://huggingface.co/EleutherAI/polyglot-ko-5.8b) | 0.5976 | 0.5998 | 0.5979 | 0.6208 |
| [Polyglot-ko-12.8b](https://huggingface.co/EleutherAI/polyglot-ko-12.8b) | 0.5954 | 0.6306 | 0.6098 | 0.6118 |
| [Llama-2-Ko-7b 20B](https://huggingface.co/beomi/llama-2-ko-7b) | 0.4518 | 0.4668 | 0.4726 | 0.4828 |
| [Llama-2-Ko-7b 40B](https://huggingface.co/beomi/llama-2-ko-7b) | 0.4562 | 0.4657 | 0.4698 | 0.4774 |   
| **KO-platypus2-7B-EX(ours)** | 0.4571 | 0.4461 | 0.4371 | 0.4525 | 

## Korean VLM Evaluation
| Model | K-MMBench | K-MMStar| K-DTCBench | K-LLAVA-W | Average |
| --- | --- | --- | --- | --- | --- |
| **HumanF-MarkrAI/Gukbap-Qwen2-34B-VL🍚** | 89.10 | 68.13 | 77.08 | **69.00** | **75.83** |
| **HumanF-MarkrAI/Gukbap-Gemma2-9B-VL🍚** | 80.16 | 54.20 | 52.92 | 63.83 | 62.78 |
| Ovis2-34B | **89.56** | **68.27** | 76.25 | 53.67 | 71.94 |
| Ovis1.6-Gemma2-9B | 52.46 | 50.40 | 47.08 | 55.67 | 51.40 |
| VARCO-VISION-14B | 87.16 | 58.13 | **85.42** | 51.17 | 70.47 | 
| llama-3.2-Korean-Bllossom-AICA-5B	 | 26.01 | 21.60 | 17.08 | 45.33 | 27.51 |   
> [Test code]().
  
# Citation
[VLMEvalKit](https://github.com/open-compass/VLMEvalKit).  
[AIDC-AI](https://huggingface.co/AIDC-AI).  
[NCSoft](https://huggingface.co/NCSOFT).
