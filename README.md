# KO-LMM-FFT (full fine tuning)
🍚Gukbap-LMM Series models🍚  
**HumanF-MarkrAI/Gukbap-Gemma2-9B-VL🍚:** [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/HumanF-MarkrAI/Gukbap-Gemma2-9B-VL)   
**HumanF-MarkrAI/Gukbap-Qwen2-34B-VL🍚:** [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/HumanF-MarkrAI/Gukbap-Qwen2-34B-VL)  
  
# LMM Training⭐
In training, the datasets are private.  
Our dataset consists of **text only**!!!  
```sh
# Quick Start
sh ovis_fullfine.sh
```
    
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

# Blog (detailed introduction for Gukbap-LMM)🔥
[Gukbap-LMM blog🔥](https://kyujinpy.tistory.com/169).
  
# Result🤗

## English VLM Evaluation
| Model | MMStar | MathVista | HallusionBench | AI2D | OCRBench | MMVet | MMBench_V11 | AVG |
|:---------:|:-----:|:------:|:-----:|:-----:|:----:|:-----:|:-----:|:-----:|
| Step-1o (closed model; SOTA) | **69.3** | 74.7 | **89.1** | 55.8 | **92.6** | **82.8** | **87.3** | **78.8** |
| **HumanF-MarkrAI/Gukbap-Qwen2-34B-VL🍚** | **69.33** | **77.40** | 55.66 | 88.31 | 84.7 | 74.13 | 86.53 | 76.58 |
| **HumanF-MarkrAI/Gukbap-Gemma2-9B-VL🍚** | 62.13 | 66.00 | 84.49 | 53.01 | 82.80 | 63.90 | 82.20 | 70.65 |
| Ovis2-34B (Open) | 69.2 | 76.1 | 58.8 | **88.3** | 89.4 | 77.1 | 86.5 | 77.9 |
| Ovis1.6-Gemma2-9B (Open) | 62.00 | 67.10 | 84.42 | 51.96 | 82.60 | 64.68 | 82.20 | 70.71 |
| LLaVA-OneVision-72B | 65.8 | 68.4 | 47.9 | 86.2 | 74.1| 60.6 | 84.5 | 69.6 |
| VARCO-VISION-14B (NCSoft) | 64.1 | 67.6 | 46.8 | 83.9 | 81.5 | 53.0 | 81.2 | 68.3 |
| GPT-4o-mini-20240718 | 54.8 | 52.4 | 46.1 | 77.8 | 78.5 | 66.9 | 76.0 | 64.6 |
> Using [VLMEvalKit](https://github.com/open-compass/VLMEvalKit).
  
## Korean VLM Evaluation
| Model | K-MMBench | K-MMStar| K-DTCBench | K-LLAVA-W | Average |
| --- | --- | --- | --- | --- | --- |
| **HumanF-MarkrAI/Gukbap-Qwen2-34B-VL🍚** | 89.10 | 68.13 | 77.08 | **69.00** | **75.83** |
| **HumanF-MarkrAI/Gukbap-Gemma2-9B-VL🍚** | 80.16 | 54.20 | 52.92 | 63.83 | 62.78 |
| Ovis2-34B | **89.56** | **68.27** | 76.25 | 53.67 | 71.94 |
| Ovis1.6-Gemma2-9B | 52.46 | 50.40 | 47.08 | 55.67 | 51.40 |
| VARCO-VISION-14B | 87.16 | 58.13 | **85.42** | 51.17 | 70.47 | 
| llama-3.2-Korean-Bllossom-AICA-5B	 | 26.01 | 21.60 | 17.08 | 45.33 | 27.51 |   
> [Test code (ours)](https://github.com/Marker-Inc-Korea/KoVLMEval).
   
# Citation
[VLMEvalKit](https://github.com/open-compass/VLMEvalKit).  
[AIDC-AI](https://huggingface.co/AIDC-AI).  
[NCSoft](https://huggingface.co/NCSOFT).
