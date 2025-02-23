#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python ovis_fullfine.py \
    --base_model AIDC-AI/Ovis2-34B \
    --data-path  ...train_path... \
    --val_data_path ...valid_path... \
    --output_dir ...your_path...  \
    --batch_size 128 \
    --micro_batch_size 1 \
    --num_epochs 10 \
    --learning_rate 2e-5 \
    --cutoff_len 2048 \
    --val_flag False \
    --add_eos_token True \
    --lr_scheduler 'cosine'